import asyncio
import logging
import re
import subprocess
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from text_merge import merge_partials

BASE = 'mistralai/Voxtral-Mini-3B-2507'
ADAPTER = 'kaushiksiva/voxtral-mini-3b-tamil-lora'
SESSION_ROOT = Path('/tmp/tamil_voice_sessions')
SESSION_TTL_SECONDS = 60 * 5


@dataclass
class SessionState:
  session_id: str
  created_at: float
  updated_at: float
  chunk_paths: List[Path] = field(default_factory=list)
  partials: List[str] = field(default_factory=list)
  latest_partial: str = ''
  merged_text: str = ''
  latest_tamil_partial: str = ''
  clients: Set[WebSocket] = field(default_factory=set)


class FinalizeRequest(BaseModel):
  session_id: str


class CloseRequest(BaseModel):
  session_id: str


app = FastAPI(title='Tamil ASR Sidecar', version='0.1.0')
app.add_middleware(
  CORSMiddleware,
  allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
  allow_methods=['*'],
  allow_headers=['*']
)

sessions: Dict[str, SessionState] = {}
sessions_lock = asyncio.Lock()
inference_lock = asyncio.Lock()
cleanup_task: Optional[asyncio.Task] = None

processor = None
model = None
device = None
dtype = None
logger = logging.getLogger("asr-sidecar")


def load_model() -> None:
  global processor, model, device, dtype

  if not torch.backends.mps.is_available():
    raise RuntimeError('MPS is required for this sidecar on macOS.')

  device = torch.device('mps')
  dtype = torch.float16

  processor = AutoProcessor.from_pretrained(BASE)
  base_model = VoxtralForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=dtype,
    device_map=None,
  ).eval()
  model_with_adapter = PeftModel.from_pretrained(base_model, ADAPTER).eval()
  model_with_adapter.to(device)
  model_with_adapter.config.use_cache = True
  model = model_with_adapter


def _transcribe_file(audio_path: str, max_new_tokens: int = 48) -> str:
  assert processor is not None
  assert model is not None
  assert device is not None
  assert dtype is not None

  inputs = processor.apply_transcription_request(
    language='ta',
    audio=audio_path,
    model_id=BASE,
  )

  for key, value in list(inputs.items()):
    if not torch.is_tensor(value):
      continue
    if key == 'input_ids':
      inputs[key] = value.to(device)
    elif key == 'attention_mask':
      inputs[key] = value.to(device)
    else:
      inputs[key] = value.to(device, dtype=dtype)

  with torch.inference_mode():
    out_ids = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      repetition_penalty=1.15,
      no_repeat_ngram_size=3,
      pad_token_id=processor.tokenizer.eos_token_id,
    )

  prompt_len = inputs['input_ids'].shape[1]
  text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
  return text


def _prepare_audio_for_asr(chunk_path: Path) -> Path:
  """
  Convert browser chunk audio (typically webm/opus) to wav 16k mono for robust decoding.
  """
  if chunk_path.suffix.lower() == '.wav':
    return chunk_path

  ffmpeg = shutil.which('ffmpeg')
  if not ffmpeg:
    raise RuntimeError(
      "ffmpeg not found. Install it (brew install ffmpeg) to decode browser audio chunks."
    )

  wav_path = chunk_path.with_suffix('.wav')
  cmd = [
    ffmpeg,
    '-y',
    '-loglevel',
    'error',
    '-i',
    str(chunk_path),
    '-ac',
    '1',
    '-ar',
    '16000',
    str(wav_path),
  ]
  subprocess.run(cmd, check=True)
  return wav_path


def _should_accept_partial(text: str) -> bool:
  t = (text or '').strip()
  if not t:
    return False
  # Ignore very short noise fragments.
  return len(t.split()) >= 2 or len(t) >= 8


def _is_tamil_dominant(text: str) -> bool:
  t = (text or '').strip()
  if not t:
    return False
  tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", t))
  latin_chars = len(re.findall(r"[A-Za-z]", t))
  if tamil_chars == 0:
    return False
  return tamil_chars >= 4 and tamil_chars >= (latin_chars * 0.6)


def _is_known_hallucination(text: str) -> bool:
  lower = (text or '').lower()
  blocked = [
    "i'm sorry",
    "i am sorry",
    "could you please provide more context",
    "i didn't understand your question",
    "clarify your request",
  ]
  return any(token in lower for token in blocked)


async def broadcast_partial(session: SessionState, partial_text: str) -> None:
  if not session.clients:
    return

  stale: List[WebSocket] = []
  for ws in session.clients:
    try:
      await ws.send_json({'partialTamil': partial_text, 'sessionId': session.session_id})
    except Exception:
      stale.append(ws)

  for ws in stale:
    session.clients.discard(ws)


async def cleanup_expired_sessions_loop() -> None:
  while True:
    await asyncio.sleep(30)
    now = time.time()
    to_delete: List[str] = []

    async with sessions_lock:
      for session_id, state in sessions.items():
        if now - state.updated_at > SESSION_TTL_SECONDS:
          to_delete.append(session_id)

      for session_id in to_delete:
        state = sessions.pop(session_id, None)
        if state:
          for ws in list(state.clients):
            await ws.close(code=1000)
          session_dir = SESSION_ROOT / session_id
          shutil.rmtree(session_dir, ignore_errors=True)


@app.on_event('startup')
async def startup() -> None:
  SESSION_ROOT.mkdir(parents=True, exist_ok=True)
  load_model()

  global cleanup_task
  cleanup_task = asyncio.create_task(cleanup_expired_sessions_loop())


@app.on_event('shutdown')
async def shutdown() -> None:
  global cleanup_task
  if cleanup_task:
    cleanup_task.cancel()
    cleanup_task = None


@app.get('/healthz')
async def healthz() -> dict:
  return {
    'ok': True,
    'device': str(device),
    'dtype': str(dtype),
    'sessions': len(sessions),
    'model_loaded': model is not None,
  }


@app.post('/session/start')
async def session_start() -> dict:
  session_id = uuid.uuid4().hex
  now = time.time()
  state = SessionState(session_id=session_id, created_at=now, updated_at=now)

  async with sessions_lock:
    sessions[session_id] = state

  (SESSION_ROOT / session_id).mkdir(parents=True, exist_ok=True)

  return {'session_id': session_id, 'ws_url': f'ws://127.0.0.1:8000/ws/{session_id}'}


@app.websocket('/ws/{session_id}')
async def session_ws(websocket: WebSocket, session_id: str) -> None:
  await websocket.accept()

  async with sessions_lock:
    state = sessions.get(session_id)
    if not state:
      await websocket.send_json({'error': 'Unknown session'})
      await websocket.close(code=1008)
      return
    state.clients.add(websocket)

  try:
    while True:
      await websocket.receive_text()
  except WebSocketDisconnect:
    pass
  finally:
    async with sessions_lock:
      state = sessions.get(session_id)
      if state:
        state.clients.discard(websocket)


@app.post('/transcribe/chunk')
async def transcribe_chunk(
  audio_file: UploadFile = File(...),
  session_id: str = Form(...),
  chunk_index: int = Form(...),
  is_final_chunk: int = Form(0),
) -> dict:
  final_chunk = int(is_final_chunk) == 1

  async with sessions_lock:
    session = sessions.get(session_id)

  if not session:
    raise HTTPException(status_code=404, detail='Session not found')

  try:
    ext = Path(audio_file.filename or '').suffix or '.webm'
    chunk_path = SESSION_ROOT / session_id / f'chunk_{chunk_index:06d}{ext}'
    content = await audio_file.read()
    chunk_path.write_bytes(content)
    prepared_path = _prepare_audio_for_asr(chunk_path)

    async with inference_lock:
      try:
        token_budget = 128 if final_chunk else 48
        partial = await asyncio.to_thread(_transcribe_file, str(prepared_path), token_budget)
      except Exception as exc:
        session.updated_at = time.time()
        merged = session.merged_text or session.latest_partial
        logger.exception("ASR inference failed for chunk %s", chunk_index)
        return {
          'partial_tamil': merged,
          'chunk_index': chunk_index,
          'warning': f'ASR failed for chunk {chunk_index}: {exc}'
        }

    session.chunk_paths.append(chunk_path)
    normalized_partial = (partial or '').strip()
    if normalized_partial:
      session.latest_partial = normalized_partial
      if _is_known_hallucination(normalized_partial):
        logger.warning("Dropped hallucinated partial for chunk %s", chunk_index)
      elif _should_accept_partial(normalized_partial):
        is_tamil = _is_tamil_dominant(normalized_partial)
        if is_tamil:
          session.latest_tamil_partial = normalized_partial

        chosen_partial = session.latest_tamil_partial or normalized_partial
        if final_chunk:
          session.partials = [chosen_partial]
          session.merged_text = chosen_partial
        else:
          if not session.partials or session.partials[-1] != chosen_partial:
            session.partials.append(chosen_partial)
            if len(session.partials) > 40:
              session.partials = session.partials[-40:]
          session.merged_text = merge_partials(session.partials)
    session.updated_at = time.time()

    merged = session.merged_text or session.latest_partial
    await broadcast_partial(session, merged)

    return {'partial_tamil': merged, 'chunk_index': chunk_index}
  except Exception as exc:
    session.updated_at = time.time()
    merged = session.merged_text or session.latest_partial
    logger.error("Unhandled chunk endpoint error for chunk %s: %s", chunk_index, exc)
    logger.error(traceback.format_exc())
    return {
      'partial_tamil': merged,
      'chunk_index': chunk_index,
      'warning': f'Chunk pipeline failed for chunk {chunk_index}: {exc}'
    }


@app.post('/transcribe/finalize')
async def transcribe_finalize(body: FinalizeRequest) -> dict:
  async with sessions_lock:
    session = sessions.get(body.session_id)

  if not session:
    raise HTTPException(status_code=404, detail='Session not found')

  merged = session.merged_text or session.latest_partial
  session.updated_at = time.time()
  return {'raw_tamil': merged, 'chunk_count': len(session.chunk_paths)}


@app.post('/session/close')
async def session_close(body: CloseRequest) -> dict:
  async with sessions_lock:
    session = sessions.pop(body.session_id, None)

  if not session:
    return {'ok': True, 'closed': False}

  for ws in list(session.clients):
    await ws.close(code=1000)

  shutil.rmtree(SESSION_ROOT / body.session_id, ignore_errors=True)
  return {'ok': True, 'closed': True}
