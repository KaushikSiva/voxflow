#!/usr/bin/env python3
"""
step7_infer_voxtral_lora_transcribe_fixA.py

Fix A (recommended): Use Voxtral "Transcription Mode" via
processor.apply_transcription_request(...) instead of trying to hand-roll
[AUDIO] tokens / apply_chat_template logic.

Works with:
  - mistralai/Voxtral-Mini-3B-2507
  - a PEFT LoRA adapter dir (e.g. runs/voxtral_lora_full)

JSONL formats supported:
  A) {"audio": "...", "answer": "..."}  (your earlier format)
  B) {"audio_path": "...", "text": "..."}  (IndicVoices style)
  C) {"audio_path": "...", "transcript": "..."}  (alt)

Usage:
  python3 step7_infer_voxtral_lora_transcribe_fixA.py \
    --adapter runs/voxtral_lora_full \
    --jsonl data/dev_flac.jsonl \
    --limit 20 \
    --language ta \
    --max_new_tokens 256

Or single files:
  python3 step7_infer_voxtral_lora_transcribe_fixA.py \
    --adapter runs/voxtral_lora_full \
    --audio data/flac/xxx.flac data/flac/yyy.flac \
    --language ta
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from peft import PeftModel


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSON on line {ln}: {e}") from e
            items.append(ex)
            if limit is not None and len(items) >= limit:
                break
    return items


def pick_audio_and_ref(ex: Dict[str, Any], jsonl_dir: str) -> Tuple[str, str]:
    """
    Normalize the various dataset schemas to:
      audio_path, ref_text
    """
    audio = ex.get("audio") or ex.get("audio_path") or ex.get("path") or ex.get("wav")
    if audio is None:
        raise ValueError("JSONL item missing audio/audio_path/path/wav key")

    # If relative, make it relative to the JSONL directory (common gotcha)
    if isinstance(audio, str) and not os.path.isabs(audio):
        audio = os.path.join(jsonl_dir, audio)

    ref = ex.get("answer") or ex.get("text") or ex.get("transcript") or ""
    if ref is None:
        ref = ""
    return audio, ref


@torch.no_grad()
def transcribe_one(
    *,
    processor: Any,
    model: Any,
    base_model_id: str,
    audio_path: str,
    language: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    """
    Voxtral transcription-mode request (recommended by HF docs):
      inputs = processor.apply_transcription_request(language="en", audio=..., model_id=repo_id)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    # Build the "transcription request" inputs
    if language:
        inputs = processor.apply_transcription_request(
            language=language,
            audio=audio_path,
            model_id=base_model_id,
        )
    else:
        inputs = processor.apply_transcription_request(
            audio=audio_path,
            model_id=base_model_id,
        )

    # Move to device/dtype
    inputs = inputs.to(device, dtype=dtype)

    # Generate
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,  # avoid "temperature ignored" warnings for greedy
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # Only decode the newly generated continuation
    prompt_len = inputs.input_ids.shape[1]
    gen_ids = out_ids[:, prompt_len:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--adapter", type=str, required=True, help="LoRA adapter dir (contains adapter_model.safetensors)")
    ap.add_argument("--jsonl", type=str, default=None, help="JSONL with audio + ref text")
    ap.add_argument("--audio", type=str, nargs="*", default=None, help="One or more audio file paths")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--language", type=str, default="ta", help="Set known language for better accuracy; use '' to disable")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if not args.jsonl and not args.audio:
        raise SystemExit("Provide either --jsonl or --audio")

    language = args.language.strip() or None

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.base)

    print("Loading base model...")
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.base,
        device_map="auto" if device.type == "cuda" else None,
        dtype=dtype if device.type == "cuda" else None,
    )

    print("Loading adapter:", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Show audio token info (sanity)
    audio_id = getattr(model.config, "audio_token_id", None)
    if audio_id is not None:
        try:
            audio_tok = processor.tokenizer.convert_ids_to_tokens(audio_id)
        except Exception:
            audio_tok = "<unknown>"
        print("audio_token_id:", audio_id, "audio_token_str:", audio_tok)

    # Build eval list
    pairs: List[Tuple[str, str]] = []
    if args.jsonl:
        data = read_jsonl(args.jsonl, limit=args.limit)
        base_dir = os.path.dirname(os.path.abspath(args.jsonl))
        for ex in data:
            a, ref = pick_audio_and_ref(ex, base_dir)
            pairs.append((a, ref))
    else:
        assert args.audio is not None
        for a in args.audio[: args.limit]:
            pairs.append((a, ""))

    # Run
    for i, (audio_path, ref) in enumerate(pairs):
        print(f"[{i}] audio: {audio_path}")
        if ref:
            print("REF:", ref)

        hyp = transcribe_one(
            processor=processor,
            model=model,
            base_model_id=args.base,
            audio_path=audio_path,
            language=language,
            device=device,
            dtype=dtype if device.type == "cuda" else torch.float32,
            max_new_tokens=args.max_new_tokens,
        )

        print("HYP:", hyp)
        print("-" * 80)


if __name__ == "__main__":
    main()
