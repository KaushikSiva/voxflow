# ASR Sidecar

Local FastAPI sidecar for realtime Tamil ASR with Voxtral + LoRA on Apple MPS.

## Run

```bash
cd services/asr-sidecar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /healthz`
- `POST /session/start`
- `WS /ws/{session_id}`
- `POST /transcribe/chunk`
- `POST /transcribe/finalize`
- `POST /session/close`
