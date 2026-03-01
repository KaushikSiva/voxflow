# Tamil Voice Flow

This repository now includes a Wispr-style Tamil dictation pipeline:

- `services/asr-sidecar`: FastAPI ASR sidecar (local Voxtral+LoRA on MPS)
- `apps/web`: SvelteKit app with shadcn-style UI for realtime transcription + Tanglish cleanup

## Start sidecar

```bash
cd services/asr-sidecar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Start web app

```bash
cd apps/web
npm install
cp .env.example .env
npm run dev
```

Then open `http://127.0.0.1:5173`.
