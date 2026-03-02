# VoxFlow Web (SvelteKit + shadcn-style UI)

## Features

- Push-to-talk realtime dictation UI.
- Live `Raw Tamil` updates from ASR WebSocket stream.
- Finalize flow that generates `Tanglish` and `Clean Tanglish`.
- Copy actions + error/status indicators.

## Required env

Copy `.env.example` to `.env`:

```bash
ASR_SIDECAR_URL=http://127.0.0.1:8000
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## Run

```bash
cd apps/web
npm install
npm run dev
```

Open: `http://127.0.0.1:5173`

## API routes

- `POST /api/session/start`
- `POST /api/session/:id/chunk`
- `POST /api/session/:id/finalize`
- `POST /api/session/:id/close`
- `POST /api/text/process`
- `GET /api/health`
