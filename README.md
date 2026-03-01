# Tamil Voice Flow

This repository now includes a Wispr-style Tamil dictation pipeline:

- `services/asr-sidecar`: FastAPI ASR sidecar (local Voxtral+LoRA on MPS)
- `apps/web`: SvelteKit app with shadcn-style UI for realtime transcription + Tanglish cleanup
- `training`: step-by-step Tamil ASR data prep + QLoRA training scripts
- `inference`: inference/evaluation scripts

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

## Training (step-by-step)

Training scripts are organized as one file per step in `training/`:

1. `step_01_download_parquets.py`
2. `step_02_export_flac_all.py`
3. `step_03_split_train_dev.py`
4. `step_04_make_smoketest_subsets.py`
5. `step_05_train_voxtral_qlora.py`
6. `step_06_inference_check_gpu.py`

Use the full instructions and commands in:

- `training/README.md`
