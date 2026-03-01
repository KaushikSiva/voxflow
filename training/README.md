# Training Pipeline

This folder contains one script per training step.

## Step files

1. `step_01_download_parquets.py`
2. `step_02_export_flac_all.py`
3. `step_03_split_train_dev.py`
4. `step_04_make_smoketest_subsets.py`
5. `step_05_train_voxtral_qlora.py`
6. `step_06_inference_check_gpu.py`
7. `step_07_backfill_wandb.py`

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U huggingface_hub pyarrow
pip install -U transformers peft bitsandbytes accelerate soundfile
pip install -U "mistral-common[audio]"
```

Optional system dependency (recommended for audio):

```bash
# macOS
brew install ffmpeg
```

## Step-by-step

### 1) Download Tamil parquet shards

```bash
python training/step_01_download_parquets.py
```

Expected output: dataset shards under `~/datasets/indicvoices_parquet`.

### 2) Export FLAC + build manifest

```bash
python training/step_02_export_flac_all.py
```

Expected outputs:
- `data/flac/*.flac`
- `data/all_flac.jsonl`

### 3) Split train/dev by speaker

```bash
python training/step_03_split_train_dev.py
```

Expected outputs:
- `data/train_flac.jsonl`
- `data/dev_flac.jsonl`

### 4) Create smoke-test subsets

```bash
python training/step_04_make_smoketest_subsets.py
```

Expected outputs:
- `data/train_flac_2k.jsonl`
- `data/dev_flac_200.jsonl`

### 5) Train Voxtral QLoRA

Run smoke training first:

```bash
python training/step_05_train_voxtral_qlora.py \
  --train_jsonl data/train_flac_2k.jsonl \
  --out_dir runs/voxtral_lora_smoke \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8
```

Then full training:

```bash
python training/step_05_train_voxtral_qlora.py \
  --train_jsonl data/train_flac.jsonl \
  --out_dir runs/voxtral_lora_full \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8
```

Optional: enable W&B tracking (hackathon step 3 style):

```bash
python training/step_05_train_voxtral_qlora.py \
  --train_jsonl data/train_flac.jsonl \
  --out_dir runs/voxtral_lora_full \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 8 \
  --wandb \
  --wandb_project mistral-hackathon \
  --wandb_run_name voxtral-ta-run-01
```

When `--wandb` is enabled, the trainer will:
- initialize a W&B run
- log training metrics (`loss`, `lr`, `label_frac`, `steps/s`)
- upload `out_dir` as a model artifact at the end

### 6) GPU inference check / evaluation

```bash
python training/step_06_inference_check_gpu.py \
  --adapter runs/voxtral_lora_full \
  --jsonl data/dev_flac.jsonl \
  --limit 20 \
  --language ta \
  --max_new_tokens 256
```

### 7) Retrospective W&B backfill (existing model/runs)

For your adapter on Hugging Face (`kaushiksiva/voxtral-mini-3b-tamil-lora`):

```bash
python training/step_07_backfill_wandb.py \
  --wandb_project mistral-hackathon \
  --wandb_run_name voxtral-ta-retro \
  --model_repo kaushiksiva/voxtral-mini-3b-tamil-lora \
  --checkpoints_dir runs/voxtral_lora_full/checkpoints \
  --allow_hf_download
```

If you want artifact upload from a local adapter directory instead of HF:

```bash
python training/step_07_backfill_wandb.py \
  --wandb_project mistral-hackathon \
  --wandb_run_name voxtral-ta-retro-local \
  --local_adapter_dir runs/voxtral_lora_full \
  --checkpoints_dir runs/voxtral_lora_full/checkpoints
```

What step 07 logs:
- checkpoint timeline points (`step`, `epoch`, `micro`, `saved_at_unix`) if `trainer_state.pt` files exist
- model artifact to W&B (from local adapter dir or HF model repo)

## Notes on dedup/cleanup

- Removed duplicate trainer variants in this folder.
- Kept a single trainer for step 05: `step_05_train_voxtral_qlora.py`.
- Kept inference/eval check in training as step 06:
  - `step_06_inference_check_gpu.py`
