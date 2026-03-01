# Training Pipeline

This folder contains one script per training step.

## Step files

1. `step_01_download_parquets.py`
2. `step_02_export_flac_all.py`
3. `step_03_split_train_dev.py`
4. `step_04_make_smoketest_subsets.py`
5. `step_05_train_voxtral_qlora.py`

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

## Notes on dedup/cleanup

- Removed duplicate trainer variants in this folder.
- Kept a single trainer for step 05: `step_05_train_voxtral_qlora.py`.
- Moved inference/eval script out of training:
  - `inference/step_02_eval_lora_transcribe_jsonl.py`
