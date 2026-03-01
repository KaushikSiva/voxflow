---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-3B-2507
tags:
- voxtral
- speech
- asr
- tamil
- lora
---

# Voxtral Mini 3B – Tamil LoRA Adapter

This repo contains a **LoRA adapter** fine-tuned for **Tamil ASR** on top of `mistralai/Voxtral-Mini-3B-2507`.

## What’s in this repo?

- `adapter_model.safetensors` – LoRA weights  
- `adapter_config.json` – LoRA config  
- `processor_config.json` / tokenizer files (if present)

You load the **base model** + this **adapter** at runtime.

## Install

```bash
pip install -U transformers peft accelerate bitsandbytes mistral-common[audio] soundfile librosa
```

## Quick Usage

```
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from peft import PeftModel

BASE = "mistralai/Voxtral-Mini-3B-2507"
ADAPTER = "kaushiksiva/voxtral-mini-3b-tamil-lora"

processor = AutoProcessor.from_pretrained(BASE)

# Load base model
model = VoxtralForConditionalGeneration.from_pretrained(
    BASE,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

# Load LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER).eval()

# Voxtral transcription-mode request (recommended)
inputs = processor.apply_transcription_request(
    language="ta",
    audio="path/to/audio.flac",
    model_id=BASE,
)

# Move inputs to GPU
inputs = inputs.to("cuda", dtype=torch.bfloat16)

# Generate transcription
out_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=processor.tokenizer.eos_token_id,
)

# Decode only the generated continuation
prompt_len = inputs.input_ids.shape[1]
text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0]
print(text.strip())
```
