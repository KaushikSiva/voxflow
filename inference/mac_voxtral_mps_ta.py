import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from peft import PeftModel

BASE = "mistralai/Voxtral-Mini-3B-2507"
ADAPTER = "kaushiksiva/voxtral-mini-3b-tamil-lora"
AUDIO = "./0b90a6693d8d4ff9.flac"  # <-- change

assert torch.backends.mps.is_available(), "MPS not available"
device = torch.device("mps")
dtype = torch.float16

print("torch:", torch.__version__, "device:", device, "dtype:", dtype)

processor = AutoProcessor.from_pretrained(BASE)

model = VoxtralForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=dtype,
    device_map=None,
).eval()
model = PeftModel.from_pretrained(model, ADAPTER).eval()
model.to(device)

# Inference speed
model.config.use_cache = True

inputs = processor.apply_transcription_request(
    language="ta",
    audio=AUDIO,
    model_id=BASE,
)

# Move tensors carefully:
for k, v in list(inputs.items()):
    if not torch.is_tensor(v):
        continue
    if k in ("input_ids",):
        inputs[k] = v.to(device)
    elif k in ("attention_mask",):
        inputs[k] = v.to(device)  # keep as int/bool
    else:
        inputs[k] = v.to(device, dtype=dtype)  # audio features -> fp16

with torch.inference_mode():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

prompt_len = inputs["input_ids"].shape[1]
text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
print(text)