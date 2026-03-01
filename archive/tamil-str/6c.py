#!/usr/bin/env python3
"""
step6b_train_voxtral_qlora.py

QLoRA fine-tuning script for:
  mistralai/Voxtral-Mini-3B-2507

✅ Uses VoxtralForConditionalGeneration + AutoProcessor
✅ Uses processor.apply_chat_template(...) with Voxtral’s audio "path" format
✅ Targets Voxtral multimodal projector layers for LoRA
✅ Correct grad-accum logging (averages unscaled micro losses)
✅ Masks prompt loss robustly (WITHOUT return_assistant_tokens_mask, which is often blocked for processors)
✅ Pads batches correctly (required for DataLoader stacking)
✅ Saves LoRA adapter + processor to OUT_DIR

Dataset format (JSONL):
Each line:
  - "audio": path or URL to audio
  - "instruction": optional
  - "answer": required

Example JSONL:
{"audio":"./data/a.wav","instruction":"Transcribe in Tamil.","answer":"..."}
{"audio":"https://.../b.mp3","instruction":"Summarize what you hear.","answer":"..."}

Install:
  pip install -U "transformers>=4.54.0" peft bitsandbytes accelerate datasets soundfile
  pip install -U "mistral-common[audio]"

Why prompt masking is implemented this way:
- ProcessorMixin.apply_chat_template(return_assistant_tokens_mask=True) may be blocked for processors in some versions.
  We instead compute prompt length by tokenizing:
    (user-only + add_generation_prompt=True) vs (user+assistant)
  and mask labels[:prompt_len] = -100 per sample.

Refs:
- Voxtral Transformers usage shows audio chunks with {"type":"audio","path":...}. :contentReference[oaicite:0]{index=0}
- Known apply_chat_template assistant-mask issue for processors. :contentReference[oaicite:1]{index=1}
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    VoxtralForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------
# Dataset
# -----------------------------
class JsonlAudioChatDataset(Dataset):
    """
    Each item:
      {
        "audio": str,
        "instruction": str (optional),
        "answer": str (required)
      }
    """

    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {ln} of {path}: {e}") from e

                if "audio" not in ex:
                    raise ValueError(f"Missing 'audio' on line {ln}")
                if "answer" not in ex:
                    raise ValueError(f"Missing 'answer' on line {ln}")

                if "instruction" not in ex or not ex["instruction"]:
                    ex["instruction"] = "Please respond based on the audio."

                self.items.append(ex)

        if not self.items:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -----------------------------
# Collator using apply_chat_template
# -----------------------------
@dataclass
class VoxtralChatCollator:
    processor: Any
    max_length: int = 8192
    mask_prompt_loss: bool = True

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Voxtral chat format (per model README):
          {
            "role": "user",
            "content": [
              {"type":"audio","path": "..."},
              {"type":"text","text": "..."},
            ]
          }
        For training, we include an assistant message with the target answer.

        Important:
        - We MUST pad to build a batch.
        - For prompt masking, we compute prompt token lengths by tokenizing
          user-only with add_generation_prompt=True.
        """

        # Full conversations (user + assistant)
        conv_full = []
        # Prompt-only conversations (user only) for prompt length calculation
        conv_prompt = []

        for ex in examples:
            audio_path = ex["audio"]
            instruction = ex.get("instruction", "") or "Please respond based on the audio."
            answer = ex["answer"]

            user_msg = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": instruction},
                ],
            }

            conv_prompt.append([user_msg])

            conv_full.append(
                [
                    user_msg,
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            )

        common_kwargs = dict(
            return_tensors="pt",
            padding=True,          # REQUIRED for batching
            truncation=True,
            max_length=self.max_length,
            return_dict=True,
        )

        # Tokenize prompt-only with generation prompt appended
        # This yields the length of tokens up to where assistant generation starts.
        prompt_inputs = self.processor.apply_chat_template(
            conv_prompt,
            add_generation_prompt=True,
            **common_kwargs,
        )

        # Tokenize full (includes assistant text in the input)
        full_inputs = self.processor.apply_chat_template(
            conv_full,
            add_generation_prompt=False,
            **common_kwargs,
        )

        if "input_ids" not in full_inputs or "attention_mask" not in full_inputs:
            raise RuntimeError(
                "processor.apply_chat_template did not return input_ids/attention_mask. "
                "Check your transformers/mistral-common versions."
            )

        labels = full_inputs["input_ids"].clone()

        if self.mask_prompt_loss:
            if "attention_mask" not in prompt_inputs:
                raise RuntimeError("prompt_inputs missing attention_mask; cannot compute prompt lengths.")

            # prompt_lengths: number of non-pad tokens per row in the prompt-only encoding
            prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()

            # Mask prompt tokens in labels
            for i, plen in enumerate(prompt_lengths):
                plen = int(plen)
                if plen > 0:
                    labels[i, :plen] = -100

        full_inputs["labels"] = labels
        return full_inputs


# -----------------------------
# Helpers
# -----------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_and_processor(base_model: str, load_in_4bit: bool, bf16: bool):
    processor = AutoProcessor.from_pretrained(base_model)

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    model = VoxtralForConditionalGeneration.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=quant_cfg,
    )

    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, processor


def attach_lora(model: Any, r: int, alpha: int, dropout: float):
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # Standard decoder linears
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # Voxtral multimodal projector
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
        ],
    )

    model = get_peft_model(model, lora_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def get_main_device(model: torch.nn.Module) -> torch.device:
    # For device_map="auto", inputs should land on the "first" device.
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Main train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)

    # Quant / dtype
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (full/bf16)")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp16", action="store_true", default=False)

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Prompt masking
    ap.add_argument("--no_mask_prompt_loss", action="store_true")

    args = ap.parse_args()

    load_in_4bit = not args.no_4bit
    bf16 = True
    if args.fp16:
        bf16 = False
    elif args.bf16:
        bf16 = True

    mask_prompt_loss = not args.no_mask_prompt_loss

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    print("Base model:", args.base)
    print("Train JSONL:", args.train_jsonl)
    print("Out dir:", args.out_dir)
    print(f"Precision: {'bf16' if bf16 else 'fp16'}")
    print("4-bit:", load_in_4bit)
    print("mask_prompt_loss:", mask_prompt_loss)

    # Load
    model, processor = build_model_and_processor(
        base_model=args.base,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
    )

    # LoRA
    model = attach_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Dataset + loader
    ds = JsonlAudioChatDataset(args.train_jsonl)

    collate = VoxtralChatCollator(
        processor=processor,
        max_length=args.max_length,
        mask_prompt_loss=mask_prompt_loss,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    # Steps
    steps_per_epoch = math.ceil(len(dl) / max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # Optimizer + scheduler
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Train
    model.train()
    step = 0
    micro = 0
    t0 = time.time()
    running_loss = 0.0

    print(f"Total examples: {len(ds)}")
    print(f"Total steps:    {total_steps} (epochs={args.epochs}, batch={args.batch_size}, accum={args.grad_accum})")
    print(f"Warmup steps:   {warmup_steps}")
    print("-" * 80)

    input_dtype = torch.bfloat16 if bf16 else torch.float16
    main_device = get_main_device(model)

    for epoch in range(args.epochs):
        for batch in dl:
            micro += 1

            # Move tensors to the model input device.
            # Keep input_ids/labels as long, attention_mask as long/bool, features as bf16/fp16.
            moved = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    if k in ("input_ids", "labels"):
                        moved[k] = v.to(main_device)
                    elif k == "attention_mask":
                        moved[k] = v.to(main_device)
                    else:
                        moved[k] = v.to(main_device, dtype=input_dtype)
                else:
                    moved[k] = v
            batch = moved

            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]

            (loss / args.grad_accum).backward()
            running_loss += loss.item()

            if micro % args.grad_accum == 0:
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == 1:
                    dt = time.time() - t0
                    rate = step / max(dt, 1e-6)
                    avg_loss = running_loss / args.grad_accum
                    running_loss = 0.0
                    print(
                        f"epoch={epoch+1}/{args.epochs} "
                        f"step={step:4d}/{total_steps} "
                        f"loss={avg_loss:.4f} "
                        f"lr={sched.get_last_lr()[0]:.2e} "
                        f"steps/s={rate:.3f}"
                    )

                if step >= total_steps:
                    break

        if step >= total_steps:
            break

    print("-" * 80)
    print("Saving LoRA adapter to:", args.out_dir)
    model.save_pretrained(args.out_dir, safe_serialization=True)

    try:
        processor.save_pretrained(args.out_dir)
    except Exception as e:
        print("processor.save_pretrained failed (ok):", e)

    print("✅ Done. Next: run an inference sanity-check on a few dev samples.")
    print("Tip: load base + adapter with PeftModel.from_pretrained(...) for evaluation.")


if __name__ == "__main__":
    main()
