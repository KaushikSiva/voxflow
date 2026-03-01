#!/usr/bin/env python3
"""
step6b_train_voxtral_qlora.py

QLoRA fine-tuning script for:
  mistralai/Voxtral-Mini-3B-2507

✅ Uses VoxtralForConditionalGeneration + AutoProcessor
✅ Uses processor.apply_chat_template(...) for audio+text conversations
✅ Targets Voxtral multimodal projector layers for LoRA
✅ Correct grad-accum logging (averages unscaled micro losses)
✅ Saves LoRA adapter + processor to OUT_DIR

Dataset format (JSONL recommended):
Each line is a JSON object with:
  - "audio": path to audio file (wav/mp3/...) OR URL
  - "instruction": (optional) instruction text (e.g., "Transcribe the audio.")
  - "answer": (required) expected assistant output (e.g., transcript)

Example JSONL:
{"audio":"./data/a.wav","instruction":"Transcribe in Tamil.","answer":"..."}
{"audio":"./data/b.mp3","instruction":"Summarize what you hear.","answer":"..."}

Install (typical):
  pip install -U "transformers>=4.56.0" peft bitsandbytes accelerate datasets soundfile
  pip install -U "mistral-common[audio]"

Refs:
- Voxtral docs (Transformers): https://huggingface.co/docs/transformers/en/model_doc/voxtral
- Voxtral Mini model page:      https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- HF discussion (projector LoRA targets): https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/discussions/1
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    Very simple JSONL dataset reader.

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
                    ex["instruction"] = "Please transcribe or answer based on the audio."

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
    # If True, tries to compute labels that only learn on assistant tokens.
    # If HF/processor doesn't return an assistant_tokens_mask, falls back to labels=input_ids (learns prompt too).
    mask_prompt_loss: bool = True

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Builds a batch using:
          conversation = [
            {
              "role": "user",
              "content": [
                {"type":"audio","path": <audio>},
                {"type":"text","text": <instruction>},
              ]
            },
            {
              "role": "assistant",
              "content": <answer>
            }
          ]

        Then uses processor.apply_chat_template(conversation, ...) to produce:
          input_ids, attention_mask, input_features, feature_attention_mask (names may vary by version)

        We also create labels.
        """

        conversations = []
        for ex in examples:
            audio_path = ex["audio"]
            instruction = ex.get("instruction", "") or "Please respond based on the audio."
            answer = ex["answer"]

            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": audio_path},
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]
            conversations.append(conv)

        # Try to request assistant token mask if supported by the tokenizer/backend.
        # Some versions expose: return_assistant_tokens_mask=True
        # For multimodal processors, this may or may not work depending on installed transformers version.
        kwargs = dict(
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_dict=True,
        )

        assistant_mask = None
        inputs = None

        if self.mask_prompt_loss:
            try:
                inputs = self.processor.apply_chat_template(
                    conversations,
                    return_assistant_tokens_mask=True,
                    **kwargs,
                )
                # Some tokenizers place it here:
                # - "assistant_tokens_mask" or similar
                if isinstance(inputs, dict):
                    assistant_mask = inputs.get("assistant_tokens_mask", None)
            except TypeError:
                # processor/tokenizer does not support return_assistant_tokens_mask
                inputs = self.processor.apply_chat_template(conversations, **kwargs)
            except Exception:
                # Fall back safely
                inputs = self.processor.apply_chat_template(conversations, **kwargs)
        else:
            inputs = self.processor.apply_chat_template(conversations, **kwargs)

        # Ensure dict
        if not isinstance(inputs, dict):
            # Some backends might return a BatchEncoding-like object; it is dict-like
            inputs = dict(inputs)

        if "input_ids" not in inputs:
            raise RuntimeError(
                "processor.apply_chat_template did not return input_ids. "
                "Check your transformers/mistral-common versions."
            )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Mask out prompt tokens if we got a usable assistant mask
        if self.mask_prompt_loss and assistant_mask is not None:
            # assistant_mask shape: [B, T] with 1 for assistant tokens
            if isinstance(assistant_mask, torch.Tensor) and assistant_mask.shape == labels.shape:
                labels[assistant_mask == 0] = -100
            else:
                # If mask is unexpected, fall back to training on all tokens
                pass

        inputs["labels"] = labels
        return inputs


# -----------------------------
# Helpers
# -----------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_and_processor(
    base_model: str,
    load_in_4bit: bool,
    bf16: bool,
) -> (Any, Any):
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

    # Training-friendly toggles
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, processor


def attach_lora(
    model: Any,
    r: int,
    alpha: int,
    dropout: float,
) -> Any:
    # Required for QLoRA / k-bit training stability
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # Standard Mistral-like decoder linears
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # Voxtral-specific: multimodal projector layers
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

    ap.add_argument("--load_in_4bit", action="store_true", default=True)
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (full/bf16)")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp16", action="store_true", default=False)

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--mask_prompt_loss", action="store_true", default=True)
    ap.add_argument("--no_mask_prompt_loss", action="store_true")

    args = ap.parse_args()

    if args.no_4bit:
        args.load_in_4bit = False

    if args.fp16:
        args.bf16 = False

    if args.no_mask_prompt_loss:
        args.mask_prompt_loss = False

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    print("Base model:", args.base)
    print("Train JSONL:", args.train_jsonl)
    print("Out dir:", args.out_dir)
    print(f"Precision: {'bf16' if args.bf16 else 'fp16'}")
    print("4-bit:", args.load_in_4bit)
    print("mask_prompt_loss:", args.mask_prompt_loss)

    # Load
    model, processor = build_model_and_processor(
        base_model=args.base,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
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
        mask_prompt_loss=args.mask_prompt_loss,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
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

    # Choose dtype for inputs (audio features)
    input_dtype = torch.bfloat16 if args.bf16 else torch.float16

    for epoch in range(args.epochs):
        for batch in dl:
            micro += 1

            # Move tensors to correct device/dtype.
            # With device_map="auto", the model is sharded; but inputs can still be moved to the first parameter device.
            # For HF multimodal inputs, processor returns tensors on CPU.
            # We'll move all tensor values to the "main" device inferred from model.
            try:
                main_device = next(model.parameters()).device
            except StopIteration:
                main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    # input_ids should remain long
                    if k in ("input_ids", "labels", "attention_mask"):
                        batch[k] = v.to(main_device)
                    else:
                        batch[k] = v.to(main_device, dtype=input_dtype)

            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]

            # backprop scaled
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
