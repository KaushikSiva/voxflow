#!/usr/bin/env python3
"""
6d.py (Fix A + Option1 + path fix + prompt masking fix)

QLoRA fine-tuning script for:
  mistralai/Voxtral-Mini-3B-2507

Includes:
✅ Fix A: bypasses processor.apply_chat_template() to avoid KeyError 'mm_load_kwargs'
✅ Uses tokenizer.apply_chat_template() + feature_extractor manually
✅ Handles mistral-common validator (assistant-final) via continue_final_message
✅ Option 1 dataset normalization: audio_path + text -> audio + answer
✅ Robust audio path resolution (avoids /data/data/ double prefix)
✅ Prompt-loss masking computed via tokenizer-only prompt lengths (stable)
✅ Debug print of label_frac for first few steps
✅ Gradient checkpoint warning fixed with use_reentrant=False (when supported)

Dataset JSONL supported (your schema):
{"id":"...","audio_path":"data/flac/...flac","text":"<transcript>","speaker_id":"..."}

Install:
  pip install -U transformers peft bitsandbytes accelerate soundfile
  pip install -U "mistral-common[audio]"
Ubuntu deps:
  sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    VoxtralForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------
# Fix A: Safe chat-template application (no processor.apply_chat_template)
# -----------------------------
def _pad_2d_int(seqs: List[List[int]], pad_id: int) -> (torch.Tensor, torch.Tensor):
    """Pad list of int sequences to [B, T], return (input_ids, attention_mask)."""
    max_len = max(len(s) for s in seqs)
    bsz = len(seqs)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s)
        input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
        attn[i, :n] = 1
    return input_ids, attn


def _last_role_is_assistant(conversations: List[Any]) -> bool:
    """
    conversations is a batch list; each item is a list of messages.
    We treat assistant-final if any example ends with assistant (should be consistent within a batch).
    """
    try:
        if not conversations:
            return False
        last_conv = conversations[-1]
        if not last_conv:
            return False
        last_msg = last_conv[-1]
        return isinstance(last_msg, dict) and last_msg.get("role") == "assistant"
    except Exception:
        return False


def safe_voxtral_apply_chat_template(
    processor: Any,
    conversations: List[Any],
    *,
    add_generation_prompt: bool,
    max_length: int,
    padding: bool = True,
    truncation: bool = True,
    # audio defaults
    sampling_rate: int = 16000,
    pad_to_multiple_of: int = 480000,
    max_source_positions: int = 3000,
) -> Dict[str, torch.Tensor]:
    """
    Replacement for processor.apply_chat_template that avoids internal kwargs dispatch
    that can crash with KeyError: 'mm_load_kwargs'.

    Uses:
      - processor.tokenizer.apply_chat_template(...) to get token ids (+ sometimes loaded audio arrays)
      - processor.feature_extractor to get mel features
      - chunking into [Nchunks, feature_size, max_source_positions]
    """

    # If the last message is assistant and we are NOT adding a generation prompt,
    # mistral-common serving validator requires continue_final_message=True.
    continue_final_message = (not add_generation_prompt) and _last_role_is_assistant(conversations)

    tok_out = processor.tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=True,
        return_dict=True,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=None,  # return python lists for stability
    )

    audio = None
    if isinstance(tok_out, dict) and "audio" in tok_out:
        audio = tok_out.pop("audio")

    # Convert ids/masks to tensors; pad if ragged
    input_ids_list = tok_out["input_ids"]
    attn_list = tok_out.get("attention_mask")

    if not input_ids_list:
        raise RuntimeError("Tokenizer returned empty input_ids.")

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    is_ragged = any(len(x) != len(input_ids_list[0]) for x in input_ids_list)
    if is_ragged or attn_list is None:
        input_ids, attention_mask = _pad_2d_int(input_ids_list, pad_id)
    else:
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attn_list, dtype=torch.long)

    out: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Audio -> features
    if audio is not None:
        feats_list: List[torch.Tensor] = []
        for audio_array in audio:
            audio_inputs = processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                padding=True,
                truncation=False,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            )
            # [1, feature_size, T] -> [feature_size, T]
            input_features = audio_inputs["input_features"][0]
            feature_size = int(input_features.shape[0])
            T = int(input_features.shape[1])

            rem = T % max_source_positions
            if rem != 0:
                pad = max_source_positions - rem
                input_features = F.pad(input_features, (0, pad))
                T = int(input_features.shape[1])

            n_chunks = T // max_source_positions
            chunked = input_features.reshape(feature_size, n_chunks, max_source_positions).transpose(0, 1)
            feats_list.append(chunked)

        out["input_features"] = torch.cat(feats_list, dim=0)

    return out


# -----------------------------
# Dataset (Option 1 normalization + path fix)
# -----------------------------
class JsonlAudioChatDataset(Dataset):
    """
    Normalizes each JSONL row to:
      ex["audio"]  : str path/URL
      ex["answer"] : str transcript/target
      ex["instruction"] : str

    Supports your schema:
      {"audio_path": "...", "text": "..."}
    """

    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        self.jsonl_path = os.path.abspath(path)
        self.jsonl_dir = os.path.dirname(self.jsonl_path)
        # repo_root = parent of jsonl_dir (works for .../tamil-str/data/*.jsonl)
        self.repo_root = os.path.abspath(os.path.join(self.jsonl_dir, ".."))

        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {ln} of {path}: {e}") from e

                # audio: accept "audio", {"audio":{"path":...}}, or "audio_path" + aliases
                audio = ex.get("audio")
                if isinstance(audio, dict):
                    audio = audio.get("path") or audio.get("file") or audio.get("filepath")

                audio = (
                    audio
                    or ex.get("audio_path")
                    or ex.get("path")
                    or ex.get("audio_file")
                    or ex.get("wav")
                    or ex.get("file")
                )
                if not audio:
                    raise ValueError(
                        f"Missing audio path on line {ln}. Keys={sorted(ex.keys())}. "
                        "Expected one of: audio, audio_path, path, audio_file, wav, file or {'audio':{'path':...}}."
                    )

                # Robust path resolution (avoid /data/data/ duplication):
                # - URLs unchanged
                # - absolute unchanged
                # - if starts with "data/", resolve from repo root
                # - else resolve from JSONL directory
                if isinstance(audio, str) and not (audio.startswith("http://") or audio.startswith("https://")):
                    if os.path.isabs(audio):
                        resolved = audio
                    else:
                        if audio.startswith("data/") or audio.startswith("data\\"):
                            resolved = os.path.abspath(os.path.join(self.repo_root, audio))
                        else:
                            resolved = os.path.abspath(os.path.join(self.jsonl_dir, audio))
                    ex["audio"] = resolved
                else:
                    ex["audio"] = audio

                # answer: prefer answer, else text
                answer = ex.get("answer") or ex.get("text")
                if not answer:
                    raise ValueError(
                        f"Missing transcript/answer on line {ln}. Keys={sorted(ex.keys())}. Expected 'answer' or 'text'."
                    )
                ex["answer"] = answer

                # instruction default for Tamil ASR
                if not ex.get("instruction"):
                    ex["instruction"] = "Transcribe the audio in Tamil. Output only the transcript."

                self.items.append(ex)

        if not self.items:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -----------------------------
# Collator (Fix A helper + tokenizer-only prompt masking)
# -----------------------------
@dataclass
class VoxtralChatCollator:
    processor: Any
    max_length: int = 8192
    mask_prompt_loss: bool = True

    # audio feature params (match safe_voxtral_apply_chat_template defaults)
    sampling_rate: int = 16000
    pad_to_multiple_of: int = 480000
    max_source_positions: int = 3000

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conv_full = []
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
            conv_full.append([user_msg, {"role": "assistant", "content": answer}])

        # Full inputs (user + assistant text)
        full_inputs = safe_voxtral_apply_chat_template(
            self.processor,
            conv_full,
            add_generation_prompt=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            sampling_rate=self.sampling_rate,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_source_positions=self.max_source_positions,
        )

        if "input_ids" not in full_inputs or "attention_mask" not in full_inputs:
            raise RuntimeError("safe_voxtral_apply_chat_template did not produce input_ids/attention_mask.")

        labels = full_inputs["input_ids"].clone()

        if self.mask_prompt_loss:
            # Compute prompt token lengths purely from tokenizer template (stable; no audio feature involvement)
            prompt_tok = self.processor.tokenizer.apply_chat_template(
                conv_prompt,
                add_generation_prompt=True,
                continue_final_message=False,
                tokenize=True,
                return_dict=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.processor.tokenizer.eos_token_id
            prompt_lengths = [sum(1 for t in row if t != pad_id) for row in prompt_tok["input_ids"]]

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

    # Gradient checkpointing (silence torch warning by setting use_reentrant explicitly when supported)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
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
            # Decoder linears
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
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (full bf16/fp16)")
    ap.add_argument("--fp16", action="store_true", default=False)

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Prompt masking
    ap.add_argument("--no_mask_prompt_loss", action="store_true")

    args = ap.parse_args()

    load_in_4bit = not args.no_4bit
    bf16 = not args.fp16
    mask_prompt_loss = not args.no_mask_prompt_loss

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    print("Base model:", args.base)
    print("Train JSONL:", args.train_jsonl)
    print("Out dir:", args.out_dir)
    print(f"Precision: {'bf16' if bf16 else 'fp16'}")
    print("4-bit:", load_in_4bit)
    print("mask_prompt_loss:", mask_prompt_loss)

    model, processor = build_model_and_processor(
        base_model=args.base,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
    )

    model = attach_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

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

    steps_per_epoch = math.ceil(len(dl) / max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

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

            # Move tensors to device; keep ids/labels as long
            moved: Dict[str, Any] = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    if k in ("input_ids", "labels", "attention_mask"):
                        moved[k] = v.to(main_device)
                    else:
                        moved[k] = v.to(main_device, dtype=input_dtype)
                else:
                    moved[k] = v
            batch = moved

            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]

            (loss / args.grad_accum).backward()
            running_loss += float(loss.item())

            if micro % args.grad_accum == 0:
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                # DEBUG: verify label masking works
                if step <= 3:
                    try:
                        frac = (batch["labels"] != -100).float().mean().item()
                        print(f"[debug] label_frac={frac:.3f}")
                    except Exception:
                        pass

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


if __name__ == "__main__":
    main()
