#!/usr/bin/env python3
import os, json, math, time, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SR = 16000

def load_audio_16k_mono(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torch.nn.functional as F
        x = torch.tensor(audio).unsqueeze(0).unsqueeze(0)  # [1,1,T]
        new_len = int(x.shape[-1] * (SR / sr))
        x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
        audio = x.squeeze().cpu().numpy()
    return audio

class JsonlAudioText(Dataset):
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        if not self.items:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]
        audio_path = ex.get("audio_path") or ex.get("path") or ex.get("audio") or ex.get("audio_filepath")
        if isinstance(audio_path, dict) and "path" in audio_path:
            audio_path = audio_path["path"]
        text = ex.get("text") or ex.get("normalized") or ex.get("verbatim")
        if audio_path is None or text is None:
            raise KeyError(f"Bad example keys={list(ex.keys())}")

        # normalize relative paths like "data/flac/xxx.flac"
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(os.getcwd(), audio_path)

        return {"id": ex.get("id", str(idx)), "audio_path": audio_path, "text": str(text).strip()}

@dataclass
class Collator:
    processor: Any
    model_id: str
    max_length: int
    device: torch.device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios = [load_audio_16k_mono(x["audio_path"]) for x in batch]
        targets = [x["text"] for x in batch]

        # Conversation format: user provides audio + instruction, assistant returns transcript.
        # Use apply_chat_template(tokenize=True, return_dict=True) so processor handles audio correctly.
        conv_full = []
        conv_prompt = []
        for t in targets:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "audio", "array": None},  # placeholder
                    {"type": "text", "text": "Transcribe:"},
                ],
            }
            # prompt-only: user message only
            conv_prompt.append([user_msg])
            # full: user + assistant answer
            conv_full.append([user_msg, {"role": "assistant", "content": t}])

        # Build prompt-only encodings first to get prompt lengths for masking.
        # We inject audio arrays after creating the conversation list.
        # VoxtralProcessor reads audio from the conversation when tokenize=True, return_dict=True.
        # So we must pass actual audio via the conversation content.
        for i in range(len(conv_full)):
            conv_prompt[i][0]["content"][0] = {"type": "audio", "array": audios[i], "sampling_rate": SR}
            conv_full[i][0]["content"][0]   = {"type": "audio", "array": audios[i], "sampling_rate": SR}

        enc_prompt = self.processor.apply_chat_template(
            conv_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        enc_full = self.processor.apply_chat_template(
            conv_full,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Truncate (simple) if too long
        input_ids = enc_full["input_ids"][:, : self.max_length]
        attention_mask = enc_full["attention_mask"][:, : self.max_length]
        input_features = enc_full["input_features"]  # keep as-is (already chunked/padded internally)

        # Labels = input_ids with prompt part masked
        labels = input_ids.clone()
        prompt_lens = enc_prompt["input_ids"].shape[1]
        prompt_lens = min(prompt_lens, labels.shape[1])
        labels[:, :prompt_lens] = -100

        # Mask padding tokens too
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch_out = {
            "input_ids": input_ids.to(self.device, dtype=torch.long),
            "attention_mask": attention_mask.to(self.device, dtype=torch.long),
            "input_features": input_features.to(self.device),  # float tensor
            "labels": labels.to(self.device, dtype=torch.long),
        }
        return batch_out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=0, help="save adapter every N optimizer steps (0=off)")
    ap.add_argument("--resume", default="", help="path to adapter dir to resume from (optional)")
    return ap.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Need CUDA GPU"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.base)

    print("Loading model (QLoRA)...")
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = AutoModel.from_pretrained(
        args.base,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=(torch.float16 if args.fp16 else torch.bfloat16),
    )

    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    base = prepare_model_for_kbit_training(base)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(base, lora)

    if args.resume:
        print("Resuming adapter from:", args.resume)
        model.load_adapter(args.resume, adapter_name="default")

    model.print_trainable_parameters()

    train_ds = JsonlAudioText(args.train_jsonl)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,  # IMPORTANT: we move to GPU in collator
        collate_fn=Collator(
            processor=processor,
            model_id=args.base,
            max_length=args.max_length,
            device=torch.device("cuda"),
        ),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"Train examples={len(train_ds)} microbatches/epoch={len(train_loader)} steps={total_steps} warmup={warmup_steps}")
    model.train()

    step = 0
    micro = 0
    t0 = time.time()
    optim.zero_grad(set_to_none=True)

    for ep in range(args.epochs):
        for batch in train_loader:
            micro += 1
            with torch.amp.autocast("cuda", dtype=(torch.float16 if args.fp16 else torch.bfloat16)):
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0]

            (loss / args.grad_accum).backward()

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == 1:
                    dt = time.time() - t0
                    rate = step / max(dt, 1e-9)
                    print(f"ep={ep+1} step={step:5d}/{total_steps} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} steps/s={rate:.3f}")

                if args.save_every and (step % args.save_every == 0):
                    ckpt_dir = os.path.join(args.out_dir, f"checkpoint-step{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    try:
                        processor.save_pretrained(ckpt_dir)
                    except Exception:
                        pass
                    print("Saved checkpoint:", ckpt_dir)

    print("Saving LoRA adapter to:", args.out_dir)
    model.save_pretrained(args.out_dir)
    try:
        processor.save_pretrained(args.out_dir)
    except Exception:
        pass
    print("✅ Done.")

if __name__ == "__main__":
    main()
