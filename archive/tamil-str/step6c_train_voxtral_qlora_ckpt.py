#!/usr/bin/env python3
import os, json, math, time, glob, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SR = 16000

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        if not self.items:
            raise ValueError(f"Empty dataset: {path}")

        # for resolving relative paths like "data/flac/xxx.flac"
        self.base_dir = os.path.dirname(os.path.abspath(path))

    def __len__(self):
        return len(self.items)

    def _resolve_audio_path(self, p: str) -> str:
        p = str(p)
        if os.path.isabs(p):
            return p
        # common case: "data/flac/.."
        cand = os.path.join(os.getcwd(), p)
        if os.path.exists(cand):
            return cand
        # try relative to jsonl dir
        cand = os.path.join(self.base_dir, p)
        if os.path.exists(cand):
            return cand
        # last resort: as-is
        return p

    def __getitem__(self, idx: int):
        ex = self.items[idx]
        audio_path = ex.get("audio_path") or ex.get("path") or ex.get("audio") or ex.get("audio_filepath")
        if isinstance(audio_path, dict) and "path" in audio_path:
            audio_path = audio_path["path"]
        text = ex.get("text") or ex.get("normalized") or ex.get("verbatim")
        if audio_path is None or text is None:
            raise KeyError(f"Bad example keys={list(ex.keys())}")

        return {
            "id": ex.get("id", str(idx)),
            "audio_path": self._resolve_audio_path(audio_path),
            "text": str(text).strip(),
        }

@dataclass
class VoxtralASRCollator:
    processor: Any
    model_id: str
    language: str
    max_text_tokens: int = 256
    audio_format: str = "FLAC"   # "WAV" is also ok; this is metadata for the request builder

    def __post_init__(self):
        self.tok = self.processor.tokenizer
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios = [load_audio_16k_mono(x["audio_path"]) for x in batch]
        texts  = [x["text"] for x in batch]

        # 1) Build PROMPT+audio features in the model-native way.
        # This is the key to avoid mm_load_kwargs/apply_chat_template issues.
        prompt = self.processor.apply_transcription_request(
            language=self.language,
            model_id=self.model_id,
            audio=audios,
            format=[self.audio_format] * len(audios),
            return_tensors="pt",
        )

        passthrough = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}
        prompt_ids  = prompt["input_ids"]        # [B, Lp]
        prompt_attn = prompt["attention_mask"]   # [B, Lp]

        # 2) Tokenize transcripts (NO return_dict/tokenize kwargs; mistral-common backend rejects them)
        text_tok = self.tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate prompt + transcript; labels learn only transcript tokens
        input_ids, attention_mask, labels = [], [], []
        B = prompt_ids.size(0)
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            ids  = p_ids + t_ids
            attn = p_att + [1] * len(t_ids)
            lab  = [-100] * len(p_ids) + t_ids

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill):
            return seq + [fill] * (max_len - len(seq))

        input_ids      = torch.tensor([pad_to(x, self.pad_id) for x in input_ids], dtype=torch.long)
        attention_mask = torch.tensor([pad_to(x, 0)          for x in attention_mask], dtype=torch.long)
        labels         = torch.tensor([pad_to(x, -100)       for x in labels], dtype=torch.long)

        # Move passthrough tensors later (in main) with a safe device-cast
        batch_out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        batch_out.update(passthrough)
        return batch_out

def to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
                out[k] = v.to(device)
            else:
                out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v
    return out

def latest_checkpoint(out_dir: str) -> Optional[str]:
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    return ckpts[-1] if ckpts else None

def save_checkpoint(out_dir: str, step: int, model, optim, sched, meta: Dict[str, Any]):
    ckpt_dir = os.path.join(out_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # LoRA adapter weights
    model.save_pretrained(ckpt_dir)

    # Optim/sched state + metadata
    torch.save(
        {
            "step": step,
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "meta": meta,
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    # update symlink-ish pointer
    with open(os.path.join(out_dir, "LATEST"), "w") as f:
        f.write(str(step))

def load_checkpoint(ckpt_dir: str, model, optim, sched):
    state_path = os.path.join(ckpt_dir, "trainer_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing {state_path}")
    st = torch.load(state_path, map_location="cpu")
    optim.load_state_dict(st["optimizer"])
    sched.load_state_dict(st["scheduler"])
    return int(st.get("step", 0)), st.get("meta", {})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--language", default=os.getenv("LANG", "ta"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_text_tokens", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)   # optimizer steps
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Need CUDA GPU"

    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.base)

    print("Loading model...")
    if args.no_4bit:
        model = AutoModel.from_pretrained(args.base, device_map="auto", torch_dtype=compute_dtype)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModel.from_pretrained(
            args.base,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)

    # --- dtype alignment for Voxtral masked_scatter (text embeds must match audio embeds) ---
    try:
        emb = model.get_input_embeddings()
        emb.to(dtype=compute_dtype)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.to(dtype=compute_dtype)
    except Exception as e:
        print("[warn] could not cast embeddings/lm_head dtype:", e)
    # -------------------------------------------------------------------------------

    model.print_trainable_parameters()

    train_ds = JsonlAudioText(args.train_jsonl)

    collator = VoxtralASRCollator(
        processor=processor,
        model_id=args.base,
        language=args.language,
        max_text_tokens=args.max_text_tokens,
        audio_format="FLAC",
    )

    # IMPORTANT: don't pin_memory when you later move tensors to GPU yourself
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collator,
        drop_last=False,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = int(math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    start_step = 0
    if args.resume:
        ckpt = latest_checkpoint(args.out_dir)
        if ckpt:
            print("Resuming from:", ckpt)
            # load LoRA weights
            model.load_adapter(ckpt, adapter_name="default")
            start_step, _ = load_checkpoint(ckpt, model, optim, sched)
            print("Resumed at optimizer step:", start_step)
        else:
            print("No checkpoint found; starting fresh.")

    print(f"Train examples={len(train_ds)} microbatches/epoch={len(train_loader)} "
          f"steps={total_steps} warmup={warmup_steps} (bs={args.batch_size} accum={args.grad_accum})")

    model.train()
    t0 = time.time()
    micro = 0
    step = start_step
    optim.zero_grad(set_to_none=True)

    # Skip already-seen steps when resuming (approx: skip step*grad_accum microbatches)
    skip_micro = step * args.grad_accum

    for _ in range(int(math.ceil(args.epochs))):
        for batch in train_loader:
            micro += 1
            if micro <= skip_micro:
                continue

            batch = to_device(batch, device=torch.device("cuda"), dtype=compute_dtype)

            with torch.amp.autocast("cuda", dtype=compute_dtype):
                emb_dtype = model.get_input_embeddings().weight.dtype
                if "input_features" in batch and torch.is_tensor(batch["input_features"]):
                    batch["input_features"] = batch["input_features"].to(
                        device=batch["input_features"].device, dtype=emb_dtype
                    )

                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else out[0]

            (loss / args.grad_accum).backward()

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                step += 1

                if step % args.log_every == 0 or step == 1:
                    dt = time.time() - t0
                    rate = step / max(dt, 1e-9)
                    print(f"step={step:6d}/{total_steps} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} steps/s={rate:.3f}")

                if args.save_every > 0 and (step % args.save_every == 0):
                    save_checkpoint(
                        args.out_dir, step, model, optim, sched,
                        meta={"base": args.base, "train_jsonl": args.train_jsonl, "language": args.language}
                    )
                    print("✅ saved checkpoint", step)

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    # final save
    print("Saving final LoRA adapter to:", args.out_dir)
    model.save_pretrained(args.out_dir)
    try:
        processor.save_pretrained(args.out_dir)
    except Exception as e:
        print("processor.save_pretrained failed (ok):", e)

    save_checkpoint(
        args.out_dir, step, model, optim, sched,
        meta={"base": args.base, "train_jsonl": args.train_jsonl, "language": args.language}
    )
    print("✅ Done.")

if __name__ == "__main__":
    main()
