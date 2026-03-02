#!/usr/bin/env python3
"""Step 07: Retrospectively backfill W&B run + model artifact for an existing adapter."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download

wandb = None


@dataclass
class CheckpointPoint:
    step: int
    epoch: int
    micro: int
    saved_at_unix: float
    path: str


def parse_step_from_dir(path: Path) -> int:
    m = re.search(r"step_(\d+)", str(path))
    if m:
        return int(m.group(1))
    return 0


def collect_checkpoint_points(checkpoints_dir: Path) -> List[CheckpointPoint]:
    points: List[CheckpointPoint] = []
    if not checkpoints_dir.exists():
        return points
    try:
        import torch  # local import: avoid hard failure when torch env is incompatible
    except Exception as exc:
        print(
            "[warn] torch import failed; skipping checkpoint timeline backfill.\n"
            f"       reason: {exc}"
        )
        return points

    for trainer_state in sorted(checkpoints_dir.rglob("trainer_state.pt")):
        try:
            state = torch.load(trainer_state, map_location="cpu")
        except Exception:
            continue

        step = int(state.get("step", parse_step_from_dir(trainer_state.parent)))
        epoch = int(state.get("epoch", 0))
        micro = int(state.get("micro", 0))
        saved_at = float(state.get("saved_at_unix", 0.0))
        points.append(
            CheckpointPoint(
                step=step,
                epoch=epoch,
                micro=micro,
                saved_at_unix=saved_at,
                path=str(trainer_state.parent),
            )
        )

    points.sort(key=lambda x: x.step)
    return points


def log_checkpoint_timeline(points: List[CheckpointPoint]) -> None:
    if not points:
        return

    table = wandb.Table(columns=["step", "epoch", "micro", "saved_at_unix", "checkpoint_dir"])
    for p in points:
        wandb.log(
            {
                "backfill/checkpoint_seen": 1,
                "backfill/epoch": p.epoch,
                "backfill/micro": p.micro,
                "backfill/saved_at_unix": p.saved_at_unix,
            },
            step=p.step,
        )
        table.add_data(p.step, p.epoch, p.micro, p.saved_at_unix, p.path)

    wandb.log({"backfill/checkpoint_table": table})


def add_model_artifact(
    artifact_name: str,
    model_repo: str,
    local_adapter_dir: Optional[str],
    allow_hf_download: bool,
) -> None:
    artifact = wandb.Artifact(artifact_name, type="model")

    if local_adapter_dir:
        p = Path(local_adapter_dir)
        if not p.exists():
            raise FileNotFoundError(f"Local adapter dir not found: {p}")
        artifact.add_dir(str(p))
        artifact.metadata = {"source": "local", "path": str(p)}
        wandb.log_artifact(artifact)
        return

    if not allow_hf_download:
        raise RuntimeError(
            "No --local_adapter_dir provided and --allow_hf_download is false. "
            "Provide one of them to log the model artifact."
        )

    with tempfile.TemporaryDirectory(prefix="hf_adapter_") as tmpdir:
        snapshot_path = snapshot_download(
            repo_id=model_repo,
            repo_type="model",
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
                "README.md",
                "*.json",
            ],
        )
        artifact.add_dir(snapshot_path)
        artifact.metadata = {"source": "huggingface", "repo_id": model_repo}
        wandb.log_artifact(artifact)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_project", type=str, default="mistral-hackathon")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="voxtral-ta-retro-backfill")

    ap.add_argument("--model_repo", type=str, default="kaushiksiva/voxtral-mini-3b-tamil-lora")
    ap.add_argument("--local_adapter_dir", type=str, default="")
    ap.add_argument("--artifact_name", type=str, default="mistral-lora-adapter")
    ap.add_argument("--allow_hf_download", action="store_true", default=True)

    ap.add_argument(
        "--checkpoints_dir",
        type=str,
        default="runs/voxtral_lora_full/checkpoints",
        help="Directory containing step_*/trainer_state.pt files.",
    )
    args = ap.parse_args()

    global wandb
    try:
        import wandb as _wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb is not installed in this Python environment. "
            "Install and retry:\n"
            "  pip install wandb\n"
            "If you are using a virtualenv, activate it first."
        ) from exc
    wandb = _wandb

    wandb_kwargs = {
        "project": args.wandb_project,
        "job_type": "backfill",
        "name": args.wandb_run_name,
        "config": vars(args),
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity

    wandb.init(**wandb_kwargs)

    points = collect_checkpoint_points(Path(args.checkpoints_dir))
    wandb.summary["backfill/checkpoint_count"] = len(points)
    if points:
        wandb.summary["backfill/first_step"] = points[0].step
        wandb.summary["backfill/last_step"] = points[-1].step

    log_checkpoint_timeline(points)
    add_model_artifact(
        artifact_name=args.artifact_name,
        model_repo=args.model_repo,
        local_adapter_dir=args.local_adapter_dir or None,
        allow_hf_download=args.allow_hf_download,
    )

    print("Backfill complete")
    print("checkpoints_logged:", len(points))
    print("artifact:", args.artifact_name)
    print("model_repo:", args.model_repo)

    wandb.finish()


if __name__ == "__main__":
    main()
