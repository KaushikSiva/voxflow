#!/usr/bin/env python3
"""Step 04: Build small train/dev subsets for smoke tests."""

import json
import random
from pathlib import Path


TRAIN_INPUT = Path("data/train_flac.jsonl")
DEV_INPUT = Path("data/dev_flac.jsonl")
TRAIN_OUTPUT = Path("data/train_flac_2k.jsonl")
DEV_OUTPUT = Path("data/dev_flac_200.jsonl")

TRAIN_ROWS = 2000
DEV_ROWS = 200
SEED_TRAIN = 1
SEED_DEV = 2


def sample_jsonl(inp: Path, out: Path, n: int, seed: int) -> None:
    assert inp.exists(), f"Missing {inp}"
    rows = [json.loads(line) for line in inp.open("r", encoding="utf-8")]
    random.Random(seed).shuffle(rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows[:n]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out} -> {n} rows (from {len(rows)} total)")


def main() -> None:
    sample_jsonl(TRAIN_INPUT, TRAIN_OUTPUT, TRAIN_ROWS, SEED_TRAIN)
    sample_jsonl(DEV_INPUT, DEV_OUTPUT, DEV_ROWS, SEED_DEV)


if __name__ == "__main__":
    main()
