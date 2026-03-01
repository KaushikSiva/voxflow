#!/usr/bin/env python3
"""Step 03: Split exported JSONL into train/dev by speaker (no speaker leakage)."""

import json
import random
from collections import defaultdict
from pathlib import Path


INPUT_JSONL = Path("data/all_flac.jsonl")
TRAIN_JSONL = Path("data/train_flac.jsonl")
DEV_JSONL = Path("data/dev_flac.jsonl")

DEV_FRACTION = 0.02
SEED = 1337


def main() -> None:
    assert INPUT_JSONL.exists(), f"Missing {INPUT_JSONL}. Run Step 02 first."

    rows = [json.loads(line) for line in INPUT_JSONL.open("r", encoding="utf-8")]

    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row.get("speaker_id", "unknown")].append(row)

    speakers = list(by_speaker.keys())
    random.Random(SEED).shuffle(speakers)

    target_dev = int(len(rows) * DEV_FRACTION)
    dev_speakers = set()
    dev_count = 0
    for spk in speakers:
        if dev_count >= target_dev:
          break
        dev_speakers.add(spk)
        dev_count += len(by_speaker[spk])

    train_rows = []
    dev_rows = []
    for spk, items in by_speaker.items():
        (dev_rows if spk in dev_speakers else train_rows).extend(items)

    TRAIN_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_JSONL.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with DEV_JSONL.open("w", encoding="utf-8") as f:
        for row in dev_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Split complete")
    print("total:", len(rows))
    print("train:", len(train_rows))
    print("dev:", len(dev_rows))
    print("speakers_total:", len(speakers))
    print("speakers_dev:", len(dev_speakers))
    print("wrote:", TRAIN_JSONL, "and", DEV_JSONL)


if __name__ == "__main__":
    main()
