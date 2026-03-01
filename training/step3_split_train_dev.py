import json, random
from collections import defaultdict
from pathlib import Path

INP  = Path("data/all_flac.jsonl")
TRAIN = Path("data/train_flac.jsonl")
DEV   = Path("data/dev_flac.jsonl")

DEV_FRAC = 0.02          # 2% dev
SEED = 1337              # reproducible split

def main():
    assert INP.exists(), f"Missing {INP}. Run Step 2 first."

    rows = []
    with INP.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Group by speaker_id so dev speakers don't leak into train
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r.get("speaker_id", "unknown")].append(r)

    speakers = list(by_spk.keys())
    random.Random(SEED).shuffle(speakers)

    target_dev = int(len(rows) * DEV_FRAC)
    dev_speakers = set()
    count = 0
    for spk in speakers:
        if count >= target_dev:
            break
        dev_speakers.add(spk)
        count += len(by_spk[spk])

    train_rows, dev_rows = [], []
    for spk, items in by_spk.items():
        (dev_rows if spk in dev_speakers else train_rows).extend(items)

    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with DEV.open("w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✅ Split complete")
    print("total:", len(rows))
    print("train:", len(train_rows))
    print("dev  :", len(dev_rows))
    print("speakers_total:", len(speakers))
    print("speakers_dev  :", len(dev_speakers))
    print("wrote:", str(TRAIN), "and", str(DEV))

if __name__ == "__main__":
    main()
