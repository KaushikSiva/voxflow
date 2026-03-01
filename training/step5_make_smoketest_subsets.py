import json, random
from pathlib import Path

TRAIN_IN  = Path("data/train_flac.jsonl")
DEV_IN    = Path("data/dev_flac.jsonl")

TRAIN_OUT = Path("data/train_flac_2k.jsonl")
DEV_OUT   = Path("data/dev_flac_200.jsonl")

N_TRAIN = 2000
N_DEV   = 200

SEED_TRAIN = 1
SEED_DEV   = 2

def sample(inp: Path, out: Path, n: int, seed: int):
    assert inp.exists(), f"Missing {inp}"
    rows = [json.loads(l) for l in inp.open("r", encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows[:n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ wrote {out} -> {n} rows (from {len(rows)} total)")

def main():
    sample(TRAIN_IN, TRAIN_OUT, N_TRAIN, SEED_TRAIN)
    sample(DEV_IN, DEV_OUT, N_DEV, SEED_DEV)

if __name__ == "__main__":
    main()
