#!/usr/bin/env python3
"""Step 02: Export FLAC files from Tamil parquet rows and build a JSONL manifest."""

import glob
import hashlib
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq


PARQUET_GLOB = os.path.expanduser("~/datasets/indicvoices_tamil/tamil/train-*-of-00078.parquet")
OUT_DIR = Path("data/flac")
OUT_JSONL = Path("data/all_flac.jsonl")
PRINT_EVERY_SEC = 30


def stable_id(*parts: str) -> str:
    joined = "|".join(p or "" for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parquet_files = sorted(glob.glob(PARQUET_GLOB))
    assert parquet_files, f"No parquet files matched: {PARQUET_GLOB}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen = 0
    wrote = 0
    bad = 0
    start = time.time()
    last_print = start

    with OUT_JSONL.open("w", encoding="utf-8") as out_f:
        for i, parquet_path in enumerate(parquet_files, start=1):
            pf = pq.ParquetFile(parquet_path)
            print(f"[{i}/{len(parquet_files)}] {os.path.basename(parquet_path)} rowgroups={pf.num_row_groups}")

            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                for batch in table.to_batches(max_chunksize=512):
                    for row in batch.to_pylist():
                        seen += 1
                        try:
                            text = row.get("text")
                            if not isinstance(text, str) or not text.strip():
                                continue

                            speaker_id = str(row.get("speaker_id", "unknown"))
                            audio_file = row.get("audio_filepath")
                            if not isinstance(audio_file, dict) or not audio_file.get("bytes"):
                                continue

                            uid = stable_id(speaker_id, text.strip())
                            flac_path = OUT_DIR / f"{uid}.flac"
                            if not flac_path.exists():
                                flac_path.write_bytes(audio_file["bytes"])

                            out_f.write(
                                json.dumps(
                                    {
                                        "id": uid,
                                        "audio_path": str(flac_path),
                                        "text": text.strip(),
                                        "speaker_id": speaker_id,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            wrote += 1
                        except Exception:
                            bad += 1

                        now = time.time()
                        if now - last_print >= PRINT_EVERY_SEC:
                            elapsed = now - start
                            rate = wrote / elapsed if elapsed > 0 else 0.0
                            print(
                                f"  progress: seen={seen:,} wrote={wrote:,} bad={bad:,} "
                                f"wrote_rate={rate:,.1f}/s elapsed={elapsed/60:,.1f}m"
                            )
                            last_print = now

    elapsed = time.time() - start
    print(
        "DONE",
        f"seen={seen:,}",
        f"wrote={wrote:,}",
        f"bad={bad:,}",
        f"elapsed_min={elapsed/60:,.1f}",
        f"-> {OUT_JSONL}",
    )


if __name__ == "__main__":
    main()
