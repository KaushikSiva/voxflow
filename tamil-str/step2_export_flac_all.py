import os, json, glob, hashlib, time
import pyarrow.parquet as pq

PARQS = sorted(glob.glob(os.path.expanduser("~/datasets/indicvoices_tamil/tamil/train-*-of-00078.parquet")))
assert PARQS, "No parquets found"
print("parquets:", len(PARQS), "sample:", PARQS[:2])

OUT_DIR = "data/flac"
OUT = "data/all_flac.jsonl"
os.makedirs(OUT_DIR, exist_ok=True)

def sid(*parts):
    return hashlib.sha1(("|".join([p or "" for p in parts])).encode("utf-8")).hexdigest()[:16]

seen=0; wrote=0; bad=0
t0=time.time(); last=t0
PRINT_EVERY_SEC=30

with open(OUT, "w", encoding="utf-8") as f:
    for i, fp in enumerate(PARQS, start=1):
        pf = pq.ParquetFile(fp)
        print(f"[{i}/{len(PARQS)}] {os.path.basename(fp)} rowgroups={pf.num_row_groups}")

        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            for batch in tbl.to_batches(max_chunksize=512):
                for r in batch.to_pylist():
                    seen += 1
                    try:
                        text = r.get("text")
                        if not isinstance(text, str) or not text.strip():
                            continue
                        spk = str(r.get("speaker_id","unknown"))
                        af = r.get("audio_filepath")
                        if not isinstance(af, dict) or not af.get("bytes"):
                            continue

                        uid = sid(spk, text.strip())
                        flac_path = os.path.join(OUT_DIR, f"{uid}.flac")
                        if not os.path.exists(flac_path):
                            with open(flac_path, "wb") as wf:
                                wf.write(af["bytes"])

                        f.write(json.dumps({
                            "id": uid,
                            "audio_path": flac_path,
                            "text": text.strip(),
                            "speaker_id": spk
                        }, ensure_ascii=False) + "\n")
                        wrote += 1

                    except Exception:
                        bad += 1

                    now=time.time()
                    if now-last >= PRINT_EVERY_SEC:
                        dt = now-t0
                        rate = wrote/dt if dt>0 else 0.0
                        print(f"  progress: seen={seen:,} wrote={wrote:,} bad={bad:,} "
                              f"wrote_rate={rate:,.1f}/s elapsed={dt/60:,.1f}m")
                        last=now

dt=time.time()-t0
print("DONE seen", f"{seen:,}", "wrote", f"{wrote:,}", "bad", f"{bad:,}", "elapsed_min", f"{dt/60:,.1f}", "->", OUT)
