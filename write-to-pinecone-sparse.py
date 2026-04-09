"""
Upload records from a directory of CSV files to a Pinecone sparse index.
Pinecone embeds the `text` column server-side using learned sparse retrieval
(DeepImpact architecture) — no local model needed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import backoff
import dotenv
import pandas as pd
from pinecone import Pinecone, PineconeException
from tqdm import tqdm

dotenv.load_dotenv()

# ── Configuration — adapt these to your project ──────────────────────────────
CSV_DIR: str     = "example_data"
INDEX_NAME: str  = "sec-chunks-sparse"
NAMESPACE: str   = "sec_10k"
EMBED_MODEL: str = "pinecone-sparse-english-v0"
TEXT_FIELD: str  = "text"
ID_FIELD: str    = "_id"
# ─────────────────────────────────────────────────────────────────────────────

MAX_WORKERS: int            = 2
READ_CHUNK_SIZE: int        = 200
MAX_RETRIES: int            = 8
MAX_RETRY_TIME: int         = 300
INTER_BATCH_DELAY_S: float  = 0.5
STATE_FILE: str             = "upload_state_sparse.json"
MAX_PAYLOAD_BYTES: int      = 2 * 1024 * 1024
MAX_RECORDS_PER_BATCH: int  = 96

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {"_id": str(row[ID_FIELD])}
        for col in df.columns:
            if col == ID_FIELD:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            rec[col] = val
        records.append(rec)
    return records


def batch_by_payload(
    records: list[dict[str, Any]],
    max_bytes: int = MAX_PAYLOAD_BYTES,
    max_records: int = MAX_RECORDS_PER_BATCH,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    batch_bytes = 0
    for rec in records:
        rec_bytes = len(json.dumps(rec, default=str).encode("utf-8"))
        if rec_bytes > max_bytes:
            log.warning("Record '%s' is %s bytes — skipped.", rec.get("_id", "?"), f"{rec_bytes:,}")
            continue
        if batch and (batch_bytes + rec_bytes > max_bytes or len(batch) >= max_records):
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(rec)
        batch_bytes += rec_bytes
    if batch:
        yield batch


def _make_upsert_fn(index: Any, namespace: str):
    @backoff.on_exception(
        backoff.expo, (PineconeException, Exception),
        max_tries=MAX_RETRIES, max_time=MAX_RETRY_TIME, factor=2, jitter=backoff.full_jitter,
        on_backoff=lambda d: log.warning("Retry %s/%s in %.1fs — %s", d["tries"], MAX_RETRIES, d["wait"], d["exception"]),
    )
    def _upsert_batch(batch: list[dict[str, Any]]) -> int:
        index.upsert_records(namespace=namespace, records=batch)
        time.sleep(INTER_BATCH_DELAY_S)
        return len(batch)
    return _upsert_batch


@dataclass
class FileProgress:
    key: str
    total_rows: int
    uploaded_rows: int = 0
    status: str = "pending"


@dataclass
class UploadState:
    files: dict[str, dict] = field(default_factory=dict)

    def save(self, path: str = STATE_FILE) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str = STATE_FILE) -> UploadState:
        p = Path(path)
        if p.exists():
            return cls(files=json.loads(p.read_text()).get("files", {}))
        return cls()

    def get(self, key: str) -> Optional[dict]:
        return self.files.get(key)

    def register(self, fp: FileProgress) -> None:
        self.files[fp.key] = asdict(fp)
        self.save()

    def advance(self, key: str, rows: int) -> None:
        entry = self.files[key]
        entry["uploaded_rows"] += rows
        entry["status"] = "in_progress"
        self.save()

    def done(self, key: str) -> None:
        self.files[key]["status"] = "done"
        self.save()

    def fail(self, key: str, reason: str = "") -> None:
        self.files[key]["status"] = f"failed: {reason}"
        self.save()


def discover_csv_files(directory: str) -> list[str]:
    d = Path(directory)
    if not d.is_dir():
        log.error("CSV_DIR is not a directory: %s", directory)
        sys.exit(1)
    files = sorted(str(f) for f in d.rglob("*.csv"))
    log.info("Found %d CSV file(s) in %s", len(files), directory)
    return files


def count_csv_rows(path: str) -> int:
    with open(path, "r") as f:
        return sum(1 for _ in f) - 1


def upload_chunk(upsert_fn, df: pd.DataFrame, pbar: Optional[tqdm] = None) -> int:
    records = df_to_records(df)
    batches = list(batch_by_payload(records))
    uploaded = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(upsert_fn, b): b for b in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            try:
                n = fut.result()
                uploaded += n
                if pbar:
                    pbar.update(n)
            except Exception as e:
                log.error("Batch of %d records permanently failed: %s", len(batch), e)
    return uploaded


def process_file(upsert_fn, path: str, state: UploadState, overall_pbar: Optional[tqdm] = None) -> None:
    cached = state.get(path)
    if cached and cached["status"] == "done":
        log.info("Skipping completed file: %s", path)
        if overall_pbar:
            overall_pbar.update(cached["total_rows"])
        return

    total_rows = count_csv_rows(path)
    skip_rows = 0

    if cached and cached["status"].startswith("in_progress"):
        skip_rows = cached["uploaded_rows"]
        log.info("Resuming %s from row %s / %s", path, f"{skip_rows:,}", f"{total_rows:,}")
        if overall_pbar:
            overall_pbar.update(skip_rows)
    else:
        state.register(FileProgress(key=path, total_rows=total_rows))

    log.info("Processing %s (%s rows)", path, f"{total_rows:,}")

    try:
        reader = pd.read_csv(path, chunksize=READ_CHUNK_SIZE)
        rows_seen = 0

        for chunk_df in reader:
            chunk_len = len(chunk_df)
            if rows_seen == 0:
                missing = [c for c in [ID_FIELD, TEXT_FIELD] if c not in chunk_df.columns]
                if missing:
                    raise ValueError(f"CSV {path} is missing required column(s): {missing}. Found: {list(chunk_df.columns)}")

            if rows_seen + chunk_len <= skip_rows:
                rows_seen += chunk_len
                continue
            if rows_seen < skip_rows:
                chunk_df = chunk_df.iloc[skip_rows - rows_seen:]
                rows_seen = skip_rows

            uploaded = upload_chunk(upsert_fn, chunk_df, overall_pbar)
            state.advance(path, uploaded)
            rows_seen += chunk_len

        state.done(path)
        log.info("Completed %s", path)

    except Exception as e:
        state.fail(path, str(e))
        log.error("Failed %s: %s", path, e)
        raise


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        log.error("PINECONE_API_KEY not found in environment.")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    existing = {idx.name for idx in pc.list_indexes()}
    if INDEX_NAME not in existing:
        log.info("Creating sparse index '%s' with model '%s'…", INDEX_NAME, EMBED_MODEL)
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={"model": EMBED_MODEL, "field_map": {"text": TEXT_FIELD}},
        )
        log.info("Waiting for index to initialize…")
        while not pc.describe_index(INDEX_NAME).status.get("ready"):
            time.sleep(2)
        log.info("Index ready.")

    index = pc.Index(INDEX_NAME)
    upsert_fn = _make_upsert_fn(index, NAMESPACE)
    files = discover_csv_files(CSV_DIR)
    if not files:
        log.warning("No .csv files found in %s", CSV_DIR)
        sys.exit(0)

    state = UploadState.load()
    total_rows = 0
    for f in tqdm(files, desc="Scanning CSVs", unit="file"):
        cached = state.get(f)
        total_rows += cached["total_rows"] if cached else count_csv_rows(f)

    already_done = sum(
        v["uploaded_rows"] for v in state.files.values()
        if v["status"] in ("done",) or v["status"].startswith("in_progress")
    )

    pbar = tqdm(total=total_rows, initial=already_done, unit="rec", desc="Uploading")
    t0 = time.time()
    failed: list[str] = []

    for f in files:
        try:
            process_file(upsert_fn, f, state, pbar)
        except Exception:
            failed.append(f)

    pbar.close()
    elapsed = time.time() - t0
    done_count = sum(1 for v in state.files.values() if v["status"] == "done")

    print(f"\n{'=' * 60}\nUPLOAD SUMMARY\n{'=' * 60}")
    print(f"  Time elapsed : {elapsed / 60:.1f} min")
    print(f"  Files total  : {len(files)}")
    print(f"  Files done   : {done_count}")
    print(f"  Files failed : {len(failed)}")
    if failed:
        for fp in failed:
            print(f"    - {fp}")
        print("  Re-run to retry failed files.")
    print(f"\n  Index stats  : {index.describe_index_stats()}")


if __name__ == "__main__":
    main()
