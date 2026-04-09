# Hybrid Search Pipeline — Implementation Guide

This guide walks through building a complete hybrid search pipeline over pre-chunked SEC 10-K filing data. By the end, you will have:

- Two Pinecone indexes (dense + sparse) populated with the example data
- A 4-tab Streamlit app for exploring the pipeline at each stage

**Full pipeline:** Dense query + Sparse query → Weighted RRF → BGE Reranker → MMR → top-25 results.

---

## Architecture

```
example_data/                 ← pre-chunked CSV files (one per filing)
  └─ {filing_type}_{ticker}_{year}.csv

Pinecone:
  ├─ sec-chunks-dense         ← dense index (llama-text-embed-v2, cosine)
  └─ sec-chunks-sparse        ← sparse index (pinecone-sparse-english-v0)

app.py                        ← Streamlit app (4 tabs)
write-to-pinecone-dense.py    ← CSV → dense Pinecone index
write-to-pinecone-sparse.py   ← CSV → sparse Pinecone index
```

---

## Prerequisites

- **Python 3.11–3.13** managed via `uv`
- **Pinecone account** with inference enabled — get an API key at [pinecone.io](https://pinecone.io)
  > The BGE reranker (`bge-reranker-v2-m3`) and server-side embeddings require a plan that includes inference. Check your plan at the Pinecone console.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your Pinecone API key
```

---

## Data

Pre-chunked 10-K annual reports live in `example_data/`. Each CSV has the schema:

```
_id            unique record ID
source         original filename
filing_type    e.g. "10-k"
ticker         lowercase company ticker, e.g. "aapl"
year           integer fiscal year, e.g. 2023
chunk_index    position within the document
text           chunk content (~768 tokens)
```

Companies included: AAPL, MSFT, AMZN, F, GM, ORCL — years 2019–2024.

See `example_data/README.md` for full details.

---

## Step 1 — Upload to Dense Index

**File:** `write-to-pinecone-dense.py`

```python
"""
Upload records from a directory of CSV files to a Pinecone integrated inference index.
Pinecone embeds the `text` column server-side — no local embedding needed.
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
INDEX_NAME: str  = "sec-chunks-dense"
NAMESPACE: str   = "sec_10k"
METRIC: str      = "cosine"
EMBED_MODEL: str = "llama-text-embed-v2"
TEXT_FIELD: str  = "text"
ID_FIELD: str    = "_id"
# ─────────────────────────────────────────────────────────────────────────────

MAX_WORKERS: int            = 2
READ_CHUNK_SIZE: int        = 200
MAX_RETRIES: int            = 8
MAX_RETRY_TIME: int         = 300
INTER_BATCH_DELAY_S: float  = 0.5
STATE_FILE: str             = "upload_state_dense.json"
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
        log.info("Creating integrated inference index '%s' with model '%s'…", INDEX_NAME, EMBED_MODEL)
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
```

**Run:**
```bash
uv run python write-to-pinecone-dense.py
```

**What it does:**
- Creates the dense index automatically if it doesn't exist, using Pinecone integrated inference — `llama-text-embed-v2` embeds the `text` field server-side, no local GPU needed
- Streams CSVs in 200-row chunks to keep memory low
- Batches records under the 2 MB / 96-record Pinecone upsert limits
- Retries failed batches with exponential backoff (up to 8 tries, 300s window)
- Writes `upload_state_dense.json` for crash recovery — safe to kill and restart
- Runs 2 parallel upsert threads (kept low to avoid token-per-minute limits on server-side embedding)

**Upload time:** expect 5–15 minutes for the full dataset (36 files, ~20k chunks total) depending on your plan's throughput limits.

---

## Step 2 — Upload to Sparse Index

**File:** `write-to-pinecone-sparse.py`

Same structure as the dense uploader with these differences:

```python
INDEX_NAME: str  = "sec-chunks-sparse"
EMBED_MODEL: str = "pinecone-sparse-english-v0"
STATE_FILE: str  = "upload_state_sparse.json"
```

The sparse model (`pinecone-sparse-english-v0`) uses the DeepImpact architecture for learned sparse retrieval — better recall than BM25 for out-of-vocabulary terms while remaining keyword-interpretable.

**Run:**
```bash
uv run python write-to-pinecone-sparse.py
```

> **Note:** Both indexes share the same document IDs (`_id` column). This is critical — the pipeline fetches dense vectors by ID from the dense index for documents that only appeared in sparse results. Mismatched IDs will silently drop those documents from MMR.

---

## Step 3 — The Streamlit App

**File:** `app.py`

```python
"""
Streamlit app: Dense vs Sparse search + full hybrid pipeline.
  Dense + Sparse → Weighted RRF → BGE Reranker → MMR diversification.
"""

import json
import os

import dotenv
import numpy as np
import streamlit as st
from pinecone import Pinecone

dotenv.load_dotenv()

# ── Constants — adapt these to your project ───────────────────────────────────
DENSE_INDEX    = "sec-chunks-dense"
SPARSE_INDEX   = "sec-chunks-sparse"
NAMESPACE      = "sec_10k"
RECORD_FIELDS  = ["text", "source", "filing_type", "ticker", "year", "chunk_index"]
DENSE_MODEL    = "llama-text-embed-v2"
SPARSE_MODEL   = "pinecone-sparse-english-v0"
RERANKER_MODEL = "bge-reranker-v2-m3"
# ─────────────────────────────────────────────────────────────────────────────

# ... (full source in app.py)
```

**Run:**
```bash
uv run streamlit run app.py
```

**Tabs:**

| Tab | What it shows |
|-----|--------------|
| Dense & Sparse | Side-by-side: semantic (dense) vs keyword (sparse) results |
| RRF Fusion | Candidates merged via Weighted Reciprocal Rank Fusion |
| Reranker | BGE cross-encoder scores on the fused set |
| Full Pipeline (MMR) | Final top-25 after MMR diversification with interactive λ slider |

**Sidebar controls:**
- `top_k` — how many results to show in the Dense & Sparse tab
- `fetch_k` — candidates per leg fed into the pipeline (Tab 2–4)
- Metadata filters — `ticker` (e.g. `aapl`), `year` (e.g. `2023`), `filing_type` (e.g. `10-k`)
- RRF weights — dense/sparse balance (default 0.4 / 0.6)

**Try these queries to see the pipeline in action:**

| Query | What to observe |
|-------|----------------|
| `What are Apple's main risk factors?` | Dense returns conceptual matches; sparse latches on to "Apple", "risk" |
| `capital allocation strategy` | Dense wins — semantic understanding of the concept |
| `AWS revenue growth` | Sparse wins — exact keyword match for the acronym |
| `competition from Chinese manufacturers` | Both legs contribute; RRF fusion improves coverage |

---

## Pipeline Explained

### Why two separate indexes?

Pinecone integrated inference creates index-type-specific embeddings server-side. Dense indexes store high-dimensional float vectors (semantic meaning); sparse indexes store token-weight maps (keyword frequency). Querying them separately and fusing results gives better coverage than either alone.

### Weighted RRF

RRF avoids combining incomparable scores (a dense cosine similarity of 0.92 vs a sparse score of 14.7 mean nothing relative to each other). Instead it uses only **rank** — position in each sorted list — and combines with per-leg weights:

```
score(d) = w_dense × 1/(k + rank_dense) + w_sparse × 1/(k + rank_sparse)
```

`k=60` is the standard constant from the original RRF paper — it dampens the advantage of rank-1 vs rank-2. Weights default to 0.4 dense / 0.6 sparse and can be tuned via the sidebar slider.

### BGE Reranker

The BGE cross-encoder (`bge-reranker-v2-m3`) re-scores each (query, document) pair jointly. Unlike bi-encoders (which embed query and doc independently), cross-encoders see both at once and are far more accurate — but too slow to run over a full corpus. Running it over the top-50 RRF candidates is the standard compromise.

### MMR

MMR iteratively selects the next result that maximises:

```
MMR(i) = λ × relevance(i) − (1−λ) × max_{j∈S} cosine(i, j)
```

where `S` is the already-selected set. At λ=1 it degenerates to the reranker order. At λ=0 it maximises diversity (useful for avoiding 10 near-identical passages from the same document). The interactive slider lets you explore this tradeoff without re-querying Pinecone.

Dense vectors for all candidates are needed for inter-document cosine similarity. Documents that appeared only in the sparse leg have their vectors back-filled via `dense_index.fetch()`.

---

## Adapting to Different Data

| What to change | Where |
|---------------|-------|
| Index names | `DENSE_INDEX`, `SPARSE_INDEX` at the top of `app.py` and the two uploader scripts |
| Namespace | `NAMESPACE` in all three files |
| Data directory | `CSV_DIR` in both uploader scripts |
| Metadata fields | `RECORD_FIELDS` in `app.py`; `build_filter()` sidebar inputs; `render_*_hits()` display functions |
| Dense embedding model | `EMBED_MODEL` in `write-to-pinecone-dense.py` and `DENSE_MODEL` in `app.py` |

### Minimum required CSV schema

```
_id        unique record ID
text       content to embed and search
```

All other columns are passed through as metadata and become filterable in the app.

---

## Troubleshooting

**`sparse_indices` / `sparse_values` AttributeError** — The SDK may name sparse embedding attributes differently across versions. Inspect `sparse_emb[0].__dict__` or `dir(sparse_emb[0])` and update `get_query_embeddings()` in `app.py`.

**Reranker fails** — `bge-reranker-v2-m3` requires a Pinecone plan that includes inference. Check your plan at [app.pinecone.io](https://app.pinecone.io). As a workaround, comment out the reranker step and wire Tab 3 to display the RRF results.

**MMR returns fewer than 25 results** — Documents missing dense vectors (sparse-only hits where `fetch()` also failed) are silently skipped. Verify `include_values=True` is working on the dense index query.

**Upload rate-limit errors (429)** — Reduce `MAX_WORKERS` to 1, increase `INTER_BATCH_DELAY_S` to 1.0. The script retries with backoff so it will eventually complete without changes. State is persisted — re-running resumes from where it left off.

**Crash mid-upload** — Re-run the script. The `upload_state_*.json` files track per-file progress and skip completed files automatically.
