# Example Data

Pre-chunked SEC 10-K annual reports for use with the hybrid search demo.

## Schema

Each CSV has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `_id` | string | Unique record ID (`{filing}_{chunk_index}`) |
| `source` | string | Original filename |
| `filing_type` | string | Always `10-k` |
| `ticker` | string | Company ticker, lowercase (e.g. `aapl`) |
| `year` | integer | Fiscal year (e.g. `2023`) |
| `chunk_index` | integer | Position of chunk within the document |
| `text` | string | Chunk text (~768 tokens) |

## Contents

| Company | Tickers | Years |
|---------|---------|-------|
| Apple | `aapl` | 2019–2024 |
| Microsoft | `msft` | 2019–2024 |
| Amazon | `amzn` | 2019–2024 |
| Ford | `f` | 2019–2024 |
| General Motors | `gm` | 2019–2024 |
| Oracle | `orcl` | 2019–2024 |

**Note:** The first ~50 chunks per file are XBRL header metadata (machine-readable financial tagging). These won't rank highly for natural language queries.

## Source

Public domain filings from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar).
All 10-K filings are publicly available with no licensing restrictions.

## Bringing Your Own Data

Any CSV with `_id` and `text` columns works. All other columns become searchable metadata fields.

To adapt the app for different metadata fields, update:
- `RECORD_FIELDS` in `app.py`
- The `build_filter()` function and sidebar inputs in `app.py`
- `NAMESPACE` in all three files (optional but keeps indexes organized)
