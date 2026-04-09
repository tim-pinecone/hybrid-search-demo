# Pinecone Hybrid Search Demo

A 4-tab Streamlit app that walks through every stage of a hybrid search pipeline over SEC 10-K annual reports.

| Tab | What it shows |
|-----|--------------|
| Dense & Sparse | Side-by-side: semantic (dense) vs keyword (sparse) results |
| RRF Fusion | Both result sets merged via Weighted Reciprocal Rank Fusion |
| Reranker | BGE cross-encoder re-scores the fused candidates |
| Full Pipeline (MMR) | Top-25 results after Maximal Marginal Relevance diversification with an interactive λ slider |

**Full pipeline:** Dense query + Sparse query → Weighted RRF (0.4/0.6) → BGE Reranker → MMR → top-25

![Demo](https://github.com/tim-pinecone/hybrid-search-demo/releases/download/v1.0/demo.mp4)

---

## Architecture

```
example_data/                 ← pre-chunked SEC 10-K filings (CSV)

Pinecone:
  ├─ sec-chunks-dense         ← llama-text-embed-v2 (cosine)
  └─ sec-chunks-sparse        ← pinecone-sparse-english-v0

app.py                        ← Streamlit app
write-to-pinecone-dense.py    ← upload script (dense)
write-to-pinecone-sparse.py   ← upload script (sparse)
```

---

## Prerequisites

- **Python 3.11–3.13** + **[uv](https://docs.astral.sh/uv/)** (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Pinecone account with inference enabled** — [pinecone.io](https://pinecone.io)
  > Server-side embeddings and the BGE reranker require a plan that includes inference. Verify at [app.pinecone.io](https://app.pinecone.io).

---

## Quick Start

```bash
git clone <this-repo>
cd <repo>

cp .env.example .env
# Add your PINECONE_API_KEY to .env

uv sync

# Upload example data to both indexes (~5–15 min depending on plan throughput)
uv run python write-to-pinecone-dense.py
uv run python write-to-pinecone-sparse.py

# Launch the app
uv run streamlit run app.py
```

---

## Use with Claude Code

Clone the repo, open it in Claude Code, and run:

```
Read GUIDE.md and build the full hybrid search pipeline using the example data
in example_data/. Install dependencies, upload to both Pinecone indexes, and
launch the Streamlit app.
```

Claude Code will handle setup, upload, and launch autonomously.

---

## Example Queries

| Query | What to observe |
|-------|----------------|
| `What are Apple's main risk factors?` | Dense returns conceptual matches; sparse catches "Apple", "risk" |
| `capital allocation strategy` | Dense wins — semantic understanding of the concept |
| `AWS revenue growth` | Sparse wins — exact keyword match for the acronym |
| `competition from Chinese manufacturers` | Both legs contribute; RRF fusion improves coverage |

---

## Example Data

Pre-chunked SEC 10-K filings for AAPL, MSFT, AMZN, F, GM, ORCL — years 2019–2024.
All public domain via [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar).

See `example_data/README.md` for schema details and instructions on bringing your own data.

---

## Full Documentation

See **[GUIDE.md](GUIDE.md)** for the complete implementation walkthrough, pipeline explanation, and adaptation instructions.
