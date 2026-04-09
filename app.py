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


@st.cache_resource
def get_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("PINECONE_API_KEY not set in environment / .env")
        st.stop()
    pc = Pinecone(api_key=api_key)
    return pc, pc.Index(DENSE_INDEX), pc.Index(SPARSE_INDEX)


# ── Dense & Sparse tab helpers (integrated inference) ────────────────────────

def search(index, query: str, top_k: int, namespace: str, filters: dict | None) -> dict:
    params: dict = {
        "namespace": namespace,
        "query": {"inputs": {"text": query}, "top_k": top_k},
        "fields": RECORD_FIELDS,
    }
    if filters:
        params["query"]["filter"] = filters
    return index.search(**params).to_dict()


def build_filter(ticker: str, year: str, filing_type: str) -> dict | None:
    f: dict = {}
    if ticker:
        # Metadata is stored lowercase (e.g. "aapl", "msft")
        f["ticker"] = {"$eq": ticker.lower().strip()}
    if year:
        # Year is stored as an integer in Pinecone metadata
        try:
            f["year"] = {"$eq": int(year.strip())}
        except ValueError:
            f["year"] = {"$eq": year.strip()}
    if filing_type:
        # Filing type is stored lowercase (e.g. "10-k")
        f["filing_type"] = {"$eq": filing_type.lower().strip()}
    return f if f else None


def strip_vectors(result: dict) -> dict:
    cleaned = json.loads(json.dumps(result))
    for hit in cleaned.get("result", {}).get("hits", []):
        hit.pop("values", None)
        hit.pop("sparse_values", None)
    return cleaned


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def get_query_embeddings(pc: Pinecone, query: str) -> tuple[list[float], dict]:
    """Return (dense_vector, sparse_vector) for the query."""
    dense_emb = pc.inference.embed(
        model=DENSE_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    dense_vector = dense_emb[0].values

    sparse_emb = pc.inference.embed(
        model=SPARSE_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    s = sparse_emb[0]
    sparse_vector = {"indices": s.sparse_indices, "values": s.sparse_values}
    return dense_vector, sparse_vector


def query_both_legs(
    dense_index, sparse_index,
    dense_vector: list[float], sparse_vector: dict,
    fetch_k: int, namespace: str, filters: dict | None,
) -> tuple[list[dict], list[dict]]:
    def parse_hits(response, has_values: bool) -> list[dict]:
        return [
            {"id": m.id, "score": m.score, "metadata": m.metadata or {}, "values": m.values if has_values else []}
            for m in response.matches
        ]

    dense_resp = dense_index.query(
        vector=dense_vector, top_k=fetch_k, namespace=namespace,
        include_values=True, include_metadata=True, filter=filters,
    )
    sparse_resp = sparse_index.query(
        sparse_vector=sparse_vector, top_k=fetch_k, namespace=namespace,
        include_metadata=True, filter=filters,
    )
    return parse_hits(dense_resp, True), parse_hits(sparse_resp, False)


def fetch_dense_vectors(dense_index, ids: list[str], namespace: str) -> dict[str, list[float]]:
    result = dense_index.fetch(ids=ids, namespace=namespace)
    return {vid: vec.values for vid, vec in result.vectors.items()}


def weighted_rrf(
    dense_hits: list[dict], sparse_hits: list[dict],
    dense_weight: float = 0.4, sparse_weight: float = 0.6,
    k: int = 60, top_n: int = 50,
) -> list[dict]:
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}
    origins: dict[str, set]  = {}

    for rank, hit in enumerate(dense_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (k + rank + 1)
        docs[doc_id] = hit
        origins.setdefault(doc_id, set()).add("D")

    for rank, hit in enumerate(sparse_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight / (k + rank + 1)
        if doc_id not in docs:
            docs[doc_id] = hit
        origins.setdefault(doc_id, set()).add("S")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {"id": doc_id, "rrf_score": score, "legs": "+".join(sorted(origins[doc_id])), **docs[doc_id]}
        for doc_id, score in ranked
    ]


def rerank_with_bge(pc: Pinecone, query: str, candidates: list[dict], top_n: int = 50) -> list[dict]:
    documents = [c["metadata"].get("text", "") for c in candidates]
    result = pc.inference.rerank(
        model=RERANKER_MODEL, query=query, documents=documents,
        top_n=top_n, return_documents=True,
    )
    return [{**candidates[item.index], "rerank_score": item.score} for item in result.data]


def mmr(
    candidates: list[dict], query_embedding: list[float],
    top_n: int = 25, lambda_: float = 0.5,
) -> list[dict]:
    def cosine(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    valid = [c for c in candidates if c.get("values")]
    if not valid:
        return candidates[:top_n]

    raw_scores = np.array([c["rerank_score"] for c in valid], dtype=float)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    scores = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else np.ones_like(raw_scores)

    selected: list[int] = []
    remaining = list(range(len(valid)))

    for _ in range(min(top_n, len(valid))):
        best_idx, best_score = None, -np.inf
        for i in remaining:
            relevance  = lambda_ * scores[i]
            redundancy = (
                max(cosine(valid[i]["values"], valid[j]["values"]) for j in selected)
                if selected else 0.0
            )
            mmr_score = relevance - (1 - lambda_) * redundancy
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [valid[i] for i in selected]


# ── Display helpers ───────────────────────────────────────────────────────────

def render_search_hits(hits: list, show_raw_json: bool) -> None:
    """Render hits from index.search() — fields-format (Tab 1)."""
    if not hits:
        st.info("No results.")
        return
    for i, hit in enumerate(hits, 1):
        score   = hit.get("_score", hit.get("score", 0))
        fields  = hit.get("fields", {})
        text    = fields.get("text", "")
        ticker_ = fields.get("ticker", "")
        year_   = fields.get("year", "")
        source  = fields.get("source", "")
        chunk_  = fields.get("chunk_index", "")
        year_str  = str(int(float(year_))) if year_ else ""
        chunk_str = str(int(float(chunk_))) if chunk_ != "" else "—"
        header = f"#{i} · score `{score:.4f}`"
        if ticker_ or year_str:
            header += f"  ·  {ticker_.upper()} {year_str}"
        with st.expander(header, expanded=(i == 1)):
            mc = st.columns(3)
            mc[0].metric("Ticker", ticker_.upper() if ticker_ else "—")
            mc[1].metric("Year",   year_str or "—")
            mc[2].metric("Score",  f"{score:.4f}")
            st.caption(f"Source: {source}  ·  Chunk: {chunk_str}")
            st.markdown("**Text**")
            st.write(text)
            if show_raw_json:
                st.json(strip_vectors({"hit": hit}))


def render_pipeline_hits(
    hits: list[dict], score_key: str, score_label: str,
    show_raw_json: bool, extra_badge: str | None = None,
) -> None:
    """Render hits from the pipeline — metadata-format (Tabs 2–4)."""
    if not hits:
        st.info("No results.")
        return
    for i, hit in enumerate(hits, 1):
        score   = hit.get(score_key, 0)
        meta    = hit.get("metadata", {})
        text    = meta.get("text", "")
        ticker_ = meta.get("ticker", "")
        year_   = meta.get("year", "")
        source  = meta.get("source", "")
        chunk_  = meta.get("chunk_index", "")
        year_str  = str(int(float(year_))) if year_ else ""
        chunk_str = str(int(float(chunk_))) if chunk_ != "" else "—"
        header = f"#{i} · {score_label} `{score:.4f}`"
        if ticker_ or year_str:
            header += f"  ·  {ticker_.upper()} {year_str}"
        if extra_badge and hit.get(extra_badge):
            header += f"  ·  [{hit[extra_badge]}]"
        with st.expander(header, expanded=(i == 1)):
            n_cols = 4 if extra_badge else 3
            mc = st.columns(n_cols)
            mc[0].metric("Ticker", ticker_.upper() if ticker_ else "—")
            mc[1].metric("Year",   year_str or "—")
            mc[2].metric(score_label, f"{score:.4f}")
            if extra_badge and n_cols == 4:
                mc[3].metric("Legs", hit.get(extra_badge, "—"))
            st.caption(f"Source: {source}  ·  Chunk: {chunk_str}")
            st.markdown("**Text**")
            st.write(text)
            if show_raw_json:
                st.json({k: v for k, v in hit.items() if k != "values"})


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Hybrid Search Pipeline", page_icon="📄", layout="wide")
st.title("📄 Hybrid Search Pipeline")
st.caption(
    f"Dense: `{DENSE_INDEX}` ({DENSE_MODEL})  ·  "
    f"Sparse: `{SPARSE_INDEX}` ({SPARSE_MODEL})  ·  "
    f"Reranker: `{RERANKER_MODEL}`"
)

pc, dense_index, sparse_index = get_pinecone()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")
    top_k = st.slider("top_k  (Dense & Sparse tab)", min_value=1, max_value=50, value=5, step=1)
    fetch_k = st.slider("fetch_k  (pipeline — candidates per leg)", min_value=10, max_value=200, value=100, step=10)
    namespace = st.text_input("Namespace", value=NAMESPACE)

    st.subheader("Metadata filters (optional)")
    # ── Adapt these filter inputs to your metadata fields ─────────────────────
    ticker      = st.text_input("Ticker (e.g. aapl, msft, amzn)")
    year        = st.text_input("Year (e.g. 2023)")
    filing_type = st.text_input("Filing type (e.g. 10-k)")
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("RRF weights")
    st.caption("Applied in Tabs 2–4")
    rrf_dense_weight  = st.slider("Dense weight",  0.0, 1.0, 0.4, 0.05)
    rrf_sparse_weight = 1.0 - rrf_dense_weight
    st.write(f"Sparse weight: **{rrf_sparse_weight:.2f}**")

    show_raw_json = st.toggle("Show full JSON response", value=False)

# ── Query input ───────────────────────────────────────────────────────────────

query = st.text_area(
    "Question / query",
    placeholder="e.g. What are the main risk factors for Apple?",
    height=80,
)
search_btn = st.button("Search", type="primary", use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔵🟠 Dense & Sparse",
    "⚡ RRF Fusion",
    "🎯 Reranker",
    "✨ Full Pipeline (MMR)",
])

# ── Run search on button click ────────────────────────────────────────────────

if search_btn and not query.strip():
    st.warning("Enter a query first.")

elif search_btn and query.strip():
    filters = build_filter(ticker, year, filing_type)

    with st.spinner("Querying Pinecone…"):
        try:
            dense_result  = search(dense_index,  query, top_k, namespace, filters)
            sparse_result = search(sparse_index, query, top_k, namespace, filters)
        except Exception as e:
            st.error(f"Dense/Sparse search failed: {e}")
            st.stop()

        st.session_state.dense_result  = dense_result
        st.session_state.sparse_result = sparse_result

        try:
            dense_vector, sparse_vector = get_query_embeddings(pc, query)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

        try:
            dense_hits, sparse_hits = query_both_legs(
                dense_index, sparse_index, dense_vector, sparse_vector,
                fetch_k, namespace, filters,
            )
        except Exception as e:
            st.error(f"Pipeline query failed: {e}")
            st.stop()

        fused = weighted_rrf(dense_hits, sparse_hits, dense_weight=rrf_dense_weight, sparse_weight=rrf_sparse_weight)

        missing_ids = [h["id"] for h in fused if not h.get("values")]
        if missing_ids:
            try:
                fetched = fetch_dense_vectors(dense_index, missing_ids, namespace)
                for h in fused:
                    if not h.get("values") and h["id"] in fetched:
                        h["values"] = fetched[h["id"]]
            except Exception:
                pass

        try:
            reranked = rerank_with_bge(pc, query, fused, top_n=50)
        except Exception as e:
            st.error(f"Reranking failed: {e}")
            st.stop()

        st.session_state.pipeline = {
            "query":                 query,
            "query_dense_embedding": dense_vector,
            "dense_hits":            dense_hits,
            "sparse_hits":           sparse_hits,
            "fused":                 fused,
            "reranked":              reranked,
        }

# ── Tab 1: Dense & Sparse ─────────────────────────────────────────────────────

with tab1:
    if "dense_result" not in st.session_state:
        st.info("Run a search to see results.")
    else:
        col_dense, col_sparse = st.columns(2, gap="large")
        for col, label, result in [
            (col_dense,  "🔵 Dense",  st.session_state.dense_result),
            (col_sparse, "🟠 Sparse", st.session_state.sparse_result),
        ]:
            hits = result.get("result", {}).get("hits", [])
            with col:
                st.subheader(f"{label}  —  {len(hits)} hit(s)")
                render_search_hits(hits, show_raw_json)
                if show_raw_json:
                    with st.expander(f"Full {label} response JSON", expanded=False):
                        st.json(strip_vectors(result))

# ── Tab 2: RRF Fusion ─────────────────────────────────────────────────────────

with tab2:
    if "pipeline" not in st.session_state:
        st.info("Run a search to see results.")
    else:
        p = st.session_state.pipeline
        fused = p["fused"]
        st.subheader(f"Weighted RRF — {len(fused)} fused candidates")
        st.caption(
            f"Dense weight: **{rrf_dense_weight:.2f}**  ·  Sparse weight: **{rrf_sparse_weight:.2f}**  ·  "
            "k=60  ·  Leg badges: D=dense only, S=sparse only, D+S=both"
        )
        render_pipeline_hits(fused, "rrf_score", "RRF", show_raw_json, extra_badge="legs")

# ── Tab 3: Reranker ───────────────────────────────────────────────────────────

with tab3:
    if "pipeline" not in st.session_state:
        st.info("Run a search to see results.")
    else:
        reranked = st.session_state.pipeline["reranked"]
        st.subheader(f"BGE Reranker (`{RERANKER_MODEL}`) — {len(reranked)} results")
        st.caption("Cross-encoder scores; higher = more relevant to query.")
        render_pipeline_hits(reranked, "rerank_score", "Rerank", show_raw_json)

# ── Tab 4: Full Pipeline (MMR) ────────────────────────────────────────────────

with tab4:
    if "pipeline" not in st.session_state:
        st.info("Run a search to see results.")
    else:
        p = st.session_state.pipeline
        st.subheader("Full Pipeline: Dense + Sparse → RRF → Rerank → MMR")
        st.markdown(
            "Adjust **λ** to trade off relevance vs. diversity. "
            "Moving the slider re-applies MMR on the cached reranked results — no new Pinecone calls."
        )
        lambda_ = st.slider(
            "λ  (1.0 = pure relevance · 0.0 = pure diversity)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="mmr_lambda",
        )
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("λ", f"{lambda_:.2f}")
        col_b.metric("Reranked candidates in", len(p["reranked"]))

        final = mmr(p["reranked"], p["query_dense_embedding"], top_n=25, lambda_=lambda_)
        col_c.metric("MMR results out", len(final))

        st.caption(
            f"λ={lambda_:.2f}: "
            + ("pure relevance — identical to reranker order" if lambda_ == 1.0
               else "pure diversity — ignores relevance scores" if lambda_ == 0.0
               else f"balanced blend ({lambda_:.0%} relevance / {1-lambda_:.0%} diversity)")
        )
        render_pipeline_hits(final, "rerank_score", "Rerank", show_raw_json)
