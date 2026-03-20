"""
Presentation layer – Streamlit application.
Wires together the business and persistence layers and renders the UI.
"""
import streamlit as st

from business.document_processor import DocumentProcessor
from business.embeddings import EmbeddingService
from business.rag_service import RAGService
from config.settings import settings
from persistence.vector_store import VectorStore

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · Document Q&A",
    page_icon="📚",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Syne', sans-serif !important;
        }

        .main-title {
            font-family: 'Syne', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }

        .subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-top: 0.25rem;
            margin-bottom: 2rem;
        }

        .answer-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            line-height: 1.7;
            font-size: 0.97rem;
            color: #1e293b;
        }

        .chunk-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
            color: #334155;
            line-height: 1.6;
        }

        .chunk-meta {
            font-family: 'Syne', sans-serif;
            font-size: 0.72rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.35rem;
        }

        .score-pill {
            display: inline-block;
            background: #dbeafe;
            color: #1d4ed8;
            border-radius: 99px;
            padding: 1px 8px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-left: 6px;
        }

        .status-ok {
            background: #dcfce7;
            color: #166534;
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .divider {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 1.5rem 0;
        }

        /* Streamlit tweaks */
        .stButton > button {
            background: #0f172a;
            color: white;
            border: none;
            border-radius: 8px;
            font-family: 'Syne', sans-serif;
            font-weight: 700;
            letter-spacing: 0.03em;
            padding: 0.55rem 1.5rem;
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background: #1e40af;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Dependency injection via Streamlit cache
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_services():
    """Instantiate and cache all service objects for the session."""
    embedding_svc = EmbeddingService(settings.openai)
    vector_store = VectorStore(settings.qdrant, settings.openai.embedding_dim)
    doc_processor = DocumentProcessor(settings.chunking)
    rag_svc = RAGService(
        vector_store=vector_store,
        embedding_service=embedding_svc,
        config=settings.openai,
        top_k=settings.top_k_results,
    )
    return doc_processor, embedding_svc, vector_store, rag_svc


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<p class="main-title">📚 RAG · Document Q&A</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload PDF documents, ask questions, and get answers '
    "grounded in your content.</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar – configuration & ingestion
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=settings.openai.api_key,
        type="password",
        help="Your OpenAI secret key (sk-…). Set OPENAI_API_KEY in .env to pre-fill.",
    )
    if openai_key:
        settings.openai.api_key = openai_key

    qdrant_url = st.text_input(
        "Qdrant URL (optional)",
        value=settings.qdrant.url,
        placeholder="https://your-cluster.qdrant.io",
        help="Leave blank to use in-memory Qdrant (data lost on refresh).",
    )
    qdrant_key = st.text_input(
        "Qdrant API Key (optional)",
        value=settings.qdrant.api_key,
        type="password",
    )
    if qdrant_url:
        settings.qdrant.url = qdrant_url
        settings.qdrant.api_key = qdrant_key
        settings.qdrant.use_in_memory = False

    st.markdown("---")
    st.markdown("### 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    ingest_btn = st.button("⚡ Process & Index Documents", use_container_width=True)

    if ingest_btn:
        if not settings.openai.api_key:
            st.error("Please enter your OpenAI API key first.")
        elif not uploaded_files:
            st.warning("Upload at least one PDF before indexing.")
        else:
            doc_processor, embedding_svc, vector_store, rag_svc = get_services()

            with st.spinner("Extracting text and chunking…"):
                file_pairs = [(f.name, f.read()) for f in uploaded_files]
                chunks = doc_processor.process_multiple(file_pairs)

            st.write(f"✅ **{len(chunks)} chunks** extracted from {len(file_pairs)} file(s).")

            with st.spinner(f"Embedding {len(chunks)} chunks via OpenAI…"):
                texts = [c.text for c in chunks]
                embeddings = embedding_svc.embed_batch(texts)

            with st.spinner("Storing vectors in Qdrant…"):
                vector_store.reset()
                vector_store.upsert_chunks(chunks, embeddings)

            st.success(f"Index ready — {vector_store.count()} vectors stored.")
            st.session_state["indexed"] = True
            st.session_state["doc_names"] = [f.name for f in uploaded_files]

    # Show indexed state
    if st.session_state.get("indexed"):
        st.markdown(
            '<span class="status-ok">✓ Index loaded</span>', unsafe_allow_html=True
        )
        for name in st.session_state.get("doc_names", []):
            st.caption(f"• {name}")

    st.markdown("---")
    st.caption(
        "Model: `text-embedding-3-small` (embeddings) · "
        "`gpt-4o-mini` (answers)  \nVector DB: Qdrant"
    )

# ──────────────────────────────────────────────
# Main content – Q&A interface
# ──────────────────────────────────────────────
col_q, col_spacer = st.columns([3, 1])

with col_q:
    question = st.text_input(
        "💬 Ask a question about your documents",
        placeholder="e.g. What are the main exclusions of the policy?",
    )

ask_btn = st.button("🔍 Search & Answer")

if ask_btn:
    if not st.session_state.get("indexed"):
        st.warning("Please upload and index documents first (use the sidebar).")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif not settings.openai.api_key:
        st.error("OpenAI API key is required.")
    else:
        _, _, _, rag_svc = get_services()

        with st.spinner("Retrieving relevant chunks and generating answer…"):
            result = rag_svc.query(question)

        # ── Answer ──
        st.markdown("#### 🤖 Answer")
        st.markdown(
            f'<div class="answer-box">{result.answer}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Retrieved context ──
        st.markdown(f"#### 🗂️ Retrieved Context ({len(result.retrieved_chunks)} chunks)")

        for i, chunk in enumerate(result.retrieved_chunks, 1):
            score_pct = f"{chunk.score * 100:.1f}%"
            st.markdown(
                f"""
                <div class="chunk-card">
                    <div class="chunk-meta">
                        Chunk {i} · {chunk.source}
                        <span class="score-pill">similarity {score_pct}</span>
                    </div>
                    {chunk.text}
                </div>
                """,
                unsafe_allow_html=True,
            )
