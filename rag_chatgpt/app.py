# app.py
import pickle
import time
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import numpy as np

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
DATA_FILE   = Path("data/index.pkl")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-3.5-turbo"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

/* Main header */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-header { color: #a0aec0; font-size: 0.95rem; margin-bottom: 1.5rem; }

/* Chat messages */
.chat-user {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 1px solid #667eea55;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
    text-align: right;
}
.chat-bot {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #30305a;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
}
.chat-label-user { color: #667eea; font-size: 0.75rem; font-weight: 600; text-align: right; margin-bottom: 4px; }
.chat-label-bot  { color: #764ba2; font-size: 0.75rem; font-weight: 600; margin-bottom: 4px; }

/* Source card */
.source-card {
    background: #1a1a2e;
    border: 1px solid #30305a;
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #a0aec0;
    font-size: 0.85rem;
}
.source-card .src-title { color: #667eea; font-weight: 600; font-size: 0.9rem; }

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #30305a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-num  { font-size: 1.8rem; font-weight: 700; color: #667eea; }
.stat-label{ font-size: 0.8rem; color: #718096; margin-top: 4px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0f0c29 !important; border-right: 1px solid #30305a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Input styling */
.stTextInput > div > div > input {
    background: #1a1a2e !important;
    border: 1px solid #30305a !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    padding: 14px 18px !important;
    font-size: 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px #667eea33 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #1a1a2e !important;
    border: 1px solid #30305a !important;
    border-radius: 8px !important;
    color: #a0aec0 !important;
}

/* Divider */
hr { border-color: #30305a !important; }

/* Badge */
.badge {
    display: inline-block;
    background: #667eea22;
    border: 1px solid #667eea55;
    color: #667eea;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-green  { background: #22c55e22; border-color: #22c55e55; color: #22c55e; }
.badge-orange { background: #f59e0b22; border-color: #f59e0b55; color: #f59e0b; }
.badge-red    { background: #ef444422; border-color: #ef444455; color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ── Try initialising OpenAI client gracefully ────────────────────────────────
try:
    client = OpenAI()
    _openai_available = True
except Exception:
    client = None
    _openai_available = False

# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_index():
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading local embedding model…")
def get_local_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading local language model…")
def get_local_llm():
    from transformers import pipeline
    return pipeline("text-generation", model="distilgpt2", truncation=True)

# ── Embedding ────────────────────────────────────────────────────────────────
def embed_query(text: str, is_local: bool = False) -> np.ndarray:
    if is_local or not _openai_available:
        model = get_local_embed_model()
        embs = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return np.array(embs[0], dtype="float32")
    try:
        res = client.embeddings.create(model=EMBED_MODEL, input=[text])
        return np.array(res.data[0].embedding, dtype="float32")
    except Exception:
        model = get_local_embed_model()
        embs = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return np.array(embs[0], dtype="float32")

# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 4) -> list:
    bundle = load_index()
    index  = bundle["index"]
    chunks = bundle["chunks"]
    meta   = bundle["meta"]
    is_local = bundle.get("dim") == 384

    qvec = embed_query(query, is_local).reshape(1, -1)
    distances, ids = index.search(qvec, top_k)

    results = []
    for rank, idx in enumerate(ids[0]):
        if idx == -1:
            continue
        results.append({
            "text":     chunks[idx],
            "source":   meta[idx]["source"],
            "chunk_id": meta[idx]["chunk_id"],
            "distance": float(distances[0][rank]),
            "score":    round(max(0, 1 - float(distances[0][rank]) / 10) * 100, 1),
        })
    return results

def build_context(results: list) -> str:
    parts = []
    for r in results:
        parts.append(f"[Source: {r['source']} | Chunk: {r['chunk_id']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)

# ── Answer generation ────────────────────────────────────────────────────────
def generate_answer(question: str, context: str, chat_history: list,
                    temperature: float = 0.3, max_tokens: int = 512,
                    force_local: bool = False) -> tuple[str, str]:
    """Returns (answer_text, model_used)."""

    history_msgs = []
    for turn in chat_history[-6:]:           # last 3 turns for context
        history_msgs.append({"role": "user",      "content": turn["question"]})
        history_msgs.append({"role": "assistant", "content": turn["answer"]})

    system_msg = (
        "You are a knowledgeable assistant. Answer the user's question "
        "using ONLY the provided document context. "
        "Be concise, accurate, and cite the source when possible. "
        "If the context is insufficient, say so honestly."
    )
    messages = [
        {"role": "system", "content": system_msg},
        *history_msgs,
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]

    if not force_local and _openai_available and client:
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content, f"OpenAI · {CHAT_MODEL}"
        except Exception as e:
            err = str(e).lower()
            if "quota" not in err and "429" not in err and "insufficient" not in err:
                return f"**OpenAI Error:** {e}", "error"
            # fall through to local

    # Local fallback
    try:
        llm    = get_local_llm()
        prompt = (
            f"Context: {context[:1200]}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        res = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, pad_token_id=50256)
        full_text = res[0]["generated_text"]
        # Strip the prompt — return only the generated continuation
        answer_text = full_text[len(prompt):].strip()
        if not answer_text:
            answer_text = full_text.strip()
        return answer_text, "Local · distilgpt2"
    except Exception as local_e:
        return (
            f"Both OpenAI and the local model failed.\n\n"
            f"Local error: {local_e}\n\n"
            f"**Retrieved context:**\n{context[:1000]}",
            "error"
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG Assistant")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")

    top_k = st.slider("Top-K results", 1, 10, 4,
                       help="Number of document chunks to retrieve per query")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
                            help="Higher = more creative answers")
    max_tokens = st.slider("Max answer length (tokens)", 64, 1024, 512, 64)
    force_local = st.toggle("🔒 Force local model (no OpenAI)",
                             value=(not _openai_available),
                             help="Use the local flan-t5 model even if OpenAI is available")

    st.markdown("---")
    st.markdown("### 📊 Index Info")

    if DATA_FILE.exists():
        bundle = load_index()
        num_chunks = len(bundle["chunks"])
        dim        = bundle.get("dim", "?")
        embed_mod  = bundle.get("model", "local")
        sources    = list({m["source"] for m in bundle["meta"]})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{num_chunks}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{len(sources)}</div><div class="stat-label">Docs</div></div>', unsafe_allow_html=True)

        st.markdown(f"**Embed model:** `{embed_mod}`")
        st.markdown(f"**Vector dim:** `{dim}`")

        st.markdown("**Documents:**")
        for s in sources:
            st.markdown(f"- 📄 `{s}`")
    else:
        st.warning("No index found. Run `python ingest.py` first.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Export")
    if st.button("💾 Download Chat as JSON"):
        if st.session_state.get("chat_history"):
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "⬇️ Download",
                data=chat_json,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.info("No chat history yet.")

# ── Main area ─────────────────────────────────────────────────────────────────
if not DATA_FILE.exists():
    st.error("❌ No index found. Please run `python ingest.py` to build the document index first.")
    st.stop()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<div class="main-header">🧠 RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents — powered by retrieval-augmented generation</div>', unsafe_allow_html=True)

# Status badge
if force_local:
    st.markdown('<span class="badge badge-orange">⚡ Local Model</span>', unsafe_allow_html=True)
elif _openai_available:
    st.markdown('<span class="badge badge-green">✅ OpenAI Connected</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge badge-red">⚠️ OpenAI Unavailable</span>', unsafe_allow_html=True)

st.markdown("")

# Tabs
tab_chat, tab_docs, tab_history = st.tabs(["💬 Chat", "📚 Document Browser", "📜 History"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    # Render chat history
    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-label-user">You</div><div class="chat-user">{turn["question"]}</div>', unsafe_allow_html=True)
        model_badge = f'<span class="badge">{turn.get("model_used","")}</span>'
        st.markdown(f'<div class="chat-label-bot">🧠 Assistant {model_badge}</div><div class="chat-bot">{turn["answer"]}</div>', unsafe_allow_html=True)

        with st.expander(f"📎 Sources ({len(turn['hits'])} chunks)", expanded=False):
            for h in turn["hits"]:
                score_color = "badge-green" if h["score"] > 60 else "badge-orange" if h["score"] > 30 else "badge-red"
                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="src-title">📄 {h["source"]}</div>'
                    f'<span class="badge {score_color}">Relevance: {h["score"]}%</span> '
                    f'<span class="badge">Chunk #{h["chunk_id"]}</span>'
                    f'<p style="margin-top:8px;line-height:1.6">{h["text"][:400]}{"…" if len(h["text"])>400 else ""}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.markdown("")

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. What are the main causes of World War I?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send ➤")

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    suggestions = [
        "Summarize the key points of the documents",
        "What are the main topics covered?",
        "Who are the key people or organizations mentioned?",
    ]
    cols = st.columns(len(suggestions))
    for i, (col, sug) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(sug, key=f"sug_{i}"):
                question = sug
                submitted = True

    if submitted and question and question.strip():
        with st.spinner("🔍 Searching documents…"):
            t0   = time.time()
            hits = retrieve(question.strip(), top_k=top_k)
            retrieval_ms = int((time.time() - t0) * 1000)
            context = build_context(hits)

        with st.spinner("🤔 Generating answer…"):
            t1 = time.time()
            answer, model_used = generate_answer(
                question.strip(), context,
                st.session_state.chat_history,
                temperature=temperature,
                max_tokens=max_tokens,
                force_local=force_local,
            )
            gen_ms = int((time.time() - t1) * 1000)

        st.session_state.chat_history.append({
            "question":   question.strip(),
            "answer":     answer,
            "hits":       hits,
            "model_used": model_used,
            "retrieval_ms": retrieval_ms,
            "gen_ms":     gen_ms,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        st.rerun()

# ── Tab 2: Document Browser ───────────────────────────────────────────────────
with tab_docs:
    st.markdown("### 📚 Document Browser")
    bundle = load_index()
    sources = list({m["source"] for m in bundle["meta"]})

    selected_doc = st.selectbox("Select a document", sources)
    search_term  = st.text_input("🔎 Filter chunks by keyword", placeholder="Type to filter…")

    doc_chunks = [
        (i, bundle["chunks"][i], bundle["meta"][i])
        for i in range(len(bundle["chunks"]))
        if bundle["meta"][i]["source"] == selected_doc
    ]

    if search_term:
        doc_chunks = [(i, c, m) for i, c, m in doc_chunks if search_term.lower() in c.lower()]

    st.markdown(f"**{len(doc_chunks)} chunk(s)** found")

    for idx, chunk, meta in doc_chunks:
        with st.expander(f"Chunk #{meta['chunk_id']} — {len(chunk.split())} words"):
            highlighted = chunk
            if search_term:
                highlighted = highlighted.replace(
                    search_term,
                    f"**{search_term}**"
                )
            st.markdown(highlighted)

# ── Tab 3: History ────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 📜 Chat History")
    if not st.session_state.chat_history:
        st.info("No history yet. Start chatting in the Chat tab!")
    else:
        for i, turn in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"[{turn['timestamp']}]  {turn['question'][:80]}…"):
                st.markdown(f"**Q:** {turn['question']}")
                st.markdown(f"**A:** {turn['answer']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", turn.get("model_used", "—").split("·")[-1].strip())
                col2.metric("Retrieval", f"{turn.get('retrieval_ms', '?')} ms")
                col3.metric("Generation", f"{turn.get('gen_ms', '?')} ms")