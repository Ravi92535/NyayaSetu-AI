"""
NyayaSetu AI - Streamlit Application
Interactive legal research assistant with Weaviate hybrid search (BNS, BNSS, BSA).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import settings
from src.generation import RAGPipeline, get_llm_model
from src.generation.chain import _is_valid_api_key
from src.ingestion import DocumentChunker, MultiFormatLoader
from src.retrieval import VectorStoreManager
from src.utils import logger


# Page Configuration
st.set_page_config(
    page_title="NyayaSetu AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStoreManager:
    """Singleton VectorStoreManager connected to Weaviate."""
    return VectorStoreManager()


@st.cache_resource(show_spinner=False)
def ensure_corpus_preloaded(_vs: VectorStoreManager) -> int:
    """
    Ensures BNS, BNSS, and BSA statutes are embedded into Weaviate once.
    Reuses existing persistent index on future runs.
    """
    count = _vs.count()
    if count > 0:
        logger.info(f"Using pre-embedded persistent statutory corpus ({count} sections ready).")
        return count

    logger.info("Statutory corpus empty. Pre-embedding BNS, BNSS, and BSA into Weaviate...")
    chunker = DocumentChunker()
    processed_dir = settings.BASE_DIR / settings.DATA_PROCESSED_DIR
    all_statute_docs = []
    for fname in ["bns.txt", "bnss.txt", "bsa.txt"]:
        file_path = processed_dir / fname
        if file_path.exists():
            docs = MultiFormatLoader.load_file(file_path)
            sections = chunker.split_statute_sections(docs)
            all_statute_docs.extend(sections)
            logger.info(f"Loaded {len(sections)} sections from {fname}")

    if all_statute_docs:
        _vs.add_documents(all_statute_docs)
        logger.info(f"Successfully embedded and persisted {_vs.count()} statutory sections.")
    return _vs.count()


@st.cache_resource(show_spinner=False)
def get_pipeline(_vs: VectorStoreManager) -> RAGPipeline:
    """Singleton RAGPipeline instance."""
    return RAGPipeline(vector_store=_vs)


# Initialize components
vs = get_vector_store()
total_sections = ensure_corpus_preloaded(vs)
pipeline = get_pipeline(vs)

# Initialize Session State
DEFAULT_MODELS = {
    "groq": "qwen/qwen3.8-27b",
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4o-mini",
}

if "provider_keys" not in st.session_state:
    st.session_state.provider_keys = {
        "groq": (os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY) if _is_valid_api_key(os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY) else "",
        "gemini": (os.environ.get("GEMINI_API_KEY") or settings.GEMINI_API_KEY) if _is_valid_api_key(os.environ.get("GEMINI_API_KEY") or settings.GEMINI_API_KEY) else "",
        "openai": (os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY) if _is_valid_api_key(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY) else "",
    }

if "active_provider" not in st.session_state:
    st.session_state.active_provider = settings.LLM_PROVIDER if settings.LLM_PROVIDER in DEFAULT_MODELS else "groq"

if "active_model" not in st.session_state:
    st.session_state.active_model = settings.LLM_MODEL or DEFAULT_MODELS[st.session_state.active_provider]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to NyayaSetu AI!**\n\n"
                "• **BNS:-** Bharatiya Nyaya Sanhita\n"
                "• **BNSS:-** Bharatiya Nagarik Suraksha Sanhita\n"
                "• **BSA:-** Bharatiya Sakshya Adhiniyam"
            ),
            "sources": [],
            "latency": None,
        }
    ]


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ NyayaSetu AI")
    st.caption("Weaviate Hybrid Search (BM25 + 384d Dense Vectors)")

    # 1. Statutory Corpus Section
    with st.container():
        st.subheader("📚 Indexed Statutory Corpus")
        st.markdown(
            """
            * 📖 **Bharatiya Nyaya Sanhita (BNS)** — `358 secs`
            * 📖 **Nagarik Suraksha (BNSS)** — `532 secs`
            * 📖 **Bharatiya Sakshya (BSA)** — `170 secs`
            """
        )
        st.success(f"✔ {total_sections:,} legal sections embedded & active", icon="✅")

    st.divider()

    # 2. LLM Configuration Section
    st.subheader("🔑 LLM Settings")

    provider_options = {
        "groq": "Groq (Qwen 27B)",
        "gemini": "Google Gemini (Gemini 1.5 Flash)",
        "openai": "OpenAI (GPT-4o mini)",
    }

    selected_prov = st.selectbox(
        "Provider",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(st.session_state.active_provider),
    )

    # Model-wise key handling: if not set for this provider, input is completely empty!
    current_key_for_prov = st.session_state.provider_keys.get(selected_prov, "")

    col_key, col_toggle = st.columns([5, 1])
    with col_toggle:
        show_raw = st.toggle("👁️", value=False, help="Show raw API key text")
    with col_key:
        api_key_input = st.text_input(
            "API Key",
            value=current_key_for_prov,
            type="default" if show_raw else "password",
            placeholder=f"Paste your {selected_prov.title()} API key...",
            help="Your API key stays local to this session.",
        )

    model_input = st.text_input(
        "Model Name",
        value=DEFAULT_MODELS.get(selected_prov, ""),
        placeholder=DEFAULT_MODELS.get(selected_prov, ""),
    )

    if st.button("Save & Connect", use_container_width=True, type="primary"):
        if not api_key_input.strip():
            st.warning("Please enter a valid API key.")
        else:
            with st.spinner("Connecting & validating key..."):
                target_model = model_input.strip() or DEFAULT_MODELS[selected_prov]
                env_map = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
                os.environ[env_map[selected_prov]] = api_key_input.strip()
                setattr(settings, env_map[selected_prov], api_key_input.strip())
                settings.LLM_PROVIDER = selected_prov
                settings.LLM_MODEL = target_model

                new_llm = get_llm_model(provider=selected_prov, model_name=target_model)

                # Test live validation ping
                try:
                    if hasattr(new_llm, "invoke"):
                        new_llm.invoke("Hi")
                    st.session_state.provider_keys[selected_prov] = api_key_input.strip()
                    st.session_state.active_provider = selected_prov
                    st.session_state.active_model = target_model
                    pipeline.llm = new_llm
                    st.success(f"Connected to {selected_prov.upper()} ({target_model})!")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as ping_err:
                    err_str = str(ping_err).lower()
                    if selected_prov == "groq" and ("not exist" in err_str or "not have access" in err_str or "404" in err_str):
                        fallback_model = "qwen/qwen3.8-27b"
                        settings.LLM_MODEL = fallback_model
                        fallback_llm = get_llm_model(provider=selected_prov, model_name=fallback_model)
                        try:
                            fallback_llm.invoke("Hi")
                            st.session_state.provider_keys[selected_prov] = api_key_input.strip()
                            st.session_state.active_provider = selected_prov
                            st.session_state.active_model = fallback_model
                            pipeline.llm = fallback_llm
                            st.success(f"Connected to Groq ({fallback_model})!")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e2:
                            st.error(f"Validation failed: {e2}")
                    else:
                        st.error(f"Validation failed: {ping_err}")

    # Active status badge
    if st.session_state.provider_keys.get(st.session_state.active_provider):
        st.info(f"🟢 **Active:** {st.session_state.active_provider.upper()} (`{st.session_state.active_model}`)")
    else:
        st.warning("⚠️ **No API key set.** Please add a key above to use the assistant.")

    st.divider()

    # Footer system info
    st.caption("• **Vector Store:** Weaviate Embedded (Local HNSW)")
    st.caption("• **Dense Vectors:** `all-MiniLM-L6-v2` (384 dims)")
    st.caption("• **Retrieval:** Hybrid BM25 + Dense RRF (α=0.25)")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()


# -----------------------------------------------------------------------------
# MAIN CHAT AREA
# -----------------------------------------------------------------------------
st.title("NyayaSetu AI • Legal RAG Engine")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📑 Relevant Statutory Citations ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    fname = s.get("filename", "")
                    act = "BNSS" if "bnss" in fname else ("BSA" if "bsa" in fname else ("BNS" if "bns" in fname else s.get("act_code", "").upper()))
                    sec_no = s.get("section_no")
                    title = s.get("title")
                    header = f"**{act} • Section {sec_no}**" + (f" *({title})*" if title else "")
                    st.markdown(header)
                    st.markdown(f"> *{s.get('snippet', '')}*")
        if msg.get("latency"):
            st.caption(f"⏱️ Retrieval & reasoning time: {msg['latency']}s")

# User Input
if prompt := st.chat_input("Ask a question about BNS, BNSS, or BSA..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching statutes with Weaviate hybrid & reasoning..."):
            res = pipeline.query(prompt, top_k=5)
            answer_text = res.get("answer", "")
            sources = res.get("sources", [])
            latency = res.get("latency_seconds", 0.0)

            st.markdown(answer_text)

            if sources:
                with st.expander(f"📑 Relevant Statutory Citations ({len(sources)})"):
                    for s in sources:
                        fname = s.get("filename", "")
                        act = "BNSS" if "bnss" in fname else ("BSA" if "bsa" in fname else ("BNS" if "bns" in fname else s.get("act_code", "").upper()))
                        sec_no = s.get("section_no")
                        title = s.get("title")
                        header = f"**{act} • Section {sec_no}**" + (f" *({title})*" if title else "")
                        st.markdown(header)
                        st.markdown(f"> *{s.get('snippet', '')}*")

            st.caption(f"⏱️ Retrieval & reasoning time: {latency}s")

    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
        "latency": latency,
    })


if __name__ == "__main__":
    from streamlit.runtime import exists

    if not exists():
        # Launched directly via `python app/main.py`
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())

