"""
app.py - Minimal, Modern Streamlit Web UI for Universal PDF RAG Chatbot
Provides conversational document QA with LangGraph memory and page-level source citations.
Includes live streaming token generation with instant Stop button responsiveness.
"""

import os
import sys
import warnings

# Suppress HuggingFace / Transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*")

# Silence Chroma / PostHog telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "1"

try:
    import chromadb.telemetry.product.posthog as _chroma_posthog
    _chroma_posthog.Posthog.capture = lambda self, event: None
except Exception:
    pass

import hashlib
import html
import uuid
from pathlib import Path
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

try:
    from streamlit.runtime.scriptrunner import StopException
except ImportError:
    try:
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException
    except ImportError:
        class StopException(BaseException):
            pass

import rag
import chat_store

# -------------------------------------------------------------
# STREAMING CALLBACK HANDLER (ENABLES INSTANT STOP BUTTON)
# -------------------------------------------------------------
class StreamlitStreamingCallbackHandler(BaseCallbackHandler):
    """
    Streams tokens in real time to an st.empty placeholder.
    By executing st.markdown on every token, this serves as an active yield
    point that allows Streamlit's top-right 'Stop' button to interrupt immediately.
    """
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
        self.is_answering = False

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # Only stream tokens from the answer generator, not internal query rewriter
        msg_content = ""
        if messages and messages[0]:
            msg_content = str(
                messages[0][-1].content if hasattr(messages[0][-1], "content") else messages[0]
            )
        if "helpful assistant answering questions" in msg_content:
            self.is_answering = True
            self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        if self.is_answering:
            self.text += token
            # Active Streamlit yield point: immediately catches StopException when Stop button is pressed
            self.placeholder.markdown(self.text + "▌")


# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# MINIMAL & THEME-ADAPTIVE CSS (WORKS BEAUTIFULLY IN DARK & LIGHT)
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Clean font & title styling */
    .app-header {
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        opacity: 0.7;
        margin-bottom: 1.2rem;
    }

    /* Minimal glass welcome card */
    .welcome-box {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 24px;
        background: rgba(128, 128, 128, 0.04);
        margin: 1rem 0 1.5rem 0;
    }
    .welcome-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .welcome-desc {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .steps-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .step-card {
        flex: 1;
        min-width: 170px;
        padding: 12px 14px;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        font-size: 0.85rem;
    }
    .step-card strong {
        display: block;
        margin-bottom: 4px;
        color: #818CF8;
    }

    /* Document meta card in sidebar */
    .doc-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px;
        background: rgba(128, 128, 128, 0.05);
        margin-top: 10px;
    }
    .doc-filename {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 8px;
        word-break: break-all;
    }
    .meta-pills {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 6px;
    }
    .meta-pill {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        background: rgba(129, 140, 248, 0.15);
        color: #818CF8;
        font-weight: 500;
    }
    .history-heading {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.65;
        margin: 0.25rem 0 0.5rem 0;
    }
    .history-file {
        font-size: 0.75rem;
        opacity: 0.6;
        overflow-wrap: anywhere;
        margin: -0.5rem 0 0.55rem 0.65rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
chat_store.initialize()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex}"

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if st.session_state.active_chat_id is None and st.session_state.doc_info:
    active_info = st.session_state.doc_info
    active_chat = chat_store.create_chat(active_info["filename"], active_info["hash"])
    st.session_state.active_chat_id = active_chat["id"]
    st.session_state.thread_id = active_chat["id"]
    for saved_message in st.session_state.messages:
        chat_store.add_message(
            active_chat["id"],
            saved_message["role"],
            saved_message["content"],
            saved_message.get("sources", []),
        )


def load_chat(chat_id):
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        st.error("This chat is no longer available.")
        return
    pdf_path = UPLOAD_DIR / f"{chat['pdf_hash']}.pdf"
    if not pdf_path.is_file():
        st.error(
            f"The PDF for this chat is missing: {chat['pdf_name']}. "
            "Restore the PDF to the data/uploads folder to reopen this conversation."
        )
        return
    try:
        with st.spinner(f"Opening chat for {chat['pdf_name']}..."):
            pipeline_info = rag.setup_rag_pipeline(
                pdf_path=pdf_path,
                persist_dir="chroma_db",
            )
        st.session_state.active_chat_id = chat["id"]
        st.session_state.thread_id = chat["id"]
        st.session_state.messages = chat_store.get_messages(chat["id"])
        st.session_state.rag_graph = pipeline_info["graph"]
        st.session_state.doc_info = {
            "filename": chat["pdf_name"],
            "pages": pipeline_info["num_pages"],
            "chunks": pipeline_info["num_chunks"],
            "stored_vectors": pipeline_info["stored_vectors"],
            "hash": pipeline_info["pdf_hash"],
            "cached": pipeline_info["is_cached"],
        }
    except (FileNotFoundError, ValueError, ConnectionError, RuntimeError) as error:
        st.error(f"Could not reopen this chat: {error}")


# -------------------------------------------------------------
# SIDEBAR - DOCUMENT MANAGEMENT & SETTINGS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Document Setup")
    st.caption("Upload a PDF to chat with full conversational context.")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload any PDF (textbook, paper, report, manual)."
    )

    col_btn, _ = st.columns([1, 0.01])
    with col_btn:
        process_btn = st.button("⚙️ Process PDF", use_container_width=True)

    if process_btn:
        if uploaded_file is None:
            st.warning("Please choose a PDF file first.")
        else:
            pdf_bytes = uploaded_file.getvalue()
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]
            saved_pdf_path = UPLOAD_DIR / f"{pdf_hash}.pdf"
            try:
                saved_pdf_path.write_bytes(pdf_bytes)

                with st.spinner("Indexing PDF..."):
                    pipeline_info = rag.setup_rag_pipeline(
                        pdf_path=saved_pdf_path,
                        persist_dir="chroma_db"
                    )

                    st.session_state.rag_graph = pipeline_info["graph"]
                    st.session_state.doc_info = {
                        "filename": uploaded_file.name,
                        "pages": pipeline_info["num_pages"],
                        "chunks": pipeline_info["num_chunks"],
                        "stored_vectors": pipeline_info["stored_vectors"],
                        "hash": pipeline_info["pdf_hash"],
                        "cached": pipeline_info["is_cached"]
                    }

                    chat = chat_store.create_chat(
                        pdf_name=uploaded_file.name,
                        pdf_hash=pipeline_info["pdf_hash"],
                    )
                    st.session_state.active_chat_id = chat["id"]
                    st.session_state.thread_id = chat["id"]
                    st.session_state.messages = []

                if pipeline_info["is_cached"]:
                    st.success("Reused existing index.")
                else:
                    st.success("Document indexed successfully.")

            except FileNotFoundError as e:
                st.error(f"File error: {e}")
            except ValueError as e:
                st.error(f"Invalid PDF: {e}")
            except ConnectionError as e:
                st.error(f"Ollama connection error: {e}")
            except RuntimeError as e:
                st.error(f"Processing error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # Active Document Information
    if st.session_state.doc_info:
        st.divider()
        info = st.session_state.doc_info
        cache_label = "⚡ Reused Cache" if info.get("cached") else "✨ Freshly Indexed"
        pages_label = f"{info['pages']} Pages" if info.get("pages") else "Indexed"

        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-filename">📑 {html.escape(info['filename'])}</div>
            <div class="meta-pills">
                <span class="meta-pill">{pages_label}</span>
                <span class="meta-pill">{info['stored_vectors']} Chunks</span>
                <span class="meta-pill">{cache_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat controls
    st.divider()
    st.markdown('<div class="history-heading">Your chats</div>', unsafe_allow_html=True)
    can_start_chat = st.session_state.doc_info is not None
    if st.button(
        "＋  New chat",
        use_container_width=True,
        disabled=not can_start_chat,
        type="primary",
    ):
        info = st.session_state.doc_info
        chat = chat_store.create_chat(info["filename"], info["hash"])
        st.session_state.active_chat_id = chat["id"]
        st.session_state.thread_id = chat["id"]
        st.session_state.messages = []

    chats = chat_store.list_chats()
    if chats:
        with st.expander(f"Recent conversations · {len(chats)}", expanded=True):
            for chat in chats:
                is_active = chat["id"] == st.session_state.active_chat_id
                label = chat["title"] or chat["pdf_name"]
                if st.button(
                    f"{'● ' if is_active else ''}{label}",
                    key=f"open_chat_{chat['id']}",
                    use_container_width=True,
                    type="secondary",
                ):
                    load_chat(chat["id"])
                    st.rerun()
                st.markdown(
                    f'<div class="history-file">📄 {html.escape(chat["pdf_name"])}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("Your saved conversations will appear here.")

    # System Status Expander
    with st.expander("ℹ️ System Status", expanded=False):
        is_ready, ollama_msg = rag.check_ollama_status()
        if is_ready:
            st.success(ollama_msg)
        else:
            st.warning(ollama_msg)
        st.caption(f"**Embeddings:** `{rag.EMBEDDING_MODEL}`")
        st.caption(f"**LLM:** `{rag.OLLAMA_MODEL}`")
        st.caption(f"**Context window:** `{rag.OLLAMA_NUM_CTX}` tokens")


# -------------------------------------------------------------
# MAIN CHAT AREA
# -------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">Universal PDF RAG Chatbot</div>
    <div class="app-subtitle">Private local QA powered by LangChain, LangGraph, ChromaDB & Ollama</div>
</div>
""", unsafe_allow_html=True)

# Minimal Welcome Hero (shown when no PDF is processed)
if st.session_state.rag_graph is None:
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-title">Get Started in 3 Steps</div>
        <div class="welcome-desc">Upload any document to ask detailed questions with verified page citations and multi-turn conversational memory.</div>
        <div class="steps-row">
            <div class="step-card">
                <strong>1. Upload PDF</strong>
                Select any document in the sidebar.
            </div>
            <div class="step-card">
                <strong>2. Process</strong>
                Click <em>Process PDF</em> to create or load the index.
            </div>
            <div class="step-card">
                <strong>3. Ask Anything</strong>
                Ask questions and follow-ups naturally.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} cited pages)", expanded=False):
                for fname, page in sources:
                    st.markdown(f"• **{fname}** — `Page {page}`")

# Chat Input & Response Handling
user_question = st.chat_input("Ask a question about the document...")

if user_question:
    if st.session_state.rag_graph is None:
        st.info("Please upload and process a PDF in the sidebar first.")
    else:
        # Check for natural exit/quit inputs
        if user_question.strip().lower() in {"exit", "quit"}:
            st.session_state.messages.append({"role": "user", "content": user_question})
            chat_store.add_message(
                st.session_state.active_chat_id,
                "user",
                user_question,
            )
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                exit_reply = "Chat session ended. Start a new chat or select an earlier conversation from the sidebar."
                st.markdown(exit_reply)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": exit_reply,
                    "sources": []
                })
                chat_store.add_message(
                    st.session_state.active_chat_id,
                    "assistant",
                    exit_reply,
                )
        else:
            # Display user message
            prior_messages = list(st.session_state.messages)
            st.session_state.messages.append({"role": "user", "content": user_question})
            chat_store.add_message(
                st.session_state.active_chat_id,
                "user",
                user_question,
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            # Generate Assistant Response with Live Token Streaming
            with st.chat_message("assistant"):
                placeholder = st.empty()
                stream_handler = StreamlitStreamingCallbackHandler(placeholder)

                with st.spinner("Searching document context..."):
                    try:
                        result = rag.ask_question(
                            graph=st.session_state.rag_graph,
                            question=user_question,
                            thread_id=st.session_state.thread_id,
                            callbacks=[stream_handler],
                            history=[
                                {"role": message["role"], "content": message["content"]}
                                for message in prior_messages
                                if message["role"] in {"user", "assistant"}
                            ],
                        )

                        answer = result.get("answer", stream_handler.text or "No answer generated.")
                        sources = [
                            (st.session_state.doc_info["filename"], page)
                            for _, page in result.get("sources", [])
                        ]

                        # Final clean render (removes the blinking cursor ▌)
                        placeholder.markdown(answer)

                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)} cited pages)", expanded=False):
                                for fname, page in sources:
                                    st.markdown(f"• **{fname}** — `Page {page}`")

                        # Record to chat history
                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                        st.session_state.messages.append(assistant_message)
                        chat_store.add_message(
                            st.session_state.active_chat_id,
                            "assistant",
                            answer,
                            sources,
                        )

                    except StopException:
                        # User clicked the top-right 'Stop' button in Streamlit
                        partial = stream_handler.text.strip()
                        msg = (partial + "\n\n*(Generation stopped by user)*") if partial else "*(Generation stopped by user)*"
                        placeholder.markdown(msg)
                        assistant_message = {
                            "role": "assistant",
                            "content": msg,
                            "sources": []
                        }
                        st.session_state.messages.append(assistant_message)
                        chat_store.add_message(
                            st.session_state.active_chat_id,
                            "assistant",
                            msg,
                        )
                        st.stop()
                    except ConnectionError as e:
                        st.error(f"LLM Connection Error: {e}")
                        error_message = f"I couldn't answer this question because Ollama is unavailable: {e}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        })
                        chat_store.add_message(
                            st.session_state.active_chat_id,
                            "assistant",
                            error_message,
                        )
                    except Exception as e:
                        st.error(f"Error answering question: {e}")
                        error_message = f"I couldn't answer this question: {e}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        })
                        chat_store.add_message(
                            st.session_state.active_chat_id,
                            "assistant",
                            error_message,
                        )
