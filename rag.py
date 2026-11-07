"""
rag.py - Core RAG Application Logic for Universal PDF RAG Chatbot
Preserves exact chunking, embedding, vector storage, retrieval,
and LangGraph conversational memory workflow from the original notebook.
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

import re
import gc
import json
import time
import shutil
import hashlib
import urllib.request
from pathlib import Path
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# CONFIGURATION & CONSTANTS (PRESERVED FROM NOTEBOOK)
# ============================================================
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_NUM_CTX = 4096
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8
FETCH_K = 24
MMR_LAMBDA = 0.5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_THREAD_ID = "pdf-chat-session"

# Table of Contents line detection pattern
TOC_LINE = re.compile(r"(\s{4,}|(?:\.\s?){4,})\d{1,4}\s*$")

# Number word dictionary for chapter detection
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20
}
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


# ============================================================
# PROMPTS (PRESERVED FROM NOTEBOOK)
# ============================================================
rewrite_prompt = ChatPromptTemplate.from_template("""
You are rewriting a user's follow-up question for document retrieval.

Conversation history:
{history}

Latest user question:
{question}

Rewrite the latest question as one standalone search query.
- Preserve the user's intent.
- Resolve pronouns such as he, she, it, they, this, and that using the conversation.
- Do not answer the question.
- If the question is already standalone, return it unchanged.
- Return only the rewritten question.
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about the provided PDF.

Use ONLY the supplied PDF context to answer. Do not invent facts that are not supported by the context.

If the answer is not present in the retrieved context, your ENTIRE reply must be exactly:
"This information is not covered in the provided PDF."
In that case, do not add anything else -- no partial answer, no outside/general knowledge, no apology.

Give a COMPLETE and SPECIFIC answer, not just a related or partial one:
- Cover every part of the question, not just the part that is easiest to answer.
- Include concrete details from the context (definitions, examples, names, numbers) rather than a vague summary.
- Do not stop at a general statement when the context has a more precise answer available.

If the question asks you to compare, contrast, or state a difference between two or more things
(for example "difference between X and Y", "X vs Y", "how does X differ from Y"), you MUST
explicitly address EACH thing being compared, not just one side -- state what each one is before
comparing them, so neither side is left out.

Answer naturally and directly. For explanation questions, explain in simple language when appropriate.

PDF context:
{context}

Question:
{question}

Answer:
""")


# ============================================================
# HELPER FUNCTIONS (PRESERVED FROM NOTEBOOK)
# ============================================================
def roman_to_int(value: str) -> int:
    """Convert a roman numeral string to an integer."""
    total, previous = 0, 0
    for char in reversed(value.upper()):
        current = ROMAN_VALUES.get(char, 0)
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def detect_chapter(text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Detect basic chapter/section headings near the beginning of a page.
    Generic detection: works on any PDF with standard chapter headings.
    """
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    lines = [x for x in lines if x]
    sample = lines[:12]
    for line in sample:
        m = re.match(r"^chapter\s*[:\-]?\s*(\d{1,3})\b", line, re.I)
        if m:
            return int(m.group(1)), line
        m = re.match(r"^chapter\s+([ivxlcdm]+)\b", line, re.I)
        if m:
            return roman_to_int(m.group(1)), line
        m = re.match(
            r"^chapter\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
            line,
            re.I
        )
        if m:
            return NUMBER_WORDS[m.group(1).lower()], line
    return None, None


def is_toc_page(text: str, threshold: float = 0.4) -> bool:
    """
    A page is considered table-of-contents if most of its lines end
    in a page number after a run of spaces or dot leaders.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    hits = sum(1 for l in lines if TOC_LINE.search(l))
    return hits / len(lines) >= threshold


def clean_response(response: Any) -> str:
    """Convert LangChain model content into clean printable text."""
    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(x for x in parts if x).strip()

    return str(content).strip()


def format_context(documents: List[Document]) -> str:
    """Format retrieved documents with location headers into prompt context."""
    blocks = []
    for doc in documents:
        page = doc.metadata.get("page", "?")
        chapter = doc.metadata.get("chapter")
        heading = doc.metadata.get("chapter_heading")
        location = f"Page {page}"
        if chapter is not None:
            location += f" | Chapter {chapter}"
        if heading:
            location += f" | {heading}"
        blocks.append(f"[{location}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def source_list(documents: List[Document], default_source: str = "document.pdf") -> List[Tuple[str, Any]]:
    """Extract deduplicated (filename, page) tuples from document metadata."""
    seen = set()
    sources = []
    for doc in documents:
        page = doc.metadata.get("page", "?")
        source_file = doc.metadata.get("source_file", default_source)
        key = (source_file, page)
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources


def compute_pdf_hash(pdf_path: str | Path) -> str:
    """Compute 12-char SHA-256 hash of PDF content to isolate ChromaDB collections."""
    path = Path(pdf_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ============================================================
# CORE RAG FUNCTIONS
# ============================================================
_cached_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_ollama_base_url() -> str:
    """Prefer the runtime-configured Ollama URL, otherwise fall back to localhost."""
    for env_key in ("OLLAMA_HOST", "OLLAMA_BASE_URL"):
        value = os.environ.get(env_key)
        if value:
            cleaned = value.strip().rstrip("/")
            if "//" not in cleaned:
                cleaned = f"http://{cleaned}"
            return cleaned
    return "http://localhost:11434"


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Load or return cached HuggingFace embeddings model."""
    global _cached_embeddings
    if _cached_embeddings is None:
        try:
            _cached_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e
    return _cached_embeddings


def get_llm(model_name: str = OLLAMA_MODEL, temperature: float = 0) -> ChatOllama:
    """Return ChatOllama with an explicit context window for retrieved text."""
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=OLLAMA_NUM_CTX,
        base_url=get_ollama_base_url(),
    )


def check_ollama_status(model_name: str = OLLAMA_MODEL) -> Tuple[bool, str]:
    """
    Check if the local Ollama instance is reachable and model is installed.
    Returns (is_ready, message).
    """
    base_url = get_ollama_base_url()
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                model_base = model_name.split(":")[0]
                has_model = any(
                    model_name == m or model_name in m or m.startswith(model_base)
                    for m in models
                )
                if has_model:
                    return True, f"Ollama is online with model '{model_name}' available."
                else:
                    return False, (
                        f"Ollama is running, but model '{model_name}' was not found. "
                        f"Run 'ollama pull {model_name}' in your terminal."
                    )
    except Exception:
        return False, (
            f"Ollama is not reachable at {base_url}. "
            "Please ensure Ollama is installed and running ('ollama serve')."
        )
    return True, "Ollama check passed."


def process_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Load a PDF document, clean text, discard TOC pages, assign 1-based page numbers,
    detect chapters, and store source metadata.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path.resolve()}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a .pdf file, received: '{path.name}'")

    try:
        loader = PyPDFLoader(str(path))
        raw_pages = loader.load()
    except Exception as e:
        raise ValueError(f"Invalid or unreadable PDF file '{path.name}': {e}") from e

    if not raw_pages:
        raise ValueError(f"The PDF file '{path.name}' is empty or contains no pages.")

    pdf_id = compute_pdf_hash(path)
    pages: List[Document] = []

    for doc in raw_pages:
        text = doc.page_content

        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        elif not isinstance(text, str):
            text = str(text)

        # Remove null characters and drop lone UTF-16 surrogates
        text = text.replace("\x00", " ")
        text = text.encode("utf-8", errors="ignore").decode("utf-8").strip()

        if not text:
            continue

        # Skip table of contents pages
        if is_toc_page(text):
            continue

        original_page = doc.metadata.get("page")
        if isinstance(original_page, int):
            page_number = original_page + 1
        else:
            page_number = len(pages) + 1

        doc.page_content = text
        doc.metadata["page"] = page_number
        doc.metadata["source_file"] = path.name
        doc.metadata["pdf_id"] = pdf_id

        # Chapter detection
        number, heading = detect_chapter(text)
        doc.metadata["chapter"] = number
        doc.metadata["chapter_heading"] = heading

        pages.append(doc)

    if not pages:
        raise ValueError(
            f"No readable text content found in '{path.name}'. "
            "The file may contain only scanned images or unextractable text."
        )

    return pages


def create_chunks(
    pages: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split loaded document pages into chunks and format metadata for Chroma.
    """
    if not pages:
        raise ValueError("Cannot create chunks: no pages provided.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = text_splitter.split_documents(pages)
    clean_chunks: List[Document] = []

    for i, doc in enumerate(raw_chunks):
        text = str(doc.page_content).replace("\x00", " ").encode("utf-8", errors="ignore").decode("utf-8").strip()
        if not text:
            continue

        # Make metadata Chroma-compatible (no None or nested types)
        metadata: Dict[str, Any] = {}
        for key, value in doc.metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        metadata["chunk_id"] = i
        clean_chunks.append(Document(page_content=text, metadata=metadata))

    if not clean_chunks:
        raise ValueError("No valid chunks were produced from the document pages.")

    return clean_chunks


def build_vectorstore(
    chunks: Optional[List[Document]] = None,
    pdf_path: Optional[str | Path] = None,
    persist_dir: str | Path = "chroma_db",
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    force_rebuild: bool = False
) -> Chroma:
    """
    Build or reuse a persistent ChromaDB vector store isolated by PDF content hash.
    If the database already exists for this exact PDF and has vectors, it is reused.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if pdf_path is not None:
        pdf_hash = compute_pdf_hash(pdf_path)
    elif chunks and "pdf_id" in chunks[0].metadata:
        pdf_hash = chunks[0].metadata["pdf_id"]
    else:
        raise ValueError("Either pdf_path or chunks with 'pdf_id' metadata must be provided.")

    db_dir = Path(persist_dir) / pdf_hash
    collection_name = f"pdf_{pdf_hash}"

    # Reuse existing vectorstore if available and not forcing a rebuild
    if db_dir.exists() and not force_rebuild:
        try:
            vs = Chroma(
                collection_name=collection_name,
                persist_directory=str(db_dir),
                embedding_function=embeddings
            )
            count = vs._collection.count()
            if count > 0:
                return vs
        except Exception:
            # If opening existing collection fails, fall through to rebuilding
            pass

    # Build fresh vectorstore: chunks are required
    if chunks is None:
        if pdf_path is not None:
            pages = process_pdf(pdf_path)
            chunks = create_chunks(pages)
        else:
            raise ValueError("Chunks are required to build a fresh ChromaDB vector store.")

    # Clean existing directory if needed
    if db_dir.exists():
        gc.collect()
        for _ in range(5):
            try:
                shutil.rmtree(db_dir)
                break
            except PermissionError:
                time.sleep(1)

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(db_dir)
        )
        stored = vectorstore._collection.count()
        if stored != len(chunks):
            raise RuntimeError(
                f"Ingestion check failed: only {stored} of {len(chunks)} chunks stored in ChromaDB."
            )
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"ChromaDB initialization failed: {e}") from e


def get_retriever(
    vectorstore: Chroma,
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K,
    lambda_mult: float = MMR_LAMBDA
):
    """Create an MMR retriever from the given Chroma vectorstore."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )


def safe_retrieve(
    retriever: Any,
    embeddings: HuggingFaceEmbeddings,
    vectorstore: Chroma,
    question: str,
    k: Optional[int] = None,
    fetch_k: int = FETCH_K,
    top_k: int = TOP_K
) -> List[Document]:
    """
    Perform safe retrieval: invokes MMR retriever, falling back to direct Chroma
    query if LangChain Document construction encounters malformed null fields.
    """
    try:
        return retriever.invoke(question)
    except Exception:
        query_vector = embeddings.embed_query(question)
        raw = vectorstore._collection.query(
            query_embeddings=[query_vector],
            n_results=k or fetch_k,
            include=["documents", "metadatas"]
        )
        docs = []
        if raw and "documents" in raw and raw["documents"]:
            for text, meta in zip(raw["documents"][0], raw["metadatas"][0]):
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata=meta or {}))
        return docs[: (k or top_k)]


# ============================================================
# LANGGRAPH STATE & WORKFLOW
# ============================================================
class RAGState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    conversation_history: List[Dict[str, str]]
    standalone_question: str
    retrieved_docs: list
    answer: str
    sources: list


def create_rag_graph(
    retriever: Any,
    embeddings: HuggingFaceEmbeddings,
    vectorstore: Chroma,
    llm: Optional[ChatOllama] = None,
    checkpointer: Optional[MemorySaver] = None
):
    """
    Build and compile the LangGraph RAG workflow with conversation memory.
    Nodes:
    1. rewrite_question: Resolves pronouns and conversation history.
    2. retrieve_documents: Retrieves relevant chunks using MMR / safe_retrieve.
    3. generate_answer: Prompts Ollama with strictly grounded context.
    """
    if llm is None:
        llm = get_llm()

    if checkpointer is None:
        checkpointer = MemorySaver()

    def rewrite_question_node(state: RAGState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        latest = messages[-1].content if messages else ""
        conversation_history = state.get("conversation_history")
        if conversation_history is None:
            history_text = "\n".join(
                f"{type(m).__name__}: {m.content}" for m in messages[:-1][-6:]
            )
        else:
            history_text = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in conversation_history[-6:]
            )

        if not history_text:
            return {"standalone_question": latest}

        prompt = rewrite_prompt.format_messages(
            history=history_text,
            question=latest
        )
        rewritten = clean_response(llm.invoke(prompt))
        return {"standalone_question": rewritten or latest}

    def retrieve_documents_node(state: RAGState) -> Dict[str, Any]:
        question = state["standalone_question"]
        docs = safe_retrieve(retriever, embeddings, vectorstore, question)
        return {
            "retrieved_docs": docs,
            "sources": source_list(docs)
        }

    def generate_answer_node(state: RAGState) -> Dict[str, Any]:
        question = state["standalone_question"]
        docs = state.get("retrieved_docs", [])
        context = format_context(docs)

        prompt = answer_prompt.format_messages(
            context=context,
            question=question
        )
        answer = clean_response(llm.invoke(prompt))
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }

    workflow = StateGraph(RAGState)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.add_edge(START, "rewrite_question")
    workflow.add_edge("rewrite_question", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(checkpointer=checkpointer)


def _is_ollama_connection_error(error: BaseException) -> bool:
    """Detect both Linux and Windows Ollama connection failures."""
    err = str(error).lower()
    markers = (
        "11434",
        "connection refused",
        "connection error",
        "failed to connect",
        "no connection could be made",
        "actively refused",
        "cannot connect",
        "connection reset by peer",
        "temporarily unavailable",
        "name or service not known",
        "connection aborted",
        "failed to establish a new connection",
        "could not connect to ollama",
    )
    return any(marker in err for marker in markers) or isinstance(error, ConnectionError)


def ask_question(
    graph: Any,
    question: str,
    thread_id: str = DEFAULT_THREAD_ID,
    callbacks: Optional[list] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Execute a question against the compiled LangGraph workflow with thread memory.
    Supports callbacks for live streaming and instant Stop button responsiveness.
    Returns state dict with keys: 'answer', 'sources', 'standalone_question', 'retrieved_docs'.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks

    try:
        state: Dict[str, Any] = {"messages": [HumanMessage(content=question)]}
        if history is not None:
            state["conversation_history"] = history
        return graph.invoke(
            state,
            config=config
        )
    except ConnectionError as e:
        raise ConnectionError(
            f"Could not connect to Ollama. Ensure Ollama is running and model '{OLLAMA_MODEL}' is pulled."
        ) from e
    except Exception as e:
        if _is_ollama_connection_error(e):
            raise ConnectionError(
                f"Could not connect to Ollama. Ensure Ollama is running and model '{OLLAMA_MODEL}' is pulled."
            ) from e
        raise RuntimeError(f"Error during RAG execution: {e}") from e


def setup_rag_pipeline(
    pdf_path: str | Path,
    persist_dir: str | Path = "chroma_db",
    llm: Optional[ChatOllama] = None,
    checkpointer: Optional[MemorySaver] = None
) -> Dict[str, Any]:
    """
    High-level pipeline setup:
    1. Checks if ChromaDB already has this PDF hashed & indexed.
    2. Parses & chunks PDF only if not yet indexed.
    3. Builds or attaches Chroma vectorstore.
    4. Compiles LangGraph RAG workflow with checkpointer memory.
    """
    embeddings = get_embeddings()
    pdf_hash = compute_pdf_hash(pdf_path)
    db_dir = Path(persist_dir) / pdf_hash

    is_cached = db_dir.exists() and any(db_dir.iterdir() if db_dir.is_dir() else [])
    num_pages = 0
    num_chunks = 0

    if is_cached:
        try:
            vectorstore = build_vectorstore(
                pdf_path=pdf_path,
                persist_dir=persist_dir,
                embeddings=embeddings,
                force_rebuild=False
            )
            num_chunks = vectorstore._collection.count()
        except Exception:
            pages = process_pdf(pdf_path)
            num_pages = len(pages)
            chunks = create_chunks(pages)
            num_chunks = len(chunks)
            vectorstore = build_vectorstore(
                chunks=chunks,
                pdf_path=pdf_path,
                persist_dir=persist_dir,
                embeddings=embeddings,
                force_rebuild=True
            )
            is_cached = False
    else:
        pages = process_pdf(pdf_path)
        num_pages = len(pages)
        chunks = create_chunks(pages)
        num_chunks = len(chunks)
        vectorstore = build_vectorstore(
            chunks=chunks,
            pdf_path=pdf_path,
            persist_dir=persist_dir,
            embeddings=embeddings
        )

    retriever = get_retriever(vectorstore)
    graph = create_rag_graph(
        retriever=retriever,
        embeddings=embeddings,
        vectorstore=vectorstore,
        llm=llm,
        checkpointer=checkpointer
    )

    return {
        "graph": graph,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "pdf_hash": pdf_hash,
        "is_cached": is_cached,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "stored_vectors": vectorstore._collection.count()
    }
