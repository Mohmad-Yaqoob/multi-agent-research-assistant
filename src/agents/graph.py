import os
import json
import sqlite3
import tempfile
import time
from typing import Annotated, TypedDict, Dict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.router import get_model
from src.memory.summariser import build_messages
from src.tools.code_executor import code_executor_tool

load_dotenv()

# Embeddings (shared)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ---------------------------------------------------------------------------
# Per-chat document stores, PERSISTED TO DISK.
#
# Each chat (thread_id) has its own FAISS index saved under faiss_stores/<id>/,
# plus a small JSON of which files it contains. This mirrors how chat history
# is persisted in memory.db (SQLite) — so after an app restart, an old chat
# still has BOTH its messages AND its indexed documents, instead of the docs
# silently vanishing and every question falling through to web search.
# ---------------------------------------------------------------------------
STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "faiss_stores",
)
os.makedirs(STORE_DIR, exist_ok=True)

_THREAD_STORES: Dict[str, any] = {}   # thread_id -> FAISS store (in-memory cache)
_THREAD_DOCS: Dict[str, list] = {}    # thread_id -> [{filename, pages, chunks}, ...]

def _store_path(thread_id: str) -> str:
    return os.path.join(STORE_DIR, thread_id)

def _meta_path(thread_id: str) -> str:
    return os.path.join(STORE_DIR, f"{thread_id}__docs.json")

def _persist(thread_id: str) -> bool:
    # eval threads are transient — keep them in memory only, no disk clutter
    return not thread_id.startswith("eval-")

def _load_store(thread_id: str):
    path = _store_path(thread_id)
    if os.path.isdir(path):
        try:
            return FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[store load error] {e}")
    return None

def _load_meta(thread_id: str) -> list:
    p = _meta_path(thread_id)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            _THREAD_DOCS[thread_id] = data
            return list(data)
        except Exception:
            pass
    return []

# Web search tool
search_tool = DuckDuckGoSearchRun()

# PDF ingestion — adds into THIS chat's own store. Multiple PDFs per chat
# accumulate rather than overwriting each other, and the index is saved to disk.
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name

    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source_file"] = filename

    store = _THREAD_STORES.get(thread_id) or _load_store(thread_id)
    if store is None:
        store = FAISS.from_documents(chunks, embeddings)
    else:
        store.add_documents(chunks)
    _THREAD_STORES[thread_id] = store

    meta = {"filename": filename, "pages": len(docs), "chunks": len(chunks)}
    docs_meta = _THREAD_DOCS.get(thread_id) or _load_meta(thread_id)
    docs_meta.append(meta)
    _THREAD_DOCS[thread_id] = docs_meta

    if _persist(thread_id):
        try:
            store.save_local(_store_path(thread_id))
            with open(_meta_path(thread_id), "w", encoding="utf-8") as f:
                json.dump(docs_meta, f)
        except Exception as e:
            print(f"[store save error] {e}")

    os.remove(path)
    return meta


def list_thread_ids():
    """All chat thread_ids that have saved state — from the SQLite checkpointer
    (chats with messages) plus faiss_stores/ (chats with documents). Used to
    rebuild the sidebar chat list after an app restart."""
    ids = set()
    try:
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        cur = conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
        ids.update(r[0] for r in cur.fetchall())
        conn.close()
    except Exception as e:
        print(f"[list threads error] {e}")
    if os.path.isdir(STORE_DIR):
        for name in os.listdir(STORE_DIR):
            if os.path.isdir(os.path.join(STORE_DIR, name)):
                ids.add(name)
    return [t for t in ids if not t.startswith("eval-")]
def get_thread_docs(thread_id: str):
    """Documents indexed in THIS chat only (memory or disk)."""
    if thread_id in _THREAD_DOCS:
        return list(_THREAD_DOCS[thread_id])
    return _load_meta(thread_id)

def _get_store(thread_id: str):
    """This chat's FAISS store, loading from disk if not already in memory."""
    if thread_id in _THREAD_STORES:
        return _THREAD_STORES[thread_id]
    store = _load_store(thread_id)
    if store is not None:
        _THREAD_STORES[thread_id] = store
    return store

# State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    model_used: str
    latency_ms: float
    context: str

SYSTEM_PROMPT = "You are a helpful research assistant. Be concise and accurate."

# When document/web context is available, force STRICTLY grounded answers so the
# model reports what the source says instead of leaning on its own prior knowledge.
GROUNDED_INSTRUCTION = """Answer using ONLY the information in the Context below.
Do not use outside or prior knowledge. Do not add facts, numbers, names, or claims
that are not explicitly stated in the Context.
If the Context does not contain the answer, reply exactly:
"The provided document does not contain this information."
Cite the source when you use it."""

# Agent node
def agent_node(state: State, config=None) -> dict:
    messages = state["messages"]
    thread_id = config["configurable"]["thread_id"] if config else "default"

    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    query = last_human.content if last_human else ""
    model_name, qtype = get_model(query)

    # Step 1 — RAG from THIS chat's own FAISS store.
    # If a document is indexed in this chat, ALWAYS use it (top-k chunks).
    # No relevance threshold: FAISS L2 distances don't transfer across papers,
    # so a fixed cutoff wrongly rejected valid chunks for some documents and
    # leaked the query to web search. Web search fires ONLY when this chat has
    # no document indexed at all (store is None) — that's the "general/basic
    # question" path.
    context_block = ""
    source = ""
    store = _get_store(thread_id)
    if store:
        try:
            docs = store.similarity_search(query, k=6)
            if docs:
                context_block = "\n\n".join(d.page_content for d in docs)
                sources = sorted({d.metadata.get("source_file", "document") for d in docs})
                source = ", ".join(sources)
        except Exception as e:
            print(f"[retriever error] {e}")

    # Step 2 — web search fallback ONLY when this chat has no document
    global search_tool
    if not context_block and search_tool is not None:
        try:
            web_result = search_tool.run(query)
            if web_result:
                context_block = web_result
                source = "web search"
        except Exception as e:
            print(f"[web search error] {e}")

    # Step 3 — calculator for calc queries
    calc_result = ""
    if qtype == "calc":
        try:
            result = code_executor_tool.invoke({"code": f"print({query})"})
            if result.get("success"):
                calc_result = f"\nCalculation: {result['output']}"
        except Exception:
            pass

    # Build final prompt — strict grounding when context is present
    if context_block:
        system_text = (
            SYSTEM_PROMPT
            + "\n\n" + GROUNDED_INSTRUCTION
            + f"\n\nSource: {source}\nContext:\n{context_block}\n"
        )
    else:
        system_text = SYSTEM_PROMPT + "\nNo document context is available; answer from your own general knowledge."
    if calc_result:
        system_text += calc_result

    system = SystemMessage(content=system_text)

    history = build_messages(messages[:-1], "")
    history = [m for m in history if not isinstance(m, SystemMessage)]
    final_messages = [system] + history + [HumanMessage(content=query)]

    llm = ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=1024,
    )

    t0 = time.perf_counter()
    response = llm.invoke(final_messages)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "messages": [response],
        "query_type": qtype,
        "model_used": model_name,
        "latency_ms": round(latency_ms, 1),
        "context": context_block,
    }

# Graph
def build_graph():
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile(checkpointer=checkpointer)

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph