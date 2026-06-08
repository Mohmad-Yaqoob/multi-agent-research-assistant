import uuid
import os
import sys
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "threads" not in st.session_state:
    st.session_state.threads = [st.session_state.thread_id]
if "titles" not in st.session_state:
    st.session_state.titles = {}
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}


@st.cache_resource(show_spinner="Loading agent...")
def load_agent():
    from src.agents.graph import get_graph
    return get_graph()


graph = load_agent()

# --- Restore chats from disk on startup (survives app restarts) ---
# Rebuilds the sidebar list, titles, and message history from the persisted
# SQLite checkpointer + faiss_stores, so closing and reopening the app brings
# every past chat (and its indexed documents) back.
if "restored" not in st.session_state:
    from src.agents.graph import list_thread_ids
    for tid in list_thread_ids():
        try:
            state = graph.get_state({"configurable": {"thread_id": tid}})
            msgs = state.values.get("messages", [])
            first_human = next((m for m in msgs if isinstance(m, HumanMessage)), None)
            title = first_human.content[:28] if first_human else f"Chat {tid[:6]}"
        except Exception:
            title = f"Chat {tid[:6]}"
        if tid not in st.session_state.threads:
            st.session_state.threads.append(tid)
        st.session_state.titles[tid] = title
    st.session_state.restored = True


def load_chat(tid):
    """Switch to a chat and pull its message history back from the checkpointer."""
    st.session_state.thread_id = tid
    try:
        state = graph.get_state({"configurable": {"thread_id": tid}})
        st.session_state.messages = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in state.values.get("messages", [])
            if isinstance(m, (HumanMessage, AIMessage)) and m.content
        ]
    except Exception:
        st.session_state.messages = []


with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("LangGraph · RAG · Multi-Agent")
    st.divider()

    st.subheader("💬 Chats")
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.thread_id = new_id
        st.session_state.messages = []
        st.session_state.threads.insert(0, new_id)
        st.rerun()

    for tid in st.session_state.threads[:20]:
        title = st.session_state.titles.get(tid, f"Chat {tid[:6]}")
        active = "▶ " if tid == st.session_state.thread_id else "   "
        if st.button(f"{active}{title}", key=f"t_{tid}", use_container_width=True):
            load_chat(tid)
            st.rerun()

st.title("🔬 Research Assistant")

# --- Per-chat document upload lives in the chat panel itself, so it's ---
# --- always obvious which PDFs belong to THIS chat. New chat = clean slate.
from src.agents.graph import ingest_pdf, get_thread_docs

current_docs = get_thread_docs(st.session_state.thread_id)
label = f"📚 Documents in this chat ({len(current_docs)} indexed)" if current_docs else "📚 Upload documents for this chat"

with st.expander(label, expanded=not current_docs):
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.thread_id}",  # resets per chat
    )
    if st.button("📥 Index Documents", type="primary", key=f"index_{st.session_state.thread_id}"):
        if uploaded:
            bar = st.progress(0)
            for i, pdf in enumerate(uploaded):
                ingest_pdf(pdf.getvalue(), thread_id=st.session_state.thread_id, filename=pdf.name)
                bar.progress((i + 1) / len(uploaded))
            st.success(f"✅ Indexed {len(uploaded)} document(s) for this chat")
            st.rerun()
        else:
            st.warning("Please upload PDFs first.")

    if current_docs:
        for d in current_docs:
            st.caption(f"  📄 {d['filename']} — {d['pages']} pages, {d['chunks']} chunks")

if st.session_state.last_meta:
    m = st.session_state.last_meta
    model = m.get("model_used", "")
    label = "⚡ Fast (8B)" if "8b" in model else "🧠 Smart (70B)"
    qtype = m.get("query_type", "?")
    st.caption(f"{label} · {qtype} · {m.get('latency_ms', 0):.0f}ms")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything about your documents...")

if user_input:
    if st.session_state.thread_id not in st.session_state.titles:
        st.session_state.titles[st.session_state.thread_id] = user_input[:28]
    if st.session_state.thread_id not in st.session_state.threads:
        st.session_state.threads.insert(0, st.session_state.thread_id)

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        tool_box = [None]

        def stream():
            for chunk, _ in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage) and tool_box[0] is None:
                    tool_box[0] = st.status("🔧 Using tools...", expanded=True)
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        answer = st.write_stream(stream())

        if tool_box[0]:
            tool_box[0].update(label="✅ Done", state="complete", expanded=False)

    try:
        state = graph.get_state(config)
        st.session_state.last_meta = {
            "query_type": state.values.get("query_type"),
            "model_used": state.values.get("model_used"),
            "latency_ms": state.values.get("latency_ms"),
        }
    except Exception:
        pass

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()