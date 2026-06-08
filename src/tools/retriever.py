import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool

CHROMA_DIR = "chroma_store"

_store = None

def _get_store():
    global _store
    if _store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        _store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="research_docs",
        )
    return _store

@tool
def retriever_tool(query: str) -> dict:
    """Search the indexed documents for information relevant to the query. Always try this first before web search."""
    try:
        store = _get_store()
        docs = store.similarity_search(query, k = 5)
        if not docs:
            return {"found": False, "message": "Nothing found in documents."}
        return {
            "found": True,
            "results": [
                {
                    "content" : d.page_content,
                    "source": d.metadata.get("source_file", "unknown"),
                }
                for d in docs
            ],
        }
    except Exception as e:
        return {"found": False, "error": str(e)}
    

def get_retriever():
    return _get_store().as_retriever(search_kwargs={"k": 5})