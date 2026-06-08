import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

@tool
def web_search_tool(query: str) -> dict:
    """Search the web for current information not in the documents. Use this when retriever finds nothing relevant."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=3)
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title"),
                    "content": r.get("content"),
                    "url": r.get("url"),
                }
                for r in response.get("results", [])
            ],
        }
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}