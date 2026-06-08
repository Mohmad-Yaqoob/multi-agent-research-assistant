import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=10,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Classify the query into exactly one word: simple, complex, or calc.
     simple = factual, definition, single fact
     complex = analysis, comparison, multi-step reasoning
     calc = math, numbers, calculation

     Reply with only one word."""),

    ("human", "{query}"),
])


chain = prompt | llm


def classify(query: str) -> str:
    try:
        result = chain.invoke({"query": query})
        label = result.content.strip().lower()
        if label in ("simple", "complex", "calc"):
            return label
        return "complex"
    except Exception:
        return "complex"


def get_model(query: str) -> tuple:
    qtype = classify(query)
    # 8B stays the workhorse for latency; qtype still drives the calc/tool path.
    return "llama-3.1-8b-instant", qtype
