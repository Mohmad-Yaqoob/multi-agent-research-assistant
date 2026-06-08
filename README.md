---
title: Multi Agent Research Assistant
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Multi-Agent Research Assistant

A RAG chatbot built with LangGraph and FAISS, with query routing, per-chat
document isolation, conversation summarisation, and a reproducible LLM-as-judge
evaluation pipeline.

## Evaluation Results (20-question benchmark)

Regenerate any time with `python scripts/evaluate.py --sample 20`. The table
below reflects the latest run — update it whenever you re-run the benchmark.

| Metric            | Score | Target    |
|-------------------|-------|-----------|
| Faithfulness      | 0.99  | > 0.85    |
| Answer Relevancy  | 0.82  | > 0.80    |
| P90 Latency       | 2.6s  | < 2s      |

## Architecture

- A **query router** classifies each query as simple / complex / calc.
- The **LangGraph agent** grounds answers with a per-chat **FAISS** retriever,
  falls back to **web search** when documents have no answer, and uses a
  sandboxed **code executor** for calculations.
- **Per-chat FAISS store** — each conversation has isolated document context.
- **Conversation memory** — history is summarised after 10 turns to save tokens.

## Stack

- **LLM**: Groq API — Llama 3.1 8B
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector store**: FAISS (per chat)
- **Agent framework**: LangGraph (SQLite checkpointer)
- **Web search**: DuckDuckGo
- **Calculator**: sandboxed Python code executor
- **Evaluation**: custom LLM-as-judge pipeline
- **Frontend**: Streamlit

## Features

- Upload multiple PDFs per chat with isolated document context
- Query routing (simple / complex / calc)
- Conversation summarisation after 10 turns
- Web-search fallback when documents don't cover the question
- Reproducible evaluation pipeline: faithfulness, answer relevancy, correctness

## Setup

    git clone https://github.com/Mohmad-Yaqoob/multi-agent-research-assistant
    cd multi-agent-research-assistant
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env      # then add your GROQ_API_KEY

Run the app:

    streamlit run app.py

## Evaluation

Runs every question in `tests/qa_pairs.json` through the real agent, then scores
the answers with an LLM-as-judge (faithfulness, answer relevancy, correctness)
and records latency. Writes `evaluation_report.json`.

    python scripts/evaluate.py --sample 20

The printed numbers are whatever the agent actually scores on that run — put
those in the table above rather than hand-editing the report.

## Note on HuggingFace deployment

PDF upload works when running locally. The live HuggingFace Space demonstrates
question-answering and web search without document upload, due to free-tier file
handling restrictions.
