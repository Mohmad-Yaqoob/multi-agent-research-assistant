---
title: Agentic RAG Assistant
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Agentic RAG Assistant

A retrieval-augmented document QA system built with LangGraph and FAISS. Upload
research papers per chat, ask questions grounded strictly in the documents, with
a web-search fallback for general queries, per-chat document isolation, disk
persistence across restarts, and a reproducible LLM-as-judge evaluation pipeline.

## Evaluation Results

Frontier-judged (Gemini) over a 14-question grounded benchmark on the ReAct
paper. Reproduce with `python scripts/evaluate.py --doc <pdf> --sample 14`.

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 0.97  |
| Answer Relevancy  | 0.93  |
| Answer Correctness| 0.93  |
| Median inference  | ~0.8s (Groq) |

The benchmark includes questions the paper cannot answer; the system correctly
declines to answer those rather than hallucinating, which is reflected in the
faithfulness score.

## Architecture

- **Query router** classifies each query (simple / complex / calc) via a
  LangGraph agent.
- **Grounded retrieval** — answers strictly from the chat's own FAISS index when
  a document is present; the model is instructed to use only retrieved context
  and to say so when the answer isn't in the document.
- **Web-search fallback** — DuckDuckGo is used only when the chat has no document
  indexed (general/basic questions).
- **Sandboxed code executor** for calculation queries.
- **Per-chat isolation** — each chat has its own document set; uploads in one
  chat never leak into another.
- **Disk persistence** — FAISS indexes and chat history are saved to disk, so
  documents and conversations survive app restarts.

## Stack

- **LLM**: Groq API — GPT-OSS 20B
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector store**: FAISS (per chat, persisted)
- **Agent framework**: LangGraph (SQLite checkpointer)
- **Web search**: DuckDuckGo
- **Evaluation judge**: Gemini (LLM-as-judge)
- **Frontend**: Streamlit
- **Packaging**: Docker

## Setup

    git clone https://github.com/Mohmad-Yaqoob/multi-agent-research-assistant
    cd multi-agent-research-assistant
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env      # add GROQ_API_KEY (and GEMINI_API_KEY for evaluation)

Run the app:

    streamlit run app.py

## Evaluation

Runs each question in `tests/qa_pairs.json` through the real agent, then scores
faithfulness, answer relevancy, and correctness with an LLM-as-judge (Gemini),
and records latency. Writes `evaluation_report.json`. Results are cached
per-question so a run can resume if interrupted.

    python scripts/evaluate.py --doc tests/sample_pdfs/2210.03629v3.pdf --sample 14

The printed numbers are whatever the current system scores — put those in the
table above rather than hand-editing the report.

## Docker

    docker build -t research-assistant .
    docker run -p 7860:7860 --env-file .env research-assistant

Then open http://localhost:7860.