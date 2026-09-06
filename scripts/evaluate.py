"""Evaluate the research assistant against tests/qa_pairs.json.

Runs each question through the REAL LangGraph agent (same code path the app
uses), then scores the answers with an LLM-as-judge on three metrics:
faithfulness, answer_relevancy, answer_correctness. Latency is measured from
the agent's own timing.

Usage:
    python scripts/evaluate.py --sample 20
    python scripts/evaluate.py --questions tests/qa_pairs.json --out evaluation_report.json

Requires GROQ_API_KEY in the environment (or a .env file).

Note: the numbers this prints are whatever your agent actually scores on the
run. Put THOSE numbers on your resume/README — do not hand-edit the report.
"""

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


def percentile(values, pct):
    """Linear-interpolated percentile (pct in 0-100). Returns 0.0 if empty."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def run_agent(graph, question, thread_id):
    """Run one question end-to-end; return (answer, context, latency_ms)."""
    from langchain_core.messages import HumanMessage, AIMessage

    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    state = graph.get_state(config).values

    answer = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content
            break

    return answer, state.get("context", ""), state.get("latency_ms", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="tests/qa_pairs.json")
    ap.add_argument("--out", default="evaluation_report.json")
    ap.add_argument("--sample", type=int, default=None,
                    help="evaluate only the first N questions")
    ap.add_argument("--judge-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between questions (avoids rate limits)")
    ap.add_argument("--doc", default=None,
                help="path to a PDF to index into the eval thread before running (tests real RAG, not just web fallback)")
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Add it to your environment or .env file.")

    from langchain_groq import ChatGroq
    from src.agents.graph import get_graph
    from src.evaluation.metrics import faithfulness, answer_relevancy, answer_correctness

    with open(args.questions, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    if args.sample:
        qa_pairs = qa_pairs[:args.sample]

    graph = get_graph()
    eval_thread_id = f"eval-{uuid.uuid4()}"
    if args.doc:
        from src.agents.graph import ingest_pdf
        with open(args.doc, "rb") as f:
            meta = ingest_pdf(f.read(), thread_id=eval_thread_id, filename=os.path.basename(args.doc))
        print(f"Indexed {meta['filename']} ({meta['pages']} pages, {meta['chunks']} chunks) into eval thread\n")
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    judge = ChatGoogleGenerativeAI(
        model=args.judge_model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )
    # judge = ChatGroq(model=args.judge_model, api_key=os.getenv("GROQ_API_KEY"),
    #                  temperature=0, max_tokens=200)

    faith, relev, correct, latencies = [], [], [], []

    for i, pair in enumerate(qa_pairs, 1):
        q = pair["question"]
        gt = pair.get("ground_truth", "")
        print(f"[{i}/{len(qa_pairs)}] {q[:60]}...", flush=True)

        try:
            answer, context, latency = run_agent(graph, q, eval_thread_id)
            print(f"    context[:150]: {context[:150]!r}")
        except Exception as e:
            print(f"    agent error: {e}")
            continue

        f_score = faithfulness(judge, context, answer)     # None if no context
        r_score = answer_relevancy(judge, q, answer)
        c_score = answer_correctness(judge, gt, answer) if gt else None

        if f_score is not None:
            faith.append(f_score)
        relev.append(r_score)
        if c_score is not None:
            correct.append(c_score)
        latencies.append(latency)
        print(f"    answer: {answer[:200]}")
        print(f"    faith={f_score} relev={r_score} correct={c_score} "
              f"lat={latency:.0f}ms")
        time.sleep(args.delay)

    report = {
        "num_questions": len(qa_pairs),
        "num_scored": len(relev),
        "metrics": {
            "faithfulness": mean(faith),
            "answer_relevancy": mean(relev),
            "answer_correctness": mean(correct),
        },
        "faithfulness_coverage": f"{len(faith)}/{len(qa_pairs)} had retrieved context",
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p90": round(percentile(latencies, 90), 1),
        },
        "targets_met": {
            "faithfulness_gt_085": mean(faith) > 0.85,
            "answer_relevancy_gt_080": mean(relev) > 0.80,
            "p90_lt_2000ms": percentile(latencies, 90) < 2000,
        },
        "judge_model": args.judge_model,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 44)
    print(f"  Faithfulness      {report['metrics']['faithfulness']}")
    print(f"  Answer Relevancy  {report['metrics']['answer_relevancy']}")
    print(f"  Answer Correctness{report['metrics']['answer_correctness']:>6}")
    print(f"  P50 latency       {report['latency_ms']['p50']} ms")
    print(f"  P90 latency       {report['latency_ms']['p90']} ms")
    print("=" * 44)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
