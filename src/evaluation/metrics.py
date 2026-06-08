"""LLM-as-judge metrics for the research assistant.

Two RAGAS-style metrics plus a correctness check:

- faithfulness      : are the ANSWER's claims supported by the CONTEXT the
                      agent actually retrieved? (grounding / no hallucination)
- answer_relevancy  : does the ANSWER directly address the QUESTION?
- answer_correctness: does the ANSWER agree with the reference GROUND TRUTH?

Each judge call asks the model for strict JSON {"score": 0-1, "reason": "..."}.
Scores are parsed defensively so a malformed reply degrades to a usable float
instead of crashing a 20-question run.
"""

import json
import re


def parse_score(text: str) -> float:
    """Pull a 0-1 float out of a judge reply. Robust to JSON or plain text."""
    if text is None:
        return 0.0
    text = str(text).strip()

    # Preferred path: the judge returned JSON with a "score" field.
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            if "score" in obj:
                return _clamp(float(obj["score"]))
    except (ValueError, TypeError):
        pass

    # Fallback: first number that looks like a 0-1 score.
    m = re.search(r"(?<![\d.])(0?\.\d+|0|1(?:\.0+)?)(?![\d.])", text)
    if m:
        try:
            return _clamp(float(m.group(1)))
        except ValueError:
            pass
    return 0.0


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _judge(llm, system: str, user: str) -> float:
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return parse_score(resp.content)
    except Exception:
        return 0.0


_FAITHFULNESS_SYS = (
    "You are a strict evaluator. Given CONTEXT and an ANSWER, judge how well every "
    "factual claim in the ANSWER is supported by the CONTEXT. "
    "1.0 = every claim is grounded in the context; 0.0 = the answer contradicts or "
    "invents facts not in the context. "
    'Reply ONLY with JSON: {"score": <0-1 float>, "reason": "<one sentence>"}.'
)

_RELEVANCY_SYS = (
    "You are a strict evaluator. Given a QUESTION and an ANSWER, judge how directly "
    "the ANSWER addresses the QUESTION. "
    "1.0 = fully answers what was asked; 0.0 = off-topic or evasive. "
    'Reply ONLY with JSON: {"score": <0-1 float>, "reason": "<one sentence>"}.'
)

_CORRECTNESS_SYS = (
    "You are a strict evaluator. Given a REFERENCE answer (ground truth) and a "
    "CANDIDATE answer, judge how factually consistent the CANDIDATE is with the "
    "REFERENCE. Ignore wording and length; judge the facts. "
    "1.0 = fully consistent; 0.0 = contradicts the reference. "
    'Reply ONLY with JSON: {"score": <0-1 float>, "reason": "<one sentence>"}.'
)


def faithfulness(llm, context: str, answer: str) -> float:
    if not context or not context.strip():
        return None  # no retrieved context -> metric undefined for this item
    return _judge(llm, _FAITHFULNESS_SYS, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")


def answer_relevancy(llm, question: str, answer: str) -> float:
    return _judge(llm, _RELEVANCY_SYS, f"QUESTION:\n{question}\n\nANSWER:\n{answer}")


def answer_correctness(llm, ground_truth: str, answer: str) -> float:
    return _judge(llm, _CORRECTNESS_SYS, f"REFERENCE:\n{ground_truth}\n\nCANDIDATE:\n{answer}")
