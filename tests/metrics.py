"""Project metrics: router accuracy, escalation, RAG hit rate, latency.

Run:     python -m tests.metrics
Output:  console report + metrics/results.json
"""
import json
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.router import classify
from src.config import MODELS
from src.graph import graph
from src.rag.retriever import search
from src.settings import settings

TESTS = Path(__file__).resolve().parent
OUT = TESTS.parent / "metrics"
OUT.mkdir(exist_ok=True)


def load(name: str) -> list[dict]:
    return json.loads((TESTS / name).read_text(encoding="utf-8"))


def with_retry(func: Callable[..., Any], *args: Any, attempts: int = 3, **kwargs: Any) -> Any:
    """Call func, backing off when the provider rate limit is hit.

    Args:
        func: The callable to invoke.
        *args: Positional arguments passed through to func.
        attempts: How many times to try before giving up.
        **kwargs: Keyword arguments passed through to func.

    Returns:
        Whatever func returns.

    Raises:
        RuntimeError: If the retry loop somehow exits without a result.
    """
    for attempt in range(attempts):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            exhausted = "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error)
            if not exhausted or attempt == attempts - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"with_retry exhausted {attempts} attempts without returning")

# ---------------------------------------------------------------------------
# 1. Router accuracy
# ---------------------------------------------------------------------------
def measure_router() -> dict:
    cases = load("router_testset.json")

    correct = 0
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(int)
    confidences = []
    failures = []

    for case in cases:
        decision = with_retry(classify, case["query"])
        expected, actual = case["expected"], decision.query_type
        ok = actual == expected

        correct += ok
        per_class[expected]["total"] += 1
        per_class[expected]["correct"] += ok
        confidences.append(decision.confidence)

        if not ok:
            confusion[f"{expected} -> {actual}"] += 1
            failures.append({
                "query": case["query"],
                "expected": expected,
                "actual": actual,
                "confidence": round(decision.confidence, 2),
                "reason": decision.reason,
            })

    total = len(cases)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_confidence": round(mean(confidences), 3),
        "per_class": {
            label: {**stats, "rate": round(stats["correct"] / stats["total"], 3)}
            for label, stats in sorted(per_class.items())
        },
        "confusion": dict(confusion),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# 2. Escalation on ambiguous queries
# ---------------------------------------------------------------------------
def measure_escalation() -> dict:
    """Ambiguous queries should fall below the confidence threshold."""
    cases = load("router_ambiguous.json")

    escalated = 0
    details = []
    for case in cases:
        decision = with_retry(classify, case["query"])
        low = decision.confidence < settings.confidence_threshold
        escalated += low
        details.append({
            "query": case["query"],
            "note": case.get("note", ""),
            "query_type": decision.query_type,
            "confidence": round(decision.confidence, 2),
            "escalated": low,
        })

    total = len(cases)
    return {
        "total": total,
        "escalated": escalated,
        "rate": round(escalated / total, 4),
        "threshold": settings.confidence_threshold,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 3. RAG retrieval hit rate
# ---------------------------------------------------------------------------
def measure_rag() -> dict:
    cases = load("rag_testset.json")

    hits3 = hits5 = 0
    ranks = []
    misses = []

    for case in cases:
        hits = search(case["question"], top_k=5)
        expected = case["expected_source"].lower()

        rank = None
        for position, hit in enumerate(hits, 1):
            if expected in f"{hit['url']} {hit['title']}".lower():
                rank = position
                break

        if rank:
            ranks.append(rank)
            hits5 += 1
            hits3 += rank <= 3
        else:
            misses.append({
                "question": case["question"],
                "expected_source": case["expected_source"],
                "got": [h["url"] for h in hits],
            })

    total = len(cases)
    return {
        "total": total,
        "hit_at_3": round(hits3 / total, 4),
        "hit_at_5": round(hits5 / total, 4),
        "avg_rank_when_found": round(mean(ranks), 2) if ranks else None,
        "misses": misses,
    }


# ---------------------------------------------------------------------------
# 4. Latency per agent
# ---------------------------------------------------------------------------
def measure_latency() -> dict:
    cases = load("latency_testset.json")
    timings = defaultdict(list)

    for case in cases:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        started = time.perf_counter()
        state = graph.invoke({"messages": [HumanMessage(content=case["query"])]}, config)
        elapsed = time.perf_counter() - started

        # interrupt() suspends the node before it writes handled_by,
        # so fall back to the router's classification.
        agent = state.get("handled_by") or f"{state.get('query_type')}_investigator"
        # bug_investigator stops at interrupt(): this is time to the first
        # clarifying question, not to the final verdict.
        if not state.get("final_answer"):
            agent += " (to first question)"
        timings[agent].append(elapsed)

    return {
        agent: {
            "runs": len(values),
            "mean_sec": round(mean(values), 2),
            "median_sec": round(median(values), 2),
            "min_sec": round(min(values), 2),
            "max_sec": round(max(values), 2),
        }
        for agent, values in sorted(timings.items())
    }


# ---------------------------------------------------------------------------
def run(label: str, func):
    print(f"\n=== {label} ===")
    try:
        return func()
    except Exception as error:
        print(f"  [!] skipped: {type(error).__name__}: {error}")
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> None:
    print(
        f"Provider: {settings.llm_provider} / "
        f"fast={MODELS[settings.llm_provider]['fast']}, "
        f"smart={MODELS[settings.llm_provider]['smart']}"
    )

    router = run("1/4  Router accuracy", measure_router)
    if "error" not in router:
        print(f"  Accuracy: {router['correct']}/{router['total']} = {router['accuracy']:.1%}")
        print(f"  Avg confidence: {router['avg_confidence']:.2f}")
        for label, stats in router["per_class"].items():
            print(f"    {label:8s} {stats['correct']}/{stats['total']}  {stats['rate']:.0%}")
        for pair, count in router["confusion"].items():
            print(f"    [x] {pair}: {count}")

    escalation = run("2/4  Escalation on ambiguous", measure_escalation)
    if "error" not in escalation:
        print(f"  Escalated: {escalation['escalated']}/{escalation['total']} "
              f"= {escalation['rate']:.1%} (threshold {escalation['threshold']})")
        for item in escalation["details"]:
            mark = "ok " if item["escalated"] else "[x]"
            print(f"    {mark} {item['confidence']:.2f}  {item['query'][:45]}")

    rag = run("3/4  RAG retrieval", measure_rag)
    if "error" not in rag:
        print(f"  Hit@3: {rag['hit_at_3']:.1%}   Hit@5: {rag['hit_at_5']:.1%}   (n={rag['total']})")
        if rag["avg_rank_when_found"]:
            print(f"  Avg rank when found: {rag['avg_rank_when_found']}")
        for miss in rag["misses"]:
            print(f"    [x] miss: {miss['question'][:55]}")

    if "error" not in rag:
        print("\n  ...waiting 65s for the Cohere rate limit window")
        time.sleep(65)

    latency = run("4/4  Latency", measure_latency)
    if "error" not in latency:
        for agent, stats in latency.items():
            print(f"  {agent:<34} mean {stats['mean_sec']:>6.2f}s  "
                  f"median {stats['median_sec']:>6.2f}s  (n={stats['runs']})")


    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "models": MODELS[settings.llm_provider],
        "router": router,
        "escalation": escalation,
        "rag": rag,
        "latency": latency,
    }
    path = OUT / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
