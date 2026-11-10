#!/usr/bin/env python3
"""Resource counters for baseline, RAG and agent logs. Not a quality evaluator.

python summarize_runs.py --files baseline_answer.json rag_answer.json agent_answer.json --output comparison.csv
"""
import argparse
import csv
import json
from pathlib import Path


def summarize(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = data.get("mode")
    if mode not in {"baseline", "rag", "agent"}:
        raise ValueError(f"{path}: not a baseline/RAG/agent run log")
    if mode == "agent":
        stats = data.get("stats", {})
        calls = stats.get("llm_calls", 0)
        prompt = stats.get("prompt_tokens", 0)
        generated = stats.get("generated_tokens", 0)
        retrieval = stats.get("retrieval_calls", 0)
        cached = data.get("initial_search") is not None
        accepted = data.get("accepted", False)
    else:
        response = data.get("ollama_response", {})
        calls = int("request" in data)
        prompt, generated = response.get("prompt_eval_count", 0), response.get("eval_count", 0)
        retrieval, cached, accepted = 0, mode == "rag", ""
    return {"file": str(path), "query": data.get("query", ""), "mode": mode,
            "model": data.get("model", ""), "status": data.get("status", ""),
            "llm_calls": calls, "prompt_tokens": prompt, "generated_tokens": generated,
            "new_retrieval_calls": retrieval, "cached_initial_search": cached,
            "llm_critic_accepted": accepted, "total_seconds": data.get("total_seconds", ""),
            "timing_scope": "excludes precomputed initial search; includes new searches"}


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--output", default="comparison.csv")
    args = ap.parse_args(argv)
    rows = [summarize(p) for p in args.files]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['mode']}: calls={r['llm_calls']}, output_tokens={r['generated_tokens']}, seconds={r['total_seconds']}")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    cli()
