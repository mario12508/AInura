#!/usr/bin/env python3
"""Score saved retrieval runs against manually labeled relevant chunk ids.

Labels JSONL: {"query": "...", "relevant_ids": ["id1", "id2"]}
python evaluate_retrieval.py --labels labels.jsonl --runs 'runs/search_*.json' --index recipe_index --output metrics.json

Reports binary relevance Recall@k, MRR@k, nDCG@k for dense and reranked results.
This measures search relevance, not recipe taste, nutrition or constraint compliance.
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

from retrieval import digest, load_chunks, save_json


def metrics(ids, relevant, k):
    if k < 1 or not relevant:
        raise ValueError("Metrics require positive k and nonempty relevance labels")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document ids would inflate retrieval metrics")
    head = ids[:k]
    hits = [i for i, item in enumerate(head, 1) if item in relevant]
    dcg = sum(1 / math.log2(i + 1) for i in hits)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {f"recall@{k}": len(hits) / len(relevant),
            f"mrr@{k}": 1 / hits[0] if hits else 0.0,
            f"ndcg@{k}": dcg / ideal if ideal else 0.0}


def evaluate(labels_path, run_patterns, index_dir, ks):
    if any(k < 1 for k in ks):
        raise ValueError("k must be positive")
    corpus_ids = {r["id"] for r in load_chunks(Path(index_dir) / "chunks.jsonl")}
    manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    expected_hash = manifest["hashes"]["chunks.jsonl"]
    if digest(Path(index_dir) / "chunks.jsonl") != expected_hash:
        raise ValueError("The indexed corpus was changed; rebuild the index")
    paths = sorted({p for pattern in run_patterns for p in glob.glob(pattern)})
    runs = {}
    for path in paths:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if "dense_candidates" not in value or "results" not in value:
            continue
        query = value["query"]
        if query in runs:
            raise ValueError(f"Multiple search runs for one query: {query}. Select one run per query.")
        if value["index"]["chunks_sha256"] != expected_hash:
            raise ValueError(f"Run {path} uses a different corpus")
        runs[query] = value
    rows, skipped = [], []
    seen = set()
    for number, line in enumerate(Path(labels_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        query, relevant = value["query"], value["relevant_ids"]
        if query in seen:
            raise ValueError(f"Duplicate query in labels, line {number}")
        seen.add(query)
        if not isinstance(relevant, list) or not all(isinstance(x, str) for x in relevant):
            raise ValueError(f"Invalid relevant_ids, line {number}")
        if not relevant:
            skipped.append({"query": query, "reason": "not_labeled"})
            continue
        relevant = set(relevant)
        if not relevant.issubset(corpus_ids):
            raise ValueError(f"Unknown relevant chunk ids, line {number}: {sorted(relevant - corpus_ids)}")
        if query not in runs:
            raise ValueError(f"No saved retrieval run for labeled query: {query}")
        run = runs[query]
        entry = {"query": query, "relevant_ids": sorted(relevant), "dense": {}, "reranked": None}
        for k in ks:
            if len(run["dense_candidates"]) < min(k, len(corpus_ids)):
                raise ValueError(f"Dense run has too few results to evaluate k={k}: {query}")
            entry["dense"].update(metrics([r["id"] for r in run["dense_candidates"]], relevant, k))
        if run["settings"]["rerank"]:
            entry["reranked"] = {}
            for k in ks:
                if len(run["results"]) < min(k, len(corpus_ids)):
                    raise ValueError(f"Save --top-k >= {k} to evaluate reranked results: {query}")
                entry["reranked"].update(metrics([r["id"] for r in run["results"]], relevant, k))
        rows.append(entry)
    if not rows:
        raise ValueError("No labeled queries. Fill relevant_ids with real chunk ids first.")
    averages = {}
    for mode in ("dense", "reranked"):
        values = [r[mode] for r in rows if r[mode] is not None]
        averages[mode] = {"queries": len(values), "metrics": {
            key: sum(v[key] for v in values) / len(values) for key in values[0]
        } if values else {}}
    return {"queries_evaluated": len(rows), "skipped": skipped, "averages": averages,
            "per_query": rows, "relevance": "binary, manual labels; incomplete labels can bias metrics"}


def cli(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels',required=True)
    ap.add_argument('--runs',nargs='+',required=True,help='Paths/globs; exactly one saved search run per query')
    ap.add_argument('--index',default='recipe_index')
    ap.add_argument('--k',nargs='+',type=int,default=[1,5,10])
    ap.add_argument('--output',default='metrics.json')
    args=ap.parse_args(argv)
    try:
        result=evaluate(args.labels,args.runs,args.index,args.k)
        save_json(args.output,result)
        print(json.dumps(result['averages'],ensure_ascii=False,indent=2))
        print(f"Saved: {Path(args.output).resolve()}")
        return 0
    except (ValueError,OSError,KeyError) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(cli())
