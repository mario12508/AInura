#!/usr/bin/env python3
"""Aggregate explicit human reviews; missing/duplicate cases are errors, not successes.

python3 evaluation/score_reviews.py --reviews evaluation/reviews.jsonl --output evaluation/quality.json
Each row: case_id, mode, seed, reviewer, constraints, consistency, executable,
source_grounding (bool or null for baseline), comment. This script does not judge recipes.
"""
import argparse
import json
from pathlib import Path

FIELDS = ('constraints', 'consistency', 'executable', 'source_grounding')

def aggregate(rows):
    seen, groups = set(), {}
    if not rows:
        raise ValueError('No reviews')
    for row in rows:
        key = tuple(row[k] for k in ('case_id', 'mode', 'seed', 'reviewer'))
        if key in seen:
            raise ValueError(f'Duplicate review: {key}')
        seen.add(key)
        if row['mode'] not in ('baseline', 'rag', 'agent'):
            raise ValueError('Unknown mode')
        for field in FIELDS:
            if type(row[field]) is not bool and not (field == 'source_grounding' and row[field] is None):
                raise ValueError(f'{field}: expected boolean (source_grounding may be null)')
        groups.setdefault(row['mode'], []).append(row)
    result = {}
    for mode, group in groups.items():
        metrics = {}
        for field in FIELDS:
            values = [row[field] for row in group if row[field] is not None]
            metrics[field] = dict(passed=sum(values), total=len(values),
                                  rate=sum(values)/len(values) if values else None)
        result[mode] = dict(review_count=len(group), metrics=metrics)
    return {'meaning':'Rates of supplied human judgments; not model critic verdicts',
            'coverage':'Only supplied rows; compare identical cases/seeds/reviewers', 'modes':result}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reviews', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.reviews).read_text(encoding="utf-8").splitlines() if line.strip()]
    Path(args.output).write_text(json.dumps(aggregate(rows), ensure_ascii=False, indent=2)+'\n', encoding="utf-8")

if __name__ == '__main__':
    main()
