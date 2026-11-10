#!/usr/bin/env python3
"""Run baseline/RAG/agent with matched searches. Requires local models and GPU.

Example: python3 evaluation/run_benchmark.py --limit 3 --seeds 42
Default runs all cases, one seed. Outputs are unique to each invocation.
Quality requires separate reviews, not acceptance by the internal critic.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from process_control import launch_options, stop_tree
V1 = ROOT / 'version_1/ainura_local'
V2 = ROOT / 'version_2/ainura_local'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, default=ROOT/'evaluation/cases.jsonl')
    ap.add_argument('--index', type=Path, default=V1/'recipe_index')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--timeout', type=int, default=1200)
    args = ap.parse_args()
    if args.timeout < 1 or (args.limit is not None and args.limit < 1):
        ap.error('timeout and limit must be positive')
    cases = [json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in cases]
    if len(set(ids)) != len(ids) or any(not i.replace('_','').isalnum() for i in ids):
        ap.error('case ids must be unique and alphanumeric')
    cases = cases[:args.limit] if args.limit else cases
    folder = ROOT/'evaluation'/('benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6])
    folder.mkdir(parents=True)
    (folder/'cases.jsonl').write_text(''.join(json.dumps(case,ensure_ascii=False)+'\n' for case in cases), encoding='utf-8')
    (folder/'settings.json').write_text(json.dumps({
        'index':str(args.index.resolve()), 'seeds':args.seeds, 'timeout':args.timeout,
        'python':sys.version, 'initial_search_cache':'exact query match only',
        'quality':'requires independent review; exit status measures execution only'
    },ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    rows = []

    def execute(command, log):
        started = time.perf_counter()
        with log.open('w', encoding='utf-8') as out:
            try:
                proc = subprocess.Popen([sys.executable,'-B',*map(str, command)], cwd=ROOT,
                                        stdout=out, stderr=subprocess.STDOUT, **launch_options())
                proc.wait(timeout=args.timeout)
                return dict(returncode=proc.returncode, seconds=time.perf_counter()-started)
            except subprocess.TimeoutExpired:
                stop_tree(proc)
                proc.wait()
                return dict(returncode=124, seconds=time.perf_counter()-started, error='timeout')

    for case in cases:
        case_dir = folder/case['id']; case_dir.mkdir()
        search = case_dir/'search.json'
        try:
            cached = json.loads((V1/'search.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            cached = {}
        if isinstance(cached, dict) and cached.get('query') == case['query']:
            search.write_text(json.dumps(cached, ensure_ascii=False), encoding='utf-8')
            search_status = dict(returncode=0, cached=True)
        else:
            search_status = execute([V1/'retrieval.py','search','--index',args.index,
                                      '--query',case['query'],'--output',search,'--local-only'],
                                     case_dir/'search.log')
        for seed in args.seeds:
            for mode in ('baseline','rag','agent'):
                output = case_dir/f'{mode}_{seed}.json'
                command = ([V2/'agent_recipe.py','--search',search,'--index',args.index]
                           if mode == 'agent' else [V1/'answer_recipe.py','--mode',mode])
                if mode == 'baseline':
                    command += ['--query',case['query']]
                elif mode == 'rag':
                    command += ['--search',search]
                if mode != 'baseline' and search_status['returncode']:
                    status = dict(returncode=1,error='retrieval failed')
                else:
                    status = execute(command+['--seed',seed,'--output',output,'--local-only'],
                                     output.with_suffix('.log'))
                rows.append(dict(case_id=case['id'],query=case['query'],seed=seed,mode=mode,
                                 output=str(output.relative_to(ROOT)), search=search_status, **status))
                (folder/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
                print(case['id'],seed,mode,status,flush=True)
    print('Artifacts:',folder)
    return int(any(row['returncode'] for row in rows))

if __name__ == '__main__':
    sys.exit(main())
