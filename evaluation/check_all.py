#!/usr/bin/env python3
"""Run each suite in isolation and save reproducible checks (no model downloads)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    'reviews': [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'evaluation', '-v'],
    'v1': [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'version_1/ainura_local', '-v'],
    'v2': [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'version_2/ainura_local', '-v'],
    'server': [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'version_1/webui', '-v'],
    'ui': ['node', 'version_1/webui/test_ui.cjs'],
}

def main():
    results = {}
    for name, command in SUITES.items():
        started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, encoding='utf-8',
                                    capture_output=True, timeout=120,
                                    env={**os.environ, 'PYTHONUTF8': '1'})
            results[name] = dict(command=command, returncode=result.returncode,
                                 output=result.stdout + result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            results[name] = dict(command=command, returncode=1, error=str(error))
        results[name]['seconds'] = time.perf_counter() - started
        print(name, 'PASS' if results[name]['returncode'] == 0 else 'FAIL')
    destination = ROOT / 'evaluation' / 'checks.json'
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int(any(row['returncode'] for row in results.values()))

if __name__ == '__main__':
    sys.exit(main())
