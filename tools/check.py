"""Run available offline checks without loading models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'version_2/ainura_local'
AGENTS = {
    'coordinator': ('coordinator.py', 'test_coordinator_agent.py'),
    'requirements': ('requirements.py', 'test_requirements_agent.py'),
    'recipe': ('recipe_agent.py', 'test_recipe_agent.py'),
    'critic': ('critic.py', 'test_critic_agent.py'),
    'retrieval': ('retrieval_agent.py', 'test_retrieval_agent.py'),
}


def unittest_suite(folder, pattern='test*.py'):
    return [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', folder, '-p', pattern, '-v']


def main():
    suites = {
        'rag': unittest_suite('version_1/ainura_local'),
        'runtime': unittest_suite('version_2/ainura_local', 'test_local.py'),
        'server': unittest_suite('version_1/webui'),
        'ui': ['node', 'version_1/webui/test_ui.cjs'],
    }
    for name, (module, test) in AGENTS.items():
        if (LOCAL / 'agents' / module).exists():
            if not (LOCAL / test).exists():
                print(f'{name}: missing {test}', file=sys.stderr)
                return 1
            suites[name] = unittest_suite('version_2/ainura_local', test)
    if all((LOCAL / 'agents' / module).exists() for module, _ in AGENTS.values()):
        for test in ('test_agents.py', 'test_five_agent_pipeline.py'):
            if (LOCAL / test).exists():
                suites[test.removesuffix('.py')] = unittest_suite('version_2/ainura_local', test)
    if any((ROOT / 'evaluation').glob('test*.py')):
        suites['evaluation'] = unittest_suite('evaluation')
    results = {}
    for name, command in suites.items():
        started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, encoding='utf-8',
                                    capture_output=True, timeout=120,
                                    env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            results[name] = {'returncode': result.returncode, 'output': result.stdout + result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            results[name] = {'returncode': 1, 'error': str(error)}
        results[name]['seconds'] = time.perf_counter() - started
        print(name, 'PASS' if results[name]['returncode'] == 0 else 'FAIL', flush=True)
        if results[name]['returncode']:
            print(results[name].get('output', results[name].get('error', '')))
    missing = [name for name, (module, _) in AGENTS.items() if not (LOCAL / 'agents' / module).exists()]
    if missing:
        print('Agents pending:', ', '.join(missing))
    destination = ROOT / 'artifacts/checks.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int(any(row['returncode'] for row in results.values()))


if __name__ == '__main__':
    sys.exit(main())
