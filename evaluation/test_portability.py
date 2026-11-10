import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from process_control import launch_options, stop_tree

spec = importlib.util.spec_from_file_location('run_benchmark', ROOT / 'evaluation/run_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PortabilityTests(unittest.TestCase):
    def test_stops_launched_process(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], **launch_options())
        try:
            stop_tree(proc)
            self.assertIsNotNone(proc.wait(timeout=10))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_benchmark_without_optional_search_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'evaluation').mkdir()
            cases = root / 'cases.jsonl'
            cases.write_text(json.dumps({'id': 'rice', 'query': 'Рис без молока'}, ensure_ascii=False) + '\n', encoding='utf-8')
            commands = []

            class FakeProcess:
                returncode = 0
                def __init__(self, command, **kwargs):
                    commands.append(command)
                def wait(self, **kwargs):
                    return 0

            with patch.object(benchmark, 'ROOT', root), patch.object(benchmark, 'V1', root / 'v1'), \
                 patch.object(benchmark.subprocess, 'Popen', FakeProcess), \
                 patch.object(sys, 'argv', ['benchmark', '--cases', str(cases)]):
                self.assertEqual(benchmark.main(), 0)
            self.assertEqual(len(commands), 4)  # search plus three generation modes
            manifest = next((root / 'evaluation').glob('benchmark_*/manifest.json'))
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual([r['mode'] for r in rows], ['baseline', 'rag', 'agent'])
            self.assertTrue(all(r['query'] == 'Рис без молока' for r in rows))
