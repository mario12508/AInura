import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validate_data import validate


class DataValidationTests(unittest.TestCase):
    def test_duplicate_and_missing_provenance_are_rejected(self):
        row = {'id': 'one', 'title': 'Rice', 'text': 'Cook rice', 'book': 'Book',
               'sources': [{'file': 'book.fb2', 'file_sha256': 'a' * 64}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chunks.jsonl'
            for rows in ([row, row], [{**row, 'sources': []}]):
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
                with self.assertRaises(ValueError):
                    validate(path)
