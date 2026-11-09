"""Validate the supplied recipe corpus and optional index without loading models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(chunks, report=None, index=None):
    chunks = Path(chunks)
    ids, books = set(), Counter()
    with chunks.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Line {number}: expected object')
            for field in ('id', 'title', 'text', 'book'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'Line {number}: missing {field}')
            if row['id'] in ids:
                raise ValueError(f'Line {number}: duplicate id {row["id"]}')
            sources = row.get('sources')
            if not isinstance(sources, list) or not sources:
                raise ValueError(f'Line {number}: missing provenance')
            for source in sources:
                if not isinstance(source, dict) or not source.get('file') or not source.get('file_sha256'):
                    raise ValueError(f'Line {number}: incomplete provenance')
            ids.add(row['id'])
            books[row['book']] += 1
    if not ids:
        raise ValueError('Empty corpus')
    result = {'rows': len(ids), 'books': dict(books), 'sha256': sha256(chunks),
              'scope': 'Structure and recorded provenance; not nutrition or recipe quality'}
    if report:
        source_report = json.loads(Path(report).read_text(encoding='utf-8'))
        if source_report['totals']['unique_chunks'] != len(ids):
            raise ValueError('Report row count differs from corpus')
        result['report_rows_match'] = True
    if index:
        import numpy as np
        index = Path(index)
        manifest = json.loads((index / 'manifest.json').read_text(encoding='utf-8'))
        for name in ('chunks.jsonl', 'embeddings.npy'):
            if sha256(index / name) != manifest['hashes'][name]:
                raise ValueError(f'Index hash mismatch: {name}')
        if manifest['hashes']['chunks.jsonl'] != result['sha256']:
            raise ValueError('Index uses a different corpus')
        matrix = np.load(index / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        if matrix.shape != (len(ids), manifest['embedding_dimension']) or manifest['rows'] != len(ids):
            raise ValueError('Embedding dimensions differ from manifest/corpus')
        if not np.isfinite(matrix).all():
            raise ValueError('Non-finite embeddings')
        if manifest.get('normalized') and not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3):
            raise ValueError('Embeddings are not normalized')
        result['index'] = {'shape': list(matrix.shape), 'hashes_match': True, 'finite': True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', default='version_1/recipe_chunks/chunks.jsonl')
    parser.add_argument('--report', default='version_1/recipe_chunks/report.json')
    parser.add_argument('--index')
    parser.add_argument('--output', default='artifacts/data_validation.json')
    args = parser.parse_args()
    result = validate(args.chunks, args.report, args.index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
