#!/usr/bin/env python3
"""Dense recipe retrieval + cross-encoder reranking, with saved artifacts.

python retrieval.py doctor
python retrieval.py build --chunks recipe_chunks/chunks.jsonl --index recipe_index
python retrieval.py search --index recipe_index --query 'Ужин из курицы и риса' --output search.json

Default models: BAAI/bge-m3 (CLS + L2), BAAI/bge-reranker-v2-m3.
One model occupies the accelerator at a time. CUDA uses FP16, small batches and
automatic batch reduction on OOM. CPU is supported explicitly. No text is silently
truncated. The original recipe records are stored alongside the embedding matrix.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

VERSION = "0.1.0"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
INDEX_FILES = ("chunks.jsonl", "embeddings.npy", "manifest.json")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "torch", "transformers", "sentencepiece"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_chunks(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError(f"{path}:{number}: missing nonempty string id")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{number}: missing nonempty text")
            if row["id"] in seen:
                raise ValueError(f"{path}:{number}: duplicate id {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("The chunks file is empty")
    return rows


def resolve_device(name, torch):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        print("CUDA недоступна: используется CPU. Проверь torch/CUDA через doctor.", file=sys.stderr)
        return "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна в PyTorch. Запусти doctor или выбери --device cpu.")
    return name


class HFStage:
    """Load either encoder or reranker; explicit close releases its GPU memory."""
    def __init__(self, model_id, kind, device="auto", revision=None, max_tokens=8192,
                 local_only=False):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.kind = torch, kind
        self.device = resolve_device(device, torch)
        self.model = None
        self.tokenizer = None
        started = time.perf_counter()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        kwargs = {"revision": revision or "main", "local_files_only": local_only}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        cls = AutoModel if kind == "embedding" else AutoModelForSequenceClassification
        self.model = cls.from_pretrained(model_id, torch_dtype=dtype,
                                        attn_implementation="sdpa", **kwargs)
        self.model.eval()
        self.model.config.use_cache = False
        try:
            self.model.to(self.device)
        except torch.OutOfMemoryError:
            self.close()
            raise RuntimeError("Модель не поместилась в свободную VRAM. Закрой другие GPU-процессы или укажи --device cpu.") from None
        cap = self.tokenizer.model_max_length
        self.max_tokens = min(max_tokens, int(cap)) if cap < 1000000 else max_tokens
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        self.info = {"model": model_id, "kind": kind, "revision_requested": revision or "main",
                     "revision_resolved": getattr(self.model.config, "_commit_hash", None),
                     "device": self.device, "dtype": str(dtype), "max_tokens": self.max_tokens,
                     "parameters": sum(p.numel() for p in self.model.parameters()),
                     "load_seconds": time.perf_counter() - started}

    def tokenize(self, texts, queries=None):
        if queries is None:
            result = self.tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
        else:
            result = self.tokenizer(queries, texts, padding=False, truncation=False, add_special_tokens=True)
        rows = [{k: v[i] for k, v in result.items()} for i in range(len(texts))]
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            if n > self.max_tokens:
                raise ValueError(f"Input #{i} has {n} tokens, limit={self.max_tokens}. Text was NOT truncated.")
        return rows

    def _forward(self, rows):
        torch = self.torch
        inputs = self.tokenizer.pad(rows, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
            if self.kind == "embedding":
                # The official BGE-M3 SentenceTransformers config uses CLS + L2.
                values = torch.nn.functional.normalize(output.last_hidden_state[:, 0].float(), p=2, dim=1)
            else:
                logits = output.logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError("Expected a cross-encoder with one relevance logit per pair")
                values = logits[:, 0].float()
        return values.cpu().numpy()

    def predict(self, texts, batch_size=2, queries=None, progress=False):
        if not texts:
            raise ValueError("No input texts")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        tokenized = self.tokenize(texts, queries)
        # Similar-length inputs reduce wasted padding; outputs return in input order.
        order = sorted(range(len(texts)), key=lambda i: len(tokenized[i]["input_ids"]))
        outputs, position, effective = [None] * len(texts), 0, batch_size
        last_print = 0
        started = time.perf_counter()
        while position < len(order):
            ids = order[position:position + effective]
            try:
                values = self._forward([tokenized[i] for i in ids])
            except self.torch.OutOfMemoryError:
                gc.collect()
                self.torch.cuda.empty_cache()
                if effective == 1:
                    raise RuntimeError("Недостаточно VRAM даже для batch=1. Освободи GPU или выбери --device cpu; текст не обрезан.") from None
                effective = max(1, effective // 2)
                print(f"CUDA OOM: уменьшаю batch_size до {effective}", file=sys.stderr)
                continue
            for i, value in zip(ids, values):
                outputs[i] = value
            position += len(ids)
            if progress and (position - last_print >= 200 or position == len(order)):
                print(f"{self.kind}: {position}/{len(order)} ({time.perf_counter()-started:.1f} s)", flush=True)
                last_print = position
        result = np.asarray(outputs, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Model returned NaN/Inf")
        self.info.update({"inference_seconds": time.perf_counter() - started,
                          "effective_batch_size": effective,
                          "max_input_tokens": max(len(x["input_ids"]) for x in tokenized)})
        if self.device.startswith("cuda"):
            self.info["peak_allocated_mib"] = self.torch.cuda.max_memory_allocated(self.device) / 2**20
            self.info["peak_reserved_mib"] = self.torch.cuda.max_memory_reserved(self.device) / 2**20
        return result

    def close(self):
        self.model, self.tokenizer = None, None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class DenseIndex:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported index format")
        for name, expected in self.manifest["hashes"].items():
            if digest(self.folder / name) != expected:
                raise ValueError(f"Index file {name} was changed or corrupted. Rebuild the index.")
        self.chunks = load_chunks(self.folder / "chunks.jsonl")
        self.vectors = np.load(self.folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        expected_shape = (len(self.chunks), self.manifest["embedding_dimension"])
        if self.vectors.shape != expected_shape or len(self.chunks) != self.manifest["rows"]:
            raise ValueError("Index rows and chunks do not match")
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=0.002):
            raise ValueError("Invalid normalized vectors in index")

    def retrieve(self, vector, k):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.vectors.shape[1] or not np.isfinite(vector).all():
            raise ValueError("Query embedding has wrong dimension or nonfinite values")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Query embedding is zero")
        scores = self.vectors @ (vector / norm)
        ids = np.argsort(-scores, kind="stable")[:min(k, len(self.chunks))]
        return [{**self.chunks[int(i)], "dense_score": float(scores[i]), "dense_rank": rank}
                for rank, i in enumerate(ids, 1)]


def build(args, stage_cls=HFStage):
    started = time.perf_counter()
    folder = Path(args.index)
    if folder.exists() and not folder.is_dir():
        raise ValueError("--index must be a directory")
    if not args.force and any((folder / name).exists() for name in INDEX_FILES):
        raise ValueError("Index exists. Use a new --index directory or add --force.")
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} complete recipe chunks. Loading {args.model}...", flush=True)
    stage = stage_cls(args.model, "embedding", args.device, args.revision,
                      args.max_tokens, args.local_only)
    try:
        vectors = stage.predict([r["text"] for r in chunks], args.batch_size, progress=True)
        info = dict(stage.info)
    finally:
        stage.close()
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding matrix has an invalid shape")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (lengths == 0).any():
        raise ValueError("Invalid or zero embedding")
    vectors = (vectors / lengths).astype(np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".index-build-", dir=folder.parent) as tmp:
        stage_dir = Path(tmp)
        with (stage_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for row in chunks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(stage_dir / "embeddings.npy", vectors, allow_pickle=False)
        manifest = {"format_version": 1, "code_version": VERSION, "rows": len(chunks),
                    "embedding_dimension": vectors.shape[1], "pooling": "cls", "normalized": True,
                    "source_chunks_sha256": digest(args.chunks), "embedding": info,
                    "environment": versions(), "created_utc": datetime.now(timezone.utc).isoformat(),
                    "build_seconds": time.perf_counter() - started,
                    "hashes": {name: digest(stage_dir / name) for name in INDEX_FILES[:2]}}
        save_json(stage_dir / "manifest.json", manifest)
        for name in INDEX_FILES:
            os.replace(stage_dir / name, folder / name)
    print(f"Index saved: {folder.resolve()} ({vectors.shape[0]} x {vectors.shape[1]})")
    return manifest


def search(args, stage_cls=HFStage):
    started = time.perf_counter()
    index = DenseIndex(args.index)
    config = index.manifest["embedding"]
    stage = stage_cls(config["model"], "embedding", args.device,
                      config.get("revision_resolved") or config.get("revision_requested"),
                      config["max_tokens"], args.local_only)
    try:
        vector = stage.predict([args.query], batch_size=1)[0]
        embed_info = dict(stage.info)
    finally:
        stage.close()  # Release encoder BEFORE the reranker is constructed.
    candidates = index.retrieve(vector, args.retrieve_k)
    rerank_info = None
    if args.no_rerank:
        ranked = [dict(row) for row in candidates]
    else:
        stage = stage_cls(args.reranker, "reranker", args.device, args.reranker_revision,
                          args.max_tokens, args.local_only)
        try:
            scores = stage.predict([r["text"] for r in candidates], args.rerank_batch_size,
                                   queries=[args.query] * len(candidates))
            rerank_info = dict(stage.info)
        finally:
            stage.close()
        ranked = [{**row, "rerank_logit": float(score)} for row, score in zip(candidates, scores)]
        ranked.sort(key=lambda row: (-row["rerank_logit"], row["dense_rank"]))
    results = [{**row, "rank": rank} for rank, row in enumerate(ranked[:args.top_k], 1)]
    result = {"version": VERSION, "query": args.query,
              "settings": {"retrieve_k": args.retrieve_k, "top_k": args.top_k, "rerank": not args.no_rerank},
              "index": {"rows": len(index.chunks), "chunks_sha256": index.manifest["hashes"]["chunks.jsonl"]},
              "models": {"embedding": embed_info, "reranker": rerank_info}, "environment": versions(),
              "total_seconds": time.perf_counter() - started,
              "score_meaning": "dense_score=cosine; rerank_logit=uncalibrated relevance, NOT probability",
              "dense_candidates": candidates, "results": results}
    output = Path(args.output) if args.output else Path("runs") / ("search_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:6]+".json")
    save_json(output, result)
    for row in results:
        score = row.get("rerank_logit", row["dense_score"])
        print(f"{row['rank']:2d}. {score:8.4f}  {row.get('title', row['id'])} [{row.get('book', '')}]")
    print(f"\nFull recipes, candidates and timings saved: {output.resolve()}")
    return result


def doctor():
    result = {"environment": versions()}
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update({"gpu": torch.cuda.get_device_name(0), "free_vram_mib": free / 2**20,
                           "total_vram_mib": total / 2**20})
    except ImportError:
        result["error"] = "Install requirements.txt first"
    print(json.dumps(result, ensure_ascii=False, indent=2))


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = ap.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Check dependencies, GPU and free VRAM")
    b = subs.add_parser("build", help="Build and save an embedding index once")
    b.add_argument("--chunks", required=True)
    b.add_argument("--index", default="recipe_index")
    b.add_argument("--model", default=EMBED_MODEL, help="Must be a CLS-pooling encoder compatible with BGE-M3")
    b.add_argument("--revision", default=None)
    b.add_argument("--batch-size", type=positive, default=2)
    b.add_argument("--max-tokens", type=positive, default=8192)
    b.add_argument("--force", action="store_true")
    s = subs.add_parser("search", help="Dense top-100 -> cross-encoder top-10")
    s.add_argument("--index", default="recipe_index")
    s.add_argument("--query", required=True)
    s.add_argument("--retrieve-k", type=positive, default=100)
    s.add_argument("--top-k", type=positive, default=10)
    s.add_argument("--reranker", default=RERANK_MODEL)
    s.add_argument("--reranker-revision", default=None)
    s.add_argument("--rerank-batch-size", type=positive, default=2)
    s.add_argument("--max-tokens", type=positive, default=8192)
    s.add_argument("--no-rerank", action="store_true")
    s.add_argument("--output", default=None, help="Full JSON result; explicit path is replaced if it exists")
    for p in (b, s):
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--local-only", action="store_true", help="Use already cached models without network")
    args = ap.parse_args(argv)
    if args.command == "search":
        if not args.query.strip():
            ap.error("Query cannot be empty")
        if args.top_k > args.retrieve_k:
            ap.error("--top-k cannot exceed --retrieve-k")
    try:
        if args.command == "doctor":
            doctor()
        elif args.command == "build":
            build(args)
        else:
            search(args)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
