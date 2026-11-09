"""Shared local LLM transport, budgets and retrieval tool; no agent policy."""
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import retrieval
from answer_recipe import request_ollama, token_count

class BudgetExceeded(RuntimeError):
    pass


class RoleOutputError(ValueError):
    pass


def validate_shape(value, schema, path="response"):
    """Validate the small JSON-Schema subset used by this module, without new deps."""
    kinds = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
             "string": lambda v: isinstance(v, str), "boolean": lambda v: type(v) is bool,
             "integer": lambda v: type(v) is int,
             "number": lambda v: type(v) in {int, float}, "null": lambda v: v is None}
    types = schema.get("type", list(kinds))
    types = [types] if isinstance(types, str) else types
    if not any(kinds[t](value) for t in types):
        raise RoleOutputError(f"{path}: invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise RoleOutputError(f"{path}: invalid enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise RoleOutputError(f"{path}: missing {key}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise RoleOutputError(f"{path}: unknown fields")
        for key, item in value.items():
            if key in props:
                validate_shape(item, props[key], path+"."+key)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise RoleOutputError(f"{path}: invalid array length")
        if schema.get("uniqueItems") and len({json.dumps(x,sort_keys=True) for x in value}) != len(value):
            raise RoleOutputError(f"{path}: duplicate items")
        for i, item in enumerate(value):
            validate_shape(item, schema.get("items", {}), f"{path}[{i}]")
    elif isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise RoleOutputError(f"{path}: invalid string length")
    elif type(value) in {int, float}:
        if value != value or abs(value) == float("inf") or value < schema.get("minimum", -float("inf")):
            raise RoleOutputError(f"{path}: invalid number")


def role_messages(system, data, schema):
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)+
             "\nJSON_SCHEMA:\n"+json.dumps(schema, ensure_ascii=False)}]


def source_view(row):
    return {"id": row["id"], "title": row.get("title", ""), "book": row.get("book", ""), "text": row["text"]}


def pack_sources(client, system, data, schema, rows, key, limit, output_tokens):
    """Put whole recipes into context; report every omission, never truncate."""
    selected, omitted = [], []
    for row in rows:
        proposed = selected + [row]
        payload = {**data, key: [source_view(r) for r in proposed]}
        if len(selected) >= limit:
            omitted.append({"id": row["id"], "reason": "candidate_limit"})
        elif not client.fits(role_messages(system, payload, schema), output_tokens):
            omitted.append({"id": row["id"], "reason": "context_budget"})
        else:
            selected = proposed
    payload = {**data, key: [source_view(r) for r in selected]}
    return payload, selected, omitted


def unload_ollama(base_url, model, timeout):
    request = Request(base_url.rstrip("/")+"/api/generate",
                      data=json.dumps({"model": model, "prompt": "", "stream": False, "keep_alive": 0}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data
    except (HTTPError, URLError, OSError, ValueError) as e:
        raise RuntimeError(f"Не удалось выгрузить {model} перед поиском: {e}") from e


class RoleClient:
    def __init__(self, args, record, persist, tokenizer=None, request_fn=request_ollama):
        self.args, self.record, self.persist, self.request_fn = args, record, persist, request_fn
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_only)
        self.tokenizer = tokenizer

    @property
    def calls(self):
        return len(self.record["llm_calls"])

    def fits(self, messages, output_tokens):
        return token_count(self.tokenizer, messages) + output_tokens + 256 <= self.args.num_ctx

    def call(self, role, system, data, schema, output_tokens, validate=None):
        if self.calls >= self.args.max_llm_calls:
            raise BudgetExceeded("Исчерпан лимит вызовов LLM")
        messages = role_messages(system, data, schema)
        if not self.fits(messages, output_tokens):
            raise RoleOutputError(f"Контекст роли {role} не помещается; выбери меньше основ или остановись")
        payload = {"model": self.args.model, "messages": messages, "format": schema,
                   "stream": False, "keep_alive": "2m",
                   "options": {"num_ctx": self.args.num_ctx, "num_predict": output_tokens,
                               "temperature": self.args.temperature if role == "adapter" else 0,
                               "seed": self.args.seed}}
        entry = {"number": self.calls+1, "role": role, "request": payload,
                 "estimated_prompt_tokens": token_count(self.tokenizer, messages)}
        self.record["llm_calls"].append(entry)
        self.persist()
        started = time.perf_counter()
        try:
            response = self.request_fn(self.args.ollama_url, payload, self.args.timeout)
            entry["response"] = response
            if response.get("done_reason") == "length":
                raise RoleOutputError(f"Роль {role} достигла лимита генерации")
            try:
                value = json.loads(response.get("message", {}).get("content", ""))
            except ValueError as e:
                raise RoleOutputError(f"Роль {role} вернула невалидный JSON: {e}") from e
            validate_shape(value, schema)
            if validate:
                try:
                    validate(value)
                except ValueError as e:
                    raise RoleOutputError(str(e)) from e
            entry["value"], entry["status"] = value, "ok"
            return value
        except (ValueError, RuntimeError, OSError) as e:
            entry["error"], entry["status"] = str(e), "error"
            raise
        except KeyboardInterrupt:
            entry["error"], entry["status"] = "Interrupted by user", "interrupted"
            raise
        finally:
            entry["seconds"] = time.perf_counter() - started
            self.persist()

    def unload(self):
        return unload_ollama(self.args.ollama_url, self.args.model, self.args.timeout)


class SearchTool:
    def __init__(self, args, folder, client):
        self.args, self.folder, self.client = args, folder, client

    def __call__(self, query, number):
        # Explicitly unload the local LLM before starting PyTorch on an 8GB GPU.
        self.client.unload()
        output = self.folder / f"retrieval_{number:02d}.json"
        command = [sys.executable, str(Path(retrieval.__file__).resolve()), "search",
                   "--index", str(Path(self.args.index).resolve()), "--query", query,
                   "--retrieve-k", str(self.args.retrieve_k), "--top-k", str(self.args.candidates),
                   "--device", self.args.device, "--output", str(output)]
        if self.args.local_only:
            command.append("--local-only")
        proc = subprocess.run(command, capture_output=True, text=True, timeout=self.args.search_timeout)
        if proc.returncode:
            raise RuntimeError(f"Поиск завершился с ошибкой:\n{proc.stderr[-4000:]}\n{proc.stdout[-1000:]}")
        data = json.loads(output.read_text(encoding="utf-8"))
        if data.get("query") != query:
            raise ValueError("Поисковый инструмент вернул другой запрос")
        return data
