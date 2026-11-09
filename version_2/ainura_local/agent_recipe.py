#!/usr/bin/env python3
"""Bounded multi-role recipe agent on the same local Qwen 4B model.

python agent_recipe.py --search search.json --index recipe_index --output agent_answer.json
python agent_recipe.py --query 'Ужин из курицы и риса' --index recipe_index

Coordinator dynamically chooses search/select/adapt/critique/finish/stop.
Roles run sequentially, not as several GPU-resident models. The controller
enforces budgets and requires a current passing critique before acceptance.
An LLM critic's pass is an internal decision, NOT a measured quality score.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import retrieval
from answer_recipe import (MODEL, TOKENIZER, SCHEMA as RECIPE_SCHEMA, SYSTEM as RECIPE_SYSTEM,
                           request_ollama, token_count, validate_recipe, schema_for)

from agent_runtime import (BudgetExceeded, RoleOutputError, validate_shape, role_messages,
                           source_view, pack_sources, unload_ollama, RoleClient, SearchTool)
from agents.coordinator import CoordinatorAgent, CHOICE_SCHEMA, COORDINATOR, allowed_actions
from agents.requirements import RequirementsAgent, validate_requirements
from agents.retrieval_agent import RetrievalAgent, SELECTION_SCHEMA, SELECTOR
from agents.recipe_agent import RecipeAgent
from agents.critic import CriticAgent, CRITIQUE_SCHEMA, CRITIC

VERSION = "0.3.0"


def accept_search(state, data, args):
    rows, seen = [], set()
    for row in data.get("results", []):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
            raise ValueError("Некорректная запись в поисковой выдаче")
        if row["id"] not in seen:
            rows.append(row)
            seen.add(row["id"])
    if not rows:
        raise ValueError("Поиск вернул пустую выдачу")
    state["candidates"] = rows[:args.candidates]
    state["selected"], state["selection"] = [], None
    state["search_blocked"] = False
    state["context_version"] += 1
    state["checked_version"], state["critique"] = -1, None


def normalize_search_query(value):
    """Normalize enough to catch exact retries without changing search semantics."""
    return " ".join(value.split()).casefold() if isinstance(value, str) else ""


def choose_search_query(requested, state):
    """Use a novel coordinator query, then fall back to the selector suggestion."""
    seen = {normalize_search_query(q) for q in state["search_queries"]}
    suggestion = ((state.get("selection") or {}).get("suggested_search_query"))
    for candidate in (requested, suggestion):
        normalized = normalize_search_query(candidate)
        if normalized and normalized not in seen:
            return candidate.strip()
    state["search_blocked"] = True
    raise RoleOutputError(
        "Повторный поиск заблокирован: нужна новая формулировка, отличная от уже выполненных")


def stats(record):
    calls = record["llm_calls"]
    return {"llm_calls": len(calls), "successful_calls": sum(c.get("status") == "ok" for c in calls),
            "calls_by_role": {role: sum(c["role"] == role for c in calls)
                              for role in ("requirements", "coordinator", "selector", "adapter", "critic")},
            "prompt_tokens": sum(c.get("response", {}).get("prompt_eval_count", 0) for c in calls),
            "generated_tokens": sum(c.get("response", {}).get("eval_count", 0) for c in calls),
            "llm_wall_seconds": sum(c.get("seconds", 0) for c in calls),
            "retrieval_calls": len(record["searches"]),
            "retrieval_seconds": sum(c.get("seconds", 0) for c in record["searches"])}


def run(args, tokenizer=None, request_fn=request_ollama, search_fn=None, unload_fn=None, requirements_fn=None):
    started = time.perf_counter()
    output = Path(args.output).resolve() if args.output else Path("runs").resolve() / (
        "agent_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:6]+".json")
    artifacts = output.parent / (output.stem + "_artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    initial = json.loads(Path(args.search).read_text(encoding="utf-8")) if args.search else None
    query = initial.get("query") if initial is not None else args.query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Нужен непустой исходный запрос")
    if args.index and not (Path(args.index) / "manifest.json").exists():
        raise ValueError("--index не содержит manifest.json")
    if initial is not None and args.index:
        manifest = json.loads((Path(args.index) / "manifest.json").read_text(encoding="utf-8"))
        if initial.get("index", {}).get("chunks_sha256") != manifest["hashes"]["chunks.jsonl"]:
            raise ValueError("Сохранённый поиск и --index используют разные корпуса")
    state = {"query": query, "candidates": [], "selection": None, "selected": [],
             "draft": None, "draft_count": 0, "draft_version": 0, "checked_version": -1,
             "context_version": 0, "draft_context_version": -1, "critique": None,
             "search_rounds": 0, "search_queries": [], "search_blocked": False,
             "last_error": None}
    record = {"version": VERSION, "mode": "agent", "query": query, "model": args.model,
              "settings": vars(args).copy(), "environment": retrieval.versions(),
              "status": "running", "accepted": False, "recipe": None, "state": state,
              "initial_search": initial, "searches": [], "llm_calls": [], "events": [],
              "acceptance_meaning": "Internal LLM critic pass, not an independent quality measurement",
              "timing_scope": "Includes new searches; excludes creation of a supplied --search artifact"}

    def persist():
        record["stats"] = stats(record)
        record["total_seconds"] = time.perf_counter() - started
        retrieval.save_json(output, record)

    client = RoleClient(args, record, persist, tokenizer, request_fn)
    if unload_fn:
        client.unload = unload_fn
    search_tool = search_fn or SearchTool(args, artifacts, client)
    coordinator = CoordinatorAgent(client, args)
    researcher = RetrievalAgent(client, args)
    recipe_agent = RecipeAgent(client, args)
    critic = CriticAgent(client, args)

    def do_search(search_query):
        search_query = choose_search_query(search_query, state)
        if state["search_rounds"] >= args.max_searches:
            raise RoleOutputError("Исчерпан лимит поисковых раундов")
        entry = {"query": search_query, "round": state["search_rounds"]+1}
        record["searches"].append(entry)
        state["search_rounds"] += 1
        state["search_queries"].append(search_query)
        persist()
        tool_start = time.perf_counter()
        try:
            data = search_tool(search_query, entry["round"])
            entry["result"] = data
            accept_search(state, data, args)
            entry["status"] = "ok"
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            entry.update({"status": "error", "error": str(e)})
            raise RoleOutputError(f"Инструмент поиска не выполнился: {e}") from e
        finally:
            entry["seconds"] = time.perf_counter() - tool_start
            persist()
        return search_query

    try:
        state['requirements'] = (requirements_fn(query) if requirements_fn is not None
                                 else RequirementsAgent(client, args).analyze(query))
        validate_requirements(query, state['requirements'])
        persist()
        if initial is not None:
            accept_search(state, initial, args)
            state["search_rounds"], state["search_queries"] = 1, [query]
            record["events"].append({"action": "initial_search", "cached": True})
        else:
            do_search(query)
        persist()
        while True:
            if client.calls >= args.max_llm_calls:
                raise BudgetExceeded("Исчерпан лимит вызовов LLM")
            actions = allowed_actions(state, args)
            try:
                action = "coordinator"
                decision = coordinator.decide(state)
                action = decision["action"]
                if action not in actions:
                    raise RoleOutputError(f"Недоступное действие {action}; доступны {actions}")
                event = {"action": action, "reason": decision["reason"], "call_number": client.calls}
                record["events"].append(event)
                print(f"[{client.calls}/{args.max_llm_calls}] {action}: {decision['reason']}", flush=True)
                previous_error = state["last_error"]
                state["last_error"] = None
                if action == "stop":
                    if (state["draft"] is None and state["selected"]
                            and (state.get("selection") or {}).get("sufficient")):
                        record.update(status="source_selected", source_recipes=state["selected"],
                                      reason=decision["reason"])
                    else:
                        calls_left = max(0,args.max_llm_calls-client.calls)
                        searches_left = max(0,args.max_searches-state["search_rounds"])
                        record.update(status="stopped",coordinator_reason=decision["reason"],
                                      reason=("Агент остановил подбор без принятого рецепта. "
                                              f"Осталось вызовов LLM: {calls_left}; поисковых раундов: {searches_left}."))
                    break
                if action == "finish":
                    record.update({"status": "accepted", "accepted": True,
                                   "recipe": state["draft"], "reason": decision["reason"]})
                    break
                if action == "search":
                    if not isinstance(decision["search_query"], str):
                        raise RoleOutputError("Для search нужен search_query")
                    requested_query = decision["search_query"]
                    used_query = do_search(requested_query)
                    event["search_query"] = used_query
                    if normalize_search_query(used_query) != normalize_search_query(requested_query):
                        event["requested_search_query"] = requested_query
                elif action == "select":
                    selection, visible, omitted = researcher.select(state, decision['reason'], previous_error)
                    state["selection"] = selection
                    ids = [x["id"] for x in selection["selected"]]
                    if not selection["sufficient"]:
                        event["discarded_selected_ids"] = ids
                        ids = []
                    state["selected"] = [next(r for r in visible if r["id"] == i) for i in ids]
                    state["context_version"] += 1
                    state["critique"], state["checked_version"] = None, -1
                    event["omitted_candidates"] = omitted
                elif action == "adapt":
                    draft = recipe_agent.adapt(state, decision['reason'], previous_error)
                    state["draft"] = draft
                    state["draft_count"] += 1
                    state["draft_version"] += 1
                    state["draft_context_version"] = state["context_version"]
                    state["critique"], state["checked_version"] = None, -1
                elif action == "critique":
                    state['critique'] = critic.review(state, previous_error)
                    state["checked_version"] = state["draft_version"]
                state.pop("repeated_failure", None)
                persist()
            except RoleOutputError as e:
                state["last_error"] = str(e)
                record["events"].append({"action": "recoverable_error", "error": str(e)})
                signature = [action, state["context_version"], state["draft_version"], str(e)]
                previous = state.get("repeated_failure", {})
                count = previous.get("count", 0) + 1 if previous.get("signature") == signature else 1
                state["repeated_failure"] = {"signature": signature, "count": count}
                persist()
                if client.calls >= args.max_llm_calls:
                    raise BudgetExceeded("Исчерпан лимит вызовов LLM")
                if count >= 2:
                    record.update(status="stopped", reason=(
                        f"Действие {action} дважды завершилось одной ошибкой без изменения состояния: {e}"))
                    break
    except BudgetExceeded as e:
        record.update({"status": "budget_exhausted", "reason": str(e)})
    except KeyboardInterrupt:
        record.update({"status": "interrupted", "reason": "Interrupted by user"})
    except (ValueError, RuntimeError, OSError) as e:
        record.update({"status": "error", "reason": str(e)})
    finally:
        try:
            client.unload()
        except (ValueError, RuntimeError, OSError) as e:
            record["unload_error"] = str(e)
        persist()
    print(f"\nStatus: {record['status']}; LLM calls: {record['stats']['llm_calls']}; searches: {record['stats']['retrieval_calls']}")
    if record["accepted"]:
        print(json.dumps(record["recipe"], ensure_ascii=False, indent=2))
    elif record["status"] == "source_selected":
        for source in record["source_recipes"]:
            print(source["text"])
    elif state["draft"] is not None:
        print("Сохранён черновик; он не принят системой. Причина: "+record.get("reason", ""))
    print(f"Полный лог: {output}")
    return record


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    entry = ap.add_mutually_exclusive_group(required=True)
    entry.add_argument("--search", help="Saved initial search; recommended for fair comparison with RAG")
    entry.add_argument("--query", help="Run initial retrieval directly")
    ap.add_argument("--index", help="Existing recipe_index; enables autonomous repeat searches")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--max-llm-calls", type=int, default=30)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--max-searches", type=int, default=4, help="Includes cached or fresh initial search")
    ap.add_argument("--max-sources", type=int, choices=[1,2,3], default=2)
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--retrieve-k", type=int, default=100)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--search-timeout", type=float, default=900)
    args = ap.parse_args(argv)
    if args.query is not None and not args.index:
        ap.error("--query requires --index")
    if min(args.max_llm_calls,args.max_searches,args.candidates,args.retrieve_k,args.num_ctx,args.num_predict,args.timeout,args.search_timeout) <= 0 or args.max_repairs < 0:
        ap.error("Limits must be positive; --max-repairs may be zero")
    if args.candidates > args.retrieve_k:
        ap.error("--candidates cannot exceed --retrieve-k")
    try:
        result = run(args)
        return 0 if result["accepted"] or result["status"] == "source_selected" else 2
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
