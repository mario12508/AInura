"""Coordinator: choose a legal next action within the workflow budget."""
import json
from agent_runtime import role_messages

CHOICE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["action", "reason", "search_query"],
    "properties": {
        "action": {"type": "string", "enum": ["search", "select", "adapt", "critique", "finish", "stop"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 600},
        "search_query": {"type": ["string", "null"]}}}

COORDINATOR = """Ты координатор системы рецептов. Выбери ОДНО следующее действие из
allowed_actions по фактическому состоянию. Можно выбрать search с новой формулировкой,
select для выбора другой основы, adapt для создания/исправления рецепта, critique для
проверки текущего черновика, finish после успешной проверки или stop, если задача не
решается в оставшемся бюджете. Порядок шагов не фиксирован. Не повторяй без причины
уже выполненный поиск или отбор. При исправимой ошибке передай черновик на adapt;
при неподходящей основе выбери иной рецепт или новый поиск. Если уже есть прошедший
проверку актуальный рецепт, заверши работу. При повторном поиске сохраняй исходные
ограничения пользователя. Дай короткое обоснование решения, без развёрнутых рассуждений.
search_query нужен только для search, иначе null. Ответ строго по JSON-схеме."""

def allowed_actions(state, args):
    actions = ["stop"]
    draft_current = state["draft"] is not None and state["draft_context_version"] == state["context_version"]
    verdict = (state.get("critique") or {}).get("verdict")
    if draft_current and state["checked_version"] == state["draft_version"]:
        if verdict == "pass":
            return ["finish"]
        if verdict == "unsatisfiable":
            return actions
    if draft_current and state["checked_version"] != state["draft_version"]:
        return ["stop", "critique"] if state.get("last_error") else ["critique"]
    if (args.index and state["search_rounds"] < args.max_searches
            and not state.get("search_blocked", False)):
        actions.append("search")
    needs_selection = (state.get("selection") is None
                       or (state.get("critique") or {}).get("verdict") == "new_sources")
    if state["candidates"] and needs_selection:
        actions.append("select")
    # A draft must be checked before another repair. A passing or impossible
    # draft must not be rewritten merely because repair budget remains.
    if (state["selected"] and state["draft_count"] < 1 + args.max_repairs
            and (not draft_current or verdict == "revise")):
        actions.append("adapt")
    return actions

COORDINATOR += '\nrequirements — предварительный разбор; при расхождении используй original_query.'


class CoordinatorAgent:
    role = 'coordinator'

    def __init__(self, client, args):
        self.client, self.args = client, args

    def decide(self, state):
        client, args, query = self.client, self.args, state['query']
        actions = allowed_actions(state, args)
        routing = {"original_query": query, "requirements": state.get("requirements"), "allowed_actions": actions,
                   "budget": {"llm_calls_left": args.max_llm_calls-client.calls,
                              "searches_left": args.max_searches-state["search_rounds"],
                              "drafts_left": 1+args.max_repairs-state["draft_count"]},
                   "candidate_titles": [{"id": x["id"], "title": x.get("title", "")} for x in state["candidates"]],
                   "selection": state["selection"], "draft": state["draft"],
                   "draft_is_current": state["draft_context_version"] == state["context_version"],
                   "critique": state["critique"], "previous_search_queries": state["search_queries"],
                   "last_error": state["last_error"]}
        choice_schema = json.loads(json.dumps(CHOICE_SCHEMA))
        choice_schema["properties"]["action"]["enum"] = actions
        # An oversized coordinator context cannot be repaired by calling the
        # same coordinator again without a model call consuming the budget.
        if not client.fits(role_messages(COORDINATOR, routing, choice_schema), min(384,args.num_predict)):
            raise RuntimeError("Состояние не помещается в контекст координатора; сократи запрос или увеличь --num-ctx")
        return client.call("coordinator", COORDINATOR, routing, choice_schema, min(384,args.num_predict))
