"""Recipe agent: generate or repair a recipe grounded in selected sources."""
from agent_runtime import pack_sources, RoleOutputError
from answer_recipe import SYSTEM as RECIPE_SYSTEM, schema_for, validate_recipe


class RecipeAgent:
    role = 'adapter'

    def __init__(self, client, args):
        self.client, self.args = client, args

    def adapt(self, state, instruction, previous_error=None):
        client, args, query = self.client, self.args, state['query']
        system = RECIPE_SYSTEM + "\nТы агент адаптации. Исправь конкретные замечания критика, если они переданы. Сошлись минимум на одну реально использованную основу. Перечисление имеющихся продуктов не запрещает необходимые добавки, если пользователь не ограничил набор явно. Дополнительные продукты перечисли в ingredients и явно отметь в notes, что их нужно иметь или приобрести."
        system += '\nrequirements — предварительный разбор; при расхождении используй original_query.'
        data = {"original_query": query, "requirements": state.get("requirements"), "selection": state["selection"],
                "coordinator_instruction": instruction, "previous_error": previous_error,
                "previous_draft": state["draft"], "critique": state["critique"]}
        recipe_schema = schema_for(state["selected"])
        recipe_schema["properties"]["used_source_ids"]["minItems"] = 1
        data, visible, omitted = pack_sources(client, system, data, recipe_schema,
            state["selected"], "sources", args.max_sources, args.num_predict)
        if len(visible) != len(state["selected"]):
            raise RoleOutputError("Выбранные основы не помещаются вместе с черновиком; select должен выбрать меньше основ")
        def check_recipe(value):
            validate_recipe(value, {r["id"] for r in visible})
            if not value["used_source_ids"]:
                raise ValueError("Укажи минимум одну действительно использованную основу")
        draft = client.call("adapter", system, data, recipe_schema, args.num_predict, check_recipe)
        return draft
