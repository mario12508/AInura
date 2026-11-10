import unittest
from agents.recipe_agent import RecipeAgent
from agent_runtime import RoleOutputError
from agent_test_support import RecordingClient, settings, state


class RecipeContractTests(unittest.TestCase):
    def recipe(self, ids):
        return {'title': 'Рис', 'servings': 2,
                'ingredients': [{'name': 'рис', 'amount': 100, 'unit': 'г'}],
                'steps': ['Сварить рис'], 'used_source_ids': ids, 'notes': []}

    def test_repair_receives_current_critique(self):
        data = state()
        data['critique'] = {'verdict': 'revise', 'summary': 'Уточнить граммы'}
        client = RecordingClient(self.recipe(['a']))
        RecipeAgent(client, settings()).adapt(data, 'Исправить')
        self.assertEqual(client.requests[0]['data']['critique'], data['critique'])
        self.assertEqual(client.requests[0]['data']['requirements'], data['requirements'])

    def test_ungrounded_recipe_is_rejected(self):
        with self.assertRaises(RoleOutputError):
            RecipeAgent(RecordingClient(self.recipe([])), settings()).adapt(state(), 'Создать')
