import unittest
from agents.retrieval_agent import RetrievalAgent
from agent_runtime import RoleOutputError
from agent_test_support import RecordingClient, settings, state


class RetrievalContractTests(unittest.TestCase):
    def selection(self, source):
        return {'selected': [{'id': source, 'why_fits': 'Рис', 'required_changes': [], 'risks': []}],
                'sufficient': True, 'suggested_search_query': None, 'notes': []}

    def test_selected_sources_and_requirements_are_exposed(self):
        data = state()
        client = RecordingClient(self.selection('a'))
        result, visible, omitted = RetrievalAgent(client, settings()).select(data, 'Выбрать основу')
        self.assertTrue(result['sufficient'])
        self.assertEqual([r['id'] for r in visible], ['a'])
        self.assertEqual(omitted, [])
        self.assertEqual(client.requests[0]['data']['requirements'], data['requirements'])

    def test_invented_source_is_rejected(self):
        with self.assertRaises(RoleOutputError):
            RetrievalAgent(RecordingClient(self.selection('invented')), settings()).select(state(), 'Выбрать')
