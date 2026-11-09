import unittest
from agents.coordinator import CoordinatorAgent
from agent_runtime import RoleOutputError
from agent_test_support import RecordingClient, settings, state


class CoordinatorContractTests(unittest.TestCase):
    def test_only_finish_after_current_passing_review(self):
        data = state()
        data.update(draft={'title': 'Рис'}, checked_version=1, draft_version=1,
                    draft_context_version=0, critique={'verdict': 'pass'})
        client = RecordingClient({'action': 'finish', 'reason': 'Проверено', 'search_query': None})
        CoordinatorAgent(client, settings()).decide(data)
        self.assertEqual(client.requests[0]['data']['allowed_actions'], ['finish'])
        self.assertEqual(client.requests[0]['data']['requirements'], data['requirements'])

    def test_finish_before_critique_is_rejected(self):
        client = RecordingClient({'action': 'finish', 'reason': 'Рано', 'search_query': None})
        with self.assertRaises(RoleOutputError):
            CoordinatorAgent(client, settings()).decide(state())
