import unittest
from agents.critic import CriticAgent
from agent_test_support import RecordingClient, settings, state


class CriticContractTests(unittest.TestCase):
    def test_critic_receives_draft_without_unadapted_book_text(self):
        data = state()
        data['draft'] = {'title': 'Адаптированный рис'}
        client = RecordingClient({'verdict': 'pass', 'issues': [], 'summary': 'OK'})
        CriticAgent(client, settings()).review(data)
        payload = client.requests[0]['data']
        self.assertEqual(payload['recipe'], data['draft'])
        self.assertNotIn('Книжный рецепт', str(payload))

    def test_pass_with_blocking_issue_is_rejected(self):
        value = {'verdict': 'pass', 'issues': [{'category': 'constraints', 'severity': 'blocking',
                 'message': 'Запрет нарушен', 'suggested_fix': 'Удалить молоко'}], 'summary': 'OK'}
        with self.assertRaises(ValueError):
            CriticAgent(RecordingClient(value), settings()).review(state())
