import unittest
from agents.requirements import RequirementsAgent
from agent_test_support import RecordingClient, settings, state


class RequirementsContractTests(unittest.TestCase):
    def test_explicit_constraint_has_original_evidence(self):
        data = state()
        client = RecordingClient(data['requirements'])
        result = RequirementsAgent(client, settings()).analyze(data['query'])
        self.assertEqual(result['constraints'][0]['evidence'], 'без молока')
        self.assertEqual(client.requests[0]['role'], 'requirements')

    def test_invented_constraint_is_rejected(self):
        result = state()['requirements']
        result['constraints'][0]['evidence'] = 'без арахиса'
        with self.assertRaisesRegex(ValueError, 'цитатой'):
            RequirementsAgent(RecordingClient(result), settings()).analyze('Рис без молока')
