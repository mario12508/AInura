import contextlib
import io
import json
import unittest
import agent_recipe as workflow
import test_agents as fixtures
from test_local import TinyTokenizer


class FiveAgentPipelineTests(unittest.TestCase):
    def test_all_five_roles_run_and_requirements_reach_every_role(self):
        fixture = fixtures.AgentTests()
        fixture.setUp()
        try:
            requirements = {'constraints': [{'kind': 'ingredients', 'description': 'Ужин',
                                             'evidence': 'ужин'}], 'conflicts': [], 'questions': []}
            model = fixtures.ScriptedModel([requirements, *fixture.happy()])
            with contextlib.redirect_stdout(io.StringIO()):
                result = workflow.run(fixture.args(), tokenizer=TinyTokenizer(), request_fn=model,
                                      unload_fn=fixture.unload)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['stats']['llm_calls'], 8)
            self.assertEqual(result['stats']['calls_by_role'], {
                'requirements': 1, 'coordinator': 4, 'selector': 1, 'adapter': 1, 'critic': 1})
            self.assertEqual(result['state']['requirements'], requirements)
            for call in result['llm_calls'][1:]:
                payload = json.loads(call['request']['messages'][1]['content'].split('\nJSON_SCHEMA:')[0])
                self.assertEqual(payload['requirements'], requirements)
            self.assertEqual(model.outputs, [])
        finally:
            fixture.tearDown()

    def test_invented_requirements_stop_before_retrieval(self):
        fixture = fixtures.AgentTests()
        fixture.setUp()
        try:
            index = fixture.d / 'index'
            index.mkdir()
            (index / 'manifest.json').write_text('{}', encoding='utf-8')
            model = fixtures.ScriptedModel([{'constraints': [{'kind': 'forbidden', 'description': 'Без молока',
                              'evidence': 'молоко'}], 'conflicts': [], 'questions': []}])
            def unexpected_search(*args):
                self.fail('Invalid requirements must not trigger search')
            with contextlib.redirect_stdout(io.StringIO()):
                result = workflow.run(fixture.args(search=None, query='Рис', index=str(index)),
                    tokenizer=TinyTokenizer(), request_fn=model, unload_fn=fixture.unload,
                    search_fn=unexpected_search)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['status'], 'error')
        finally:
            fixture.tearDown()
