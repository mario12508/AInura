import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.v1 = self.root/'v1'; self.v1.mkdir()
        self.v2 = self.root/'v2'; self.v2.mkdir()
        (self.v2/'agent_recipe.py').touch()
        self.index = self.root/'index'; self.index.mkdir()
        (self.index/'manifest.json').write_text('{}')
        self.patches = [patch.object(server,'RUNS',self.root/'runs'),patch.object(server,'V1',self.v1),
                        patch.object(server,'V2',self.v2),
                        patch.object(server,'INDEX',self.index),patch.object(server,'JOBS',{}),patch.object(server,'ACTIVE',None)]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_reject_invalid_input(self):
        for body in ({'query':'','mode':'v1'}, {'query':'x','mode':'bad'}, {'query':4,'mode':'v2'}):
            with self.assertRaises(ValueError): server.start_job(body)

    def test_one_job_at_a_time(self):
        with patch.object(server.threading.Thread,'start'):
            first = server.start_job({'query':'ужин','mode':'v1'})
            with self.assertRaises(RuntimeError):server.start_job({'query':'обед','mode':'v1'})
        self.assertTrue((server.RUNS/first/'job.json').exists())

    def test_matching_cached_search_and_recipe_result(self):
        (self.v1/'search.json').write_text(json.dumps({'query':'ужин','results':[{'id':'1'}]}))
        with patch.object(server.threading.Thread,'start'):
            key = server.start_job({'query':'ужин','mode':'v1','reuse':True})
        def execute(job, command, stage):
            if 'format_recipe.py' in command[2]:return 0
            self.assertIn('answer_recipe.py',command[2])
            (server.RUNS/key/'answer.json').write_text('{"status":"ok","recipe":{"title":"Рис"}}')
            return 0
        with patch.object(server,'execute',side_effect=execute) as run:
            server.worker(server.JOBS[key])
        self.assertEqual(run.call_count,2)
        self.assertEqual(server.JOBS[key]['status'],'done')
        self.assertIsNone(server.ACTIVE)
        self.assertEqual(server.snapshot(server.JOBS[key])['result']['recipe']['title'],'Рис')

    def test_different_query_runs_search_and_preserves_failure(self):
        (self.v1/'search.json').write_text('{"query":"other"}')
        with patch.object(server.threading.Thread,'start'):
            key = server.start_job({'query':'ужин','mode':'v1','reuse':True})
        with patch.object(server,'execute',return_value=1) as run:
            server.worker(server.JOBS[key])
        self.assertIn('retrieval.py',run.call_args.args[1][2])
        self.assertEqual(server.JOBS[key]['status'],'error')
        self.assertIsNone(server.ACTIVE)

    def prepared_recipe(self):
        with patch.object(server.threading.Thread,'start'):
            key=server.start_job({'query':'мясо','mode':'v2'})
        job=server.JOBS[key];job['status']='stopped';server.ACTIVE=None
        (server.RUNS/key/'answer.json').write_text(json.dumps({
            'query':'мясо','state':{'selected':[{'id':'a','text':'Запечь мясо.'}]}}))
        return key

    def test_questions_persist_and_share_gpu_lock_with_generation(self):
        key=self.prepared_recipe()
        with patch.object(server.threading.Thread,'start'):
            turn=server.start_question(key,{'question':'Без чеснока можно?'})
        with self.assertRaises(RuntimeError):server.start_question(key,{'question':'Ещё вопрос'})
        with self.assertRaises(RuntimeError):server.start_job({'query':'суп','mode':'v1'})
        def execute(job,command,stage):
            path=server.RUNS/key/'chat.json';turns=json.loads(path.read_text())
            turns[-1].update(status='done',answer='Можно.',request={'private':'prompt'})
            server.save_chat(path,turns)
            return 0
        with patch.object(server,'execute',side_effect=execute):
            server.question_worker(server.JOBS[key],turn)
        self.assertIsNone(server.ACTIVE)
        snapshot=server.snapshot(server.JOBS[key])
        self.assertEqual(snapshot['conversation'][0]['answer'],'Можно.')
        self.assertNotIn('request',snapshot['conversation'][0])
        self.assertEqual(server.JOBS[key]['status'],'stopped')
        with patch.object(server.threading.Thread,'start'):
            server.start_question(key,{'question':'А на две порции?'})
        self.assertEqual(len(json.loads((server.RUNS/key/'chat.json').read_text())),2)

    def test_server_restart_recovers_completed_history_and_marks_pending_turn(self):
        key=self.prepared_recipe()
        turns=[{'id':'old','question':'Первый','answer':'Готовый ответ','status':'done'},
               {'id':'pending','question':'Второй','status':'running'}]
        server.save_chat(server.RUNS/key/'chat.json',turns)
        with patch.object(server.sys,'argv',['server.py']), patch.object(server,'ThreadingHTTPServer') as http:
            http.return_value.serve_forever.side_effect=KeyboardInterrupt
            server.main()
        restored=server.snapshot(server.JOBS[key])['conversation']
        self.assertEqual(restored[0]['answer'],'Готовый ответ')
        self.assertEqual(restored[1]['status'],'error')
        self.assertIn('перезапуском',restored[1]['error'])

    def test_chat_can_call_agents_and_preserves_original_card(self):
        key=self.prepared_recipe();path=server.RUNS/key/'answer.json';original=path.read_text()
        with patch.object(server.threading.Thread,'start'):
            turn_id=server.start_question(key,{'question':'Найди другой рецепт без жарки','allow_agents':True})
        def execute(job,command,stage):
            self.assertEqual(server.ACTIVE,'question:'+key)
            chat_path=server.RUNS/key/'chat.json'
            if 'recipe_chat.py' in command[2]:
                turns=json.loads(chat_path.read_text());turn=turns[-1]
                if '--agent-result' in command:
                    result_path=Path(command[command.index('--agent-result')+1])
                    self.assertEqual(json.loads(result_path.read_text())['recipe']['title'],'Мясо на пару')
                    turn.update(status='done',answer='Нашёл мясо на пару.')
                else:
                    turn.update(status='agent_requested',agent_query='Другой рецепт мяса без жарки')
                server.save_chat(chat_path,turns)
            elif 'retrieval.py' in command[2]:
                output=Path(command[command.index('--output')+1])
                server.save_chat(output,{'query':command[command.index('--query')+1],'results':[{'id':'s','text':'Готовить на пару.'}]})
            elif 'agent_recipe.py' in command[2]:
                self.assertIn('--search',command)
                search=json.loads(Path(command[command.index('--search')+1]).read_text())
                self.assertTrue(search['query'].startswith('Другой рецепт мяса без жарки'))
                output=Path(command[command.index('--output')+1])
                server.save_chat(output,{'status':'accepted','accepted':True,'query':'Другой рецепт мяса без жарки',
                                        'recipe':{'title':'Мясо на пару','steps':['Готовить на пару.']}})
            elif 'format_recipe.py' in command[2]:
                self.assertEqual(job['status'],'running')
            return 0
        with patch.object(server,'execute',side_effect=execute):server.question_worker(server.JOBS[key],turn_id)
        conversation=server.snapshot(server.JOBS[key])['conversation']
        self.assertEqual(conversation[-1]['status'],'done')
        self.assertTrue(conversation[-1]['agent_used'])
        child=server.JOBS[conversation[-1]['child_id']]
        self.assertEqual(child['parent_id'],key)
        self.assertEqual(child['status'],'done')
        self.assertEqual(path.read_text(),original)
        self.assertIsNone(server.ACTIVE)

    def test_agent_chat_pauses_for_ingredients_before_starting_controller(self):
        key=self.prepared_recipe()
        with patch.object(server.threading.Thread,'start'):
            turn_id=server.start_question(key,{'question':'Другой пирог','allow_agents':True,'allow_clarifications':True})
        def execute(job,command,stage):
            if 'recipe_chat.py' in command[2]:
                path=server.RUNS/key/'chat.json';turns=json.loads(path.read_text())
                turns[-1].update(status='agent_requested',agent_query='Другой пирог')
                server.save_chat(path,turns)
            elif 'retrieval.py' in command[2]:
                server.save_chat(Path(command[command.index('--output')+1]),{'query':'Другой пирог','results':[]})
            elif 'clarify_recipe.py' in command[2]:
                server.save_chat(Path(command[command.index('--output')+1]),{'action':'ask','question':'Можно добавить муку?',
                                                                           'ingredients':['мука']})
            else:self.fail('Controller must wait for user permission')
            return 0
        with patch.object(server,'execute',side_effect=execute):server.question_worker(server.JOBS[key],turn_id)
        turn=server.snapshot(server.JOBS[key])['conversation'][-1]
        self.assertEqual(turn['status'],'done')
        self.assertTrue(turn['needs_clarification'])
        self.assertEqual(turn['child_clarification']['ingredients'],['мука'])
        self.assertEqual(server.JOBS[turn['child_id']]['status'],'waiting_input')
        self.assertIsNone(server.ACTIVE)

    def test_agent_request_cannot_bypass_unchecked_permission(self):
        key=self.prepared_recipe()
        with patch.object(server.threading.Thread,'start'):
            turn_id=server.start_question(key,{'question':'Вопрос','allow_agents':False})
        def execute(job,command,stage):
            self.assertIn('recipe_chat.py',command[2])
            path=server.RUNS/key/'chat.json';turns=json.loads(path.read_text())
            turns[-1].update(status='agent_requested',agent_query='Новый рецепт')
            server.save_chat(path,turns);return 0
        with patch.object(server,'execute',side_effect=execute) as run:
            server.question_worker(server.JOBS[key],turn_id)
        self.assertEqual(run.call_count,1)
        self.assertEqual(server.snapshot(server.JOBS[key])['conversation'][-1]['status'],'error')
        self.assertEqual(len(server.JOBS),1)
        self.assertIsNone(server.ACTIVE)

    def test_question_failure_releases_lock_and_keeps_recipe(self):
        key=self.prepared_recipe()
        with patch.object(server.threading.Thread,'start'):
            turn=server.start_question(key,{'question':'Как запечь?'})
        with patch.object(server,'execute',side_effect=RuntimeError('Ollama недоступна')):
            server.question_worker(server.JOBS[key],turn)
        self.assertIsNone(server.ACTIVE)
        self.assertEqual(server.snapshot(server.JOBS[key])['conversation'][0]['status'],'error')
        self.assertIsNotNone(server.snapshot(server.JOBS[key])['result'])

    def test_question_validation_and_missing_recipe(self):
        key=self.prepared_recipe()
        for question in ('',42,'x'*4001):
            with self.assertRaises(ValueError):server.start_question(key,{'question':question})
        with self.assertRaises(ValueError):server.start_question('missing',{'question':'Как?'})
        (server.RUNS/key/'answer.json').write_text('{}')
        with self.assertRaises(ValueError):server.start_question(key,{'question':'Как?'})
        self.assertIsNone(server.ACTIVE)

    def test_elapsed_uses_monotonic_time_instead_of_wall_clocks(self):
        key=self.prepared_recipe();job=server.JOBS[key]
        job.update(status='running',started=99999999999,_started_monotonic=100)
        with patch.object(server.time,'monotonic',return_value=107.5), patch.object(server.time,'time',return_value=-100):
            self.assertEqual(server.snapshot(job)['elapsed_seconds'],7.5)
        job.update(status='done',finished=-1,elapsed_seconds=12.5)
        self.assertEqual(server.snapshot(job)['elapsed_seconds'],12.5)
        job.pop('elapsed_seconds');job.pop('_started_monotonic')
        self.assertEqual(server.snapshot(job)['elapsed_seconds'],0)

    def test_format_existing_recipe_shares_lock_and_preserves_answer(self):
        key=self.prepared_recipe();path=server.RUNS/key/'answer.json';original=path.read_text()
        with patch.object(server.threading.Thread,'start'):
            self.assertEqual(server.start_format(key),key)
        with self.assertRaises(RuntimeError):server.start_question(key,{'question':'Вопрос'})
        self.assertTrue(server.snapshot(server.JOBS[key])['formatting'])
        with patch.object(server,'execute',return_value=0):server.format_worker(server.JOBS[key])
        self.assertIsNone(server.ACTIVE)
        self.assertEqual(path.read_text(),original)

    def test_optional_formatter_failure_does_not_lose_generated_recipe(self):
        (self.v1/'search.json').write_text(json.dumps({'query':'ужин','results':[{'id':'1'}]}))
        with patch.object(server.threading.Thread,'start'):
            key=server.start_job({'query':'ужин','mode':'v1','reuse':True})
        def execute(job,command,stage):
            if 'format_recipe.py' in command[2]:raise RuntimeError('Formatter unavailable')
            (server.RUNS/key/'answer.json').write_text('{"status":"ok","recipe":{"title":"Рис"}}')
            return 0
        with patch.object(server,'execute',side_effect=execute):server.worker(server.JOBS[key])
        job=server.snapshot(server.JOBS[key])
        self.assertEqual(job['status'],'done')
        self.assertEqual(job['result']['recipe']['title'],'Рис')
        self.assertGreaterEqual(job['elapsed_seconds'],0)
        self.assertIsNone(server.ACTIVE)

    def pause_for_ingredients(self):
        (self.v1/'search.json').write_text(json.dumps({'query':'пирог из баранины','results':[{'id':'a','text':'Пирог с тестом.'}]}))
        with patch.object(server.threading.Thread,'start'):
            key=server.start_job({'query':'пирог из баранины','mode':'v2','reuse':True,'allow_clarifications':True})
        def execute(job,command,stage):
            self.assertIn('clarify_recipe.py',command[2])
            server.save_chat(server.RUNS/key/'clarification.json',{'action':'ask','question':'Можно добавить муку и яйцо?',
                                                                 'ingredients':['мука','яйцо']})
            return 0
        with patch.object(server,'execute',side_effect=execute) as run:server.worker(server.JOBS[key])
        self.assertEqual(run.call_count,1)
        self.assertEqual(server.JOBS[key]['status'],'waiting_input')
        self.assertIsNone(server.ACTIVE)
        self.assertFalse((server.RUNS/key/'answer.json').exists())
        return key

    def test_clarification_pauses_and_resumes_without_new_search(self):
        key=self.pause_for_ingredients();job=server.JOBS[key]
        search_before=(server.RUNS/key/'search.json').read_text()
        old_elapsed=job['elapsed_seconds']
        with patch.object(server.threading.Thread,'start'):
            server.respond_clarification(key,{'clarification_id':job['clarification']['id'],'choice':'allow'})
        self.assertIn('мука, яйцо',job['clarification']['answer'])
        def execute(job,command,stage):
            if 'format_recipe.py' in command[2]:return 0
            self.assertIn('agent_recipe.py',command[2])
            generation=json.loads(Path(command[command.index('--search')+1]).read_text())
            self.assertIn('Можно добавить следующие продукты: мука, яйцо',generation['query'])
            self.assertEqual(generation['retrieval_query'],'пирог из баранины')
            server.save_chat(server.RUNS/key/'answer.json',{'status':'accepted','accepted':True,'recipe':{'title':'Пирог'}})
            return 0
        with patch.object(server,'execute',side_effect=execute):server.worker(job)
        self.assertEqual(job['status'],'done')
        self.assertGreaterEqual(job['elapsed_seconds'],old_elapsed)
        self.assertEqual((server.RUNS/key/'search.json').read_text(),search_before)
        self.assertIsNone(server.ACTIVE)

    def test_clarification_decline_and_custom_answers_are_not_approval(self):
        for choice,answer,expected in (('deny','','Нельзя добавлять'),('custom','Мука есть, яйца нет','Мука есть, яйца нет'),
                                       ('skip','','без дополнительных уточнений')):
            key=self.pause_for_ingredients();job=server.JOBS[key]
            with patch.object(server.threading.Thread,'start'):
                server.respond_clarification(key,{'clarification_id':job['clarification']['id'],'choice':choice,'answer':answer})
            self.assertIn(expected,job['clarification']['answer'])
            server.ACTIVE=None

    def test_pending_clarification_survives_restart_and_rejects_stale_reply(self):
        key=self.pause_for_ingredients();job=server.JOBS[key]
        with patch.object(server.sys,'argv',['server.py']),patch.object(server,'ThreadingHTTPServer') as http:
            http.return_value.serve_forever.side_effect=KeyboardInterrupt
            server.main()
        self.assertEqual(server.JOBS[key]['status'],'waiting_input')
        with self.assertRaises(ValueError):server.respond_clarification(key,{'clarification_id':'wrong','choice':'allow'})
        self.assertEqual(server.JOBS[key]['clarification']['status'],'waiting')
        self.assertIsNone(server.ACTIVE)

    def test_cancel_before_launch_does_not_start_process(self):
        with patch.object(server.threading.Thread,'start'):
            key = server.start_job({'query':'ужин','mode':'v1'})
        job=server.JOBS[key];job['_cancel']=True
        with patch.object(server.subprocess,'Popen') as proc:
            server.worker(job)
        proc.assert_not_called()
        self.assertEqual(job['status'],'cancelled')

if __name__=='__main__':unittest.main()
