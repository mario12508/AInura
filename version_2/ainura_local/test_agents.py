import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_recipe as a
from test_local import TinyTokenizer


def decision(action, query=None):
    return {'action':action,'reason':'Короткое обоснование.','search_query':query}


def selection(source='a', sufficient=True, suggested=None):
    return {'selected':[{'id':source,'why_fits':'Подходящая основа.',
                        'required_changes':[],'risks':[]}],
            'sufficient':sufficient,'suggested_search_query':suggested,'notes':[]}


def recipe(source='a', title='Рис'):
    return {'title':title,'servings':2,'ingredients':[{'name':'рис','amount':100,'unit':'г'}],
            'steps':['Сварить рис.'],'used_source_ids':[source],'notes':[]}


def critique(verdict='pass', severity=None):
    return {'verdict':verdict,'issues':([] if severity is None else [{
        'category':'ingredient_step','severity':severity,'message':'Нет нужного продукта.',
        'suggested_fix':'Исправить список.'}]),'summary':'Проверка завершена.'}


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs=list(outputs)
        self.requests=[]

    def __call__(self, url, payload, timeout):
        self.requests.append(payload)
        if not self.outputs:
            raise AssertionError('Unexpected additional LLM call')
        value=self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
        return {'message':{'content':content},'prompt_eval_count':100,'eval_count':20}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.d=Path(self.tmp.name)
        self.initial=self.d/'search.json'
        self.initial.write_text(json.dumps(self.search_result('ужин','a')),encoding='utf-8')
        self.unloads=0

    def tearDown(self):
        self.tmp.cleanup()

    def unload(self):
        self.unloads+=1

    def search_result(self, query, source):
        return {'query':query,'results':[{'id':source,'title':'Рис','text':'Рис готовят с овощами.',
                                          'book':'Книга','sources':[{'file':'book.fb2'}]}],
                'index':{'chunks_sha256':'corpus'},'settings':{'rerank':True}}

    def args(self, **changes):
        values=dict(search=str(self.initial),query=None,index=None,output=str(self.d/'agent.json'),
            model=a.MODEL,tokenizer=a.TOKENIZER,ollama_url='http://localhost:11434',device='cpu',
            local_only=True,max_llm_calls=18,max_repairs=2,max_searches=2,max_sources=2,
            candidates=10,retrieve_k=100,num_ctx=8192,num_predict=1536,temperature=0.2,
            seed=42,timeout=60,search_timeout=60)
        values.update(changes)
        return argparse.Namespace(**values)

    def run_script(self, outputs, args=None, **kwargs):
        model=ScriptedModel(outputs)
        with contextlib.redirect_stdout(io.StringIO()):
            result=a.run(args or self.args(),tokenizer=TinyTokenizer(),request_fn=model,
                         unload_fn=self.unload,requirements_fn=lambda query: {
                             'constraints': [], 'conflicts': [], 'questions': []},**kwargs)
        return result,model

    def happy(self, source='a'):
        return [decision('select'),selection(source),decision('adapt'),recipe(source),
                decision('critique'),critique(),decision('finish')]

    def test_critic_can_recover_after_one_truncated_response(self):
        class TruncateOnce(ScriptedModel):
            truncated=False
            def __call__(self, url, payload, timeout):
                response=super().__call__(url,payload,timeout)
                if payload['messages'][0]['content']==a.CRITIC and not self.truncated:
                    response['done_reason']='length'
                    self.truncated=True
                return response
        model=TruncateOnce([decision('select'),selection(),decision('adapt'),recipe(),
                            decision('critique'),critique(),decision('critique'),critique(),decision('finish')])
        with contextlib.redirect_stdout(io.StringIO()):
            result=a.run(self.args(),tokenizer=TinyTokenizer(),request_fn=model,unload_fn=self.unload,
                         requirements_fn=lambda query: {'constraints': [], 'conflicts': [], 'questions': []})
        self.assertTrue(result['accepted'])
        self.assertEqual(result['stats']['llm_calls'],9)
        self.assertNotIn('repeated_failure',result['state'])
        critic_calls=[c for c in result['llm_calls'] if c['role']=='critic']
        self.assertIn('лимита генерации',critic_calls[1]['request']['messages'][1]['content'])

    def test_repeated_truncated_critic_stops_after_one_retry(self):
        class TruncatingModel(ScriptedModel):
            def __call__(self, url, payload, timeout):
                response=super().__call__(url,payload,timeout)
                if payload['messages'][0]['content']==a.CRITIC:
                    response['done_reason']='length'
                return response
        model=TruncatingModel([decision('select'),selection(),decision('adapt'),recipe(),
                               decision('critique'),critique(),decision('critique'),critique()])
        with contextlib.redirect_stdout(io.StringIO()):
            result=a.run(self.args(),tokenizer=TinyTokenizer(),request_fn=model,unload_fn=self.unload,
                         requirements_fn=lambda query: {'constraints': [], 'conflicts': [], 'questions': []})
        self.assertEqual(result['status'],'stopped')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['stats']['llm_calls'],8)
        self.assertIn('дважды',result['reason'])
        self.assertEqual(result['state']['repeated_failure']['count'],2)
        self.assertEqual(self.unloads,1)

    def test_selector_reserves_id_schema_before_packing_context(self):
        ids=[str(i)+'x'*200 for i in range(10)]
        data=self.search_result('ужин',ids[0])
        data['results']=[{'id':source,'title':'Рис','text':'Рис с овощами. '*120}
                         for source in ids]
        self.initial.write_text(json.dumps(data),encoding='utf-8')
        result,_=self.run_script(self.happy(ids[0]),args=self.args(num_ctx=4096))
        self.assertTrue(result['accepted'])
        selector=next(c for c in result['llm_calls'] if c['role']=='selector')
        self.assertLessEqual(selector['estimated_prompt_tokens']+
                             selector['request']['options']['num_predict']+256,4096)
        self.assertTrue(next(e for e in result['events'] if e['action']=='select')['omitted_candidates'])

    def test_sufficient_selection_can_return_original_recipe(self):
        result,_=self.run_script([decision('select'),selection(),decision('stop')])
        self.assertEqual(result['status'],'source_selected')
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['recipe'])
        self.assertEqual(result['source_recipes'][0]['text'],'Рис готовят с овощами.')
        self.assertEqual(result['stats']['llm_calls'],3)

    def test_cli_default_budgets_allow_four_search_rounds(self):
        with patch.object(a,'run',return_value={'accepted':False,'status':'source_selected'}) as run:
            self.assertEqual(a.cli(['--search',str(self.initial)]),0)
        args=run.call_args.args[0]
        self.assertEqual(args.max_llm_calls,30)
        self.assertEqual(args.max_searches,4)
        self.assertEqual(args.max_repairs,2)

    def test_stop_reports_actual_remaining_budget_not_model_claim(self):
        stop={'action':'stop','reason':'Бюджет полностью исчерпан.','search_query':None}
        result,_=self.run_script([decision('select'),selection(sufficient=False),stop],
                                 args=self.args(max_llm_calls=30,max_searches=4))
        self.assertEqual(result['status'],'stopped')
        self.assertIn('LLM: 27',result['reason'])
        self.assertIn('раундов: 3',result['reason'])
        self.assertNotIn('полностью исчерпан',result['reason'])
        self.assertEqual(result['coordinator_reason'],'Бюджет полностью исчерпан.')

    def test_recipe_error_is_not_an_unsatisfiable_user_request(self):
        outputs=[decision('select'),selection(),decision('adapt'),recipe(),
                 decision('critique'),critique('unsatisfiable','major'),
                 decision('critique'),critique(),decision('finish')]
        result,_=self.run_script(outputs)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['llm_calls'][5]['status'],'error')
        self.assertIn('требуют revise',result['llm_calls'][5]['error'])

    def test_acceptance_requires_select_draft_and_current_critique(self):
        result,model=self.run_script(self.happy())
        self.assertTrue(result['accepted'])
        self.assertEqual(result['stats']['llm_calls'],7)
        self.assertEqual(result['stats']['calls_by_role'],{'requirements':0,'coordinator':4,'selector':1,'adapter':1,'critic':1})
        self.assertEqual(result['stats']['generated_tokens'],140)
        self.assertEqual(self.unloads,1)
        self.assertEqual(json.loads((self.d/'agent.json').read_text())['status'],'accepted')
        self.assertEqual(model.outputs,[])
        critic=next(c for c in result['llm_calls'] if c['role']=='critic')
        critic_payload=critic['request']['messages'][1]['content']
        self.assertNotIn('Рис готовят с овощами.',critic_payload)
        self.assertIn('source_references',critic_payload)

    def test_critic_can_trigger_repair_then_new_check(self):
        script=[decision('select'),selection(),decision('adapt'),recipe(),decision('critique'),
                critique('revise','major'),decision('adapt'),recipe(title='Исправленный рис'),
                decision('critique'),critique(),decision('finish')]
        result,_=self.run_script(script)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['state']['draft_count'],2)
        self.assertEqual(result['recipe']['title'],'Исправленный рис')
        self.assertEqual(result['stats']['llm_calls'],11)

    def test_coordinator_can_rewrite_search_query(self):
        index=self.d/'index';index.mkdir()
        (index/'manifest.json').write_text(json.dumps({'hashes':{'chunks.jsonl':'corpus'}}))
        calls=[]
        def search(query,number):
            calls.append((query,number))
            return self.search_result(query,'b')
        script=[decision('search','рис с овощами без молока')]+self.happy('b')
        result,model=self.run_script(script,args=self.args(index=str(index)),search_fn=search)
        self.assertTrue(result['accepted'])
        self.assertEqual(calls,[('рис с овощами без молока',2)])
        self.assertEqual(result['query'],'ужин')
        self.assertEqual(result['stats']['retrieval_calls'],1)
        selector_request=next(x for x in model.requests if x['messages'][0]['content']==a.SELECTOR)
        self.assertIn('"original_query": "ужин"',selector_request['messages'][1]['content'])

    def test_finish_before_critique_is_rejected(self):
        result,_=self.run_script([decision('finish'),decision('stop')])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['llm_calls'][0]['status'],'error')

    def test_pass_with_major_issue_cannot_be_accepted(self):
        script=[decision('select'),selection(),decision('adapt'),recipe(),decision('critique'),
                critique('pass','blocking'),decision('stop')]
        result,_=self.run_script(script)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['state']['critique'])
        self.assertTrue(any('major/blocking' in e.get('error','') for e in result['events']))

    def test_source_change_invalidates_old_draft_and_critique(self):
        state={'candidates':[{'id':'a'}],'selected':[{'id':'a'}],'draft_count':1,
               'draft':recipe(),'draft_version':1,'checked_version':1,
               'context_version':2,'draft_context_version':1,'critique':critique(),
               'search_rounds':1}
        actions=a.allowed_actions(state,self.args())
        self.assertNotIn('finish',actions)
        self.assertNotIn('critique',actions)
        self.assertIn('adapt',actions)

    def test_budget_counts_invalid_model_responses(self):
        result,model=self.run_script(['not json','still not json'],args=self.args(max_llm_calls=2))
        self.assertEqual(result['status'],'budget_exhausted')
        self.assertEqual(result['stats']['llm_calls'],2)
        self.assertFalse(result['accepted'])

    def test_draft_must_be_checked_before_repair_and_pass_cannot_be_rewritten(self):
        state={'candidates':[{'id':'a'}], 'selection':selection(),
               'selected':[{'id':'a'}], 'draft_count':1, 'draft':recipe(),
               'draft_version':1, 'checked_version':-1, 'context_version':1,
               'draft_context_version':1, 'critique':None, 'search_rounds':1}
        self.assertNotIn('adapt', a.allowed_actions(state,self.args()))
        self.assertIn('critique', a.allowed_actions(state,self.args()))
        state.update(checked_version=1, critique=critique('revise'))
        self.assertIn('adapt', a.allowed_actions(state,self.args()))
        state['critique']=critique()
        self.assertNotIn('adapt', a.allowed_actions(state,self.args()))
        self.assertIn('finish', a.allowed_actions(state,self.args()))
        self.assertEqual(a.allowed_actions(state,self.args(index='index')), ['finish'])
        for verdict in ('new_sources','unsatisfiable'):
            state['critique']=critique(verdict)
            self.assertNotIn('adapt', a.allowed_actions(state,self.args()))
        self.assertEqual(a.allowed_actions(state,self.args(index='index')), ['stop'])

    def test_coordinator_context_overflow_terminates_without_busy_loop(self):
        self.initial.write_text(json.dumps(self.search_result('x'*60000,'a')))
        result,model=self.run_script([])
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['stats']['llm_calls'],0)
        self.assertIn('контекст координатора',result['reason'])

    def test_duplicate_search_blocked_without_calling_tool(self):
        index=self.d/'index';index.mkdir()
        (index/'manifest.json').write_text(json.dumps({'hashes':{'chunks.jsonl':'corpus'}}))
        result,model=self.run_script([decision('search','ужин'),decision('stop')],
            args=self.args(index=str(index)),search_fn=lambda *args:self.fail('Duplicate search executed'))
        self.assertEqual(result['stats']['retrieval_calls'],0)
        self.assertTrue(result['state']['search_blocked'])
        self.assertNotIn('search',model.requests[1]['format']['properties']['action']['enum'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'],'stopped')

    def test_duplicate_coordinator_query_uses_fresh_selector_suggestion(self):
        index=self.d/'index';index.mkdir()
        (index/'manifest.json').write_text(json.dumps({'hashes':{'chunks.jsonl':'corpus'}}))
        calls=[]
        def search(query,number):
            calls.append((query,number))
            return self.search_result(query,'b')
        script=[decision('select'),selection(sufficient=False,suggested='рис с овощами'),
                decision('search','ужин'),decision('stop')]
        result,_=self.run_script(script,args=self.args(index=str(index)),search_fn=search)
        self.assertEqual(calls,[('рис с овощами',2)])
        self.assertEqual(result['status'],'stopped')
        search_event=next(e for e in result['events'] if e['action']=='search')
        self.assertEqual(search_event['requested_search_query'],'ужин')
        self.assertEqual(search_event['search_query'],'рис с овощами')
        self.assertNotEqual(result['status'],'budget_exhausted')

    def test_insufficient_selection_is_not_adapted_or_reselected(self):
        result,model=self.run_script([
            decision('select'),selection(sufficient=False),decision('stop')])
        self.assertEqual(result['state']['selected'],[])
        allowed=model.requests[2]['format']['properties']['action']['enum']
        self.assertEqual(allowed,['stop'])
        self.assertEqual(result['status'],'stopped')

    def test_unknown_source_id_from_selector_is_rejected(self):
        result,_=self.run_script([decision('select'),selection('unknown'),decision('stop')])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['state']['selected'],[])

    def test_previous_role_error_is_forwarded_to_the_next_role(self):
        result,model=self.run_script([decision('select'),selection('unknown')]+self.happy())
        self.assertTrue(result['accepted'])
        selectors=[p for p in model.requests if p['messages'][0]['content']==a.SELECTOR]
        self.assertIn('previous_error',selectors[1]['messages'][1]['content'])
        self.assertIn('invalid enum',selectors[1]['messages'][1]['content'])
        self.assertEqual(selectors[1]['format']['properties']['selected']['items']['properties']['id']['enum'], ['a'])

    def test_unknown_citation_from_adapter_is_rejected(self):
        result,_=self.run_script([decision('select'),selection(),decision('adapt'),recipe('unknown'),decision('stop')])
        self.assertIsNone(result['state']['draft'])
        self.assertFalse(result['accepted'])

    def test_network_error_saves_diagnostic_and_stops(self):
        result,_=self.run_script([RuntimeError('Ollama unavailable')])
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['stats']['llm_calls'],1)
        self.assertEqual(self.unloads,1)
        self.assertIn('Ollama unavailable',result['llm_calls'][0]['error'])

    def test_search_tool_unloads_qwen_before_pytorch_process(self):
        order=[]
        client=SimpleNamespace(unload=lambda:order.append('unload'))
        args=self.args(index=str(self.d/'index'))
        def process(command,**kwargs):
            order.append('search')
            output=Path(command[command.index('--output')+1])
            query=command[command.index('--query')+1]
            output.write_text(json.dumps(self.search_result(query,'b')))
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        tool=a.SearchTool(args,self.d,client)
        with patch.object(a.subprocess,'run',side_effect=process):
            result=tool('новый запрос',2)
        self.assertEqual(order,['unload','search'])
        self.assertEqual(result['query'],'новый запрос')


if __name__=='__main__':
    unittest.main()
