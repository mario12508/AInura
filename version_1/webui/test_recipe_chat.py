import json
import unittest
import recipe_chat as chat

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return [0]*(sum(len(m['content']) for m in messages)//4+20)

class ChatTests(unittest.TestCase):
    def setUp(self):
        self.result={'query':'Мясо без жарки','status':'source_selected',
                     'source_recipes':[{'id':'a','title':'Мясо в духовке','text':'Запекать мясо с чесноком.'}]}

    def test_question_uses_recipe_and_previous_answer(self):
        turns=[dict(id='1',question='Можно без чеснока?',answer='Да, можно.',status='done'),
               dict(id='2',question='А чем его заменить?',status='running')]
        def request(url,payload,timeout):
            messages=payload['messages']
            self.assertIn('Запекать мясо с чесноком.',messages[1]['content'])
            self.assertEqual(messages[-3:],[
                {'role':'user','content':'Можно без чеснока?'},
                {'role':'assistant','content':'Да, можно.'},
                {'role':'user','content':'А чем его заменить?'}])
            return {'message':{'content':json.dumps({'answer':'Используйте розмарин.'})}}
        turn=chat.answer(self.result,turns,'2',Tokenizer(),request)
        self.assertEqual(turn['answer'],'Используйте розмарин.')
        self.assertEqual(turn['status'],'done')

    def test_history_prunes_whole_turns_and_keeps_current_recipe(self):
        turns=[dict(id='1',question='x'*30000,answer='y'*30000,status='done'),
               dict(id='2',question='Сколько порций?',status='running')]
        def request(url,payload,timeout):
            self.assertEqual(len(payload['messages']),3)
            self.assertIn('Запекать мясо',payload['messages'][1]['content'])
            return {'message':{'content':'{"answer":"Порции в книге не указаны."}'}}
        chat.answer(self.result,turns,'2',Tokenizer(),request)
        self.assertEqual(turns[-1]['omitted_turn_ids'],['1'])

    def test_overflow_and_invalid_response_do_not_succeed(self):
        turns=[dict(id='1',question='Вопрос',status='running')]
        bad={'query':'x','source_recipes':[{'id':'a','text':'x'*50000}]}
        with self.assertRaisesRegex(ValueError,'не помещаются'):
            chat.answer(bad,turns,'1',Tokenizer(),lambda *args:self.fail('No model call expected'))
        with self.assertRaises(ValueError):
            chat.answer(self.result,turns,'1',Tokenizer(),lambda *args:{'message':{'content':'{"answer":""}'}})
        self.assertNotEqual(turns[-1]['status'],'done')

    def test_generated_recipe_takes_precedence_over_unadapted_book(self):
        result={**self.result,'recipe':{'title':'Без чеснока','ingredients':[],'steps':['Запечь.']}}
        context=chat.recipe_context(result)
        self.assertEqual(context['recipe']['title'],'Без чеснока')
        self.assertNotIn('book_recipes',context)
        self.assertNotIn('text',context['sources'][0])

    def test_agents_are_optional_and_explicitly_routed(self):
        turn=dict(id='1',question='Найди другой рецепт без жарки',status='running',allow_agents=True)
        def request(url,payload,timeout):
            self.assertEqual(payload['format'],chat.ROUTE_SCHEMA)
            self.assertIn('Мясо без жарки',payload['messages'][1]['content'])
            return {'message':{'content':json.dumps({'action':'agent','answer':'',
                       'agent_query':'Другой рецепт из мяса без жарки'})}}
        chat.answer(self.result,[turn],'1',Tokenizer(),request)
        self.assertEqual(turn['status'],'agent_requested')
        self.assertEqual(turn['agent_query'],'Другой рецепт из мяса без жарки')
        self.assertIn('router_request',turn)
        turn=dict(id='1',question='Найди другой',status='running',allow_agents=False)
        with self.assertRaises(ValueError):
            chat.answer(self.result,[turn],'1',Tokenizer(),lambda *args:{'message':{'content':json.dumps({
                'action':'agent','answer':'','agent_query':'Другой рецепт'})}})
        self.assertNotIn('agent_query',turn)

    def test_agents_allowed_but_simple_question_can_be_answered_directly(self):
        turn=dict(id='1',question='Можно без чеснока?',status='running',allow_agents=True)
        calls=[]
        def request(url,payload,timeout):
            calls.append(payload)
            if payload['format']==chat.ROUTE_SCHEMA:
                self.assertNotIn('Запекать мясо с чесноком.',payload['messages'][1]['content'])
                reply={'action':'answer','answer':'','agent_query':None}
            else:
                self.assertIn('Запекать мясо с чесноком.',payload['messages'][1]['content'])
                reply={'answer':'Да.'}
            return {'message':{'content':json.dumps(reply)}}
        chat.answer(self.result,[turn],'1',Tokenizer(),request)
        self.assertEqual(len(calls),2)
        self.assertEqual(turn['status'],'done')
        self.assertEqual(turn['answer'],'Да.')
        self.assertNotIn('agent_query',turn)

    def test_agent_result_used_without_recursive_routing(self):
        turn=dict(id='1',question='Другой рецепт?',status='running',allow_agents=True,agent_used=True)
        new={'query':'Курица без жарки','status':'accepted','accepted':True,
             'recipe':{'title':'Курица на пару','steps':['Готовить на пару.']}}
        def request(url,payload,timeout):
            self.assertEqual(payload['format'],chat.SCHEMA)
            self.assertIn('Курица на пару',payload['messages'][2]['content'])
            return {'message':{'content':'{"answer":"Подобран рецепт курицы на пару."}'}}
        chat.answer(self.result,[turn],'1',Tokenizer(),request,agent_result=new)
        self.assertEqual(turn['status'],'done')

    def test_clarification_switch_changes_chat_instruction(self):
        for enabled in (True,False):
            turn={'id':'1','question':'Как приготовить?','status':'running','allow_clarifications':enabled}
            def request(url,payload,timeout):
                system=payload['messages'][0]['content']
                self.assertIn('Можно задать один короткий уточняющий вопрос' if enabled else 'Уточняющие вопросы отключены',system)
                return {'message':{'content':'{"answer":"Ответ по рецепту."}'}}
            chat.answer(self.result,[turn],'1',Tokenizer(),request)
            self.assertEqual(turn['status'],'done')

    def test_empty_result_rejected(self):
        with self.assertRaises(ValueError):chat.recipe_context({})

if __name__=='__main__':unittest.main()
