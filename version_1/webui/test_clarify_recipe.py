import json
import unittest
import clarify_recipe as c
from test_recipe_chat import Tokenizer

class ClarifyTests(unittest.TestCase):
    def test_ask_is_grounded_in_request_and_returned_as_question(self):
        data={'query':'Пирог из баранины и картофеля','results':[{'id':'a','title':'Пирог','text':'Тесто из муки и яйца.'}]}
        def request(url,payload,timeout):
            self.assertIn(data['query'],payload['messages'][1]['content'])
            return {'message':{'content':json.dumps({'action':'ask','question':'Можно добавить муку и яйцо?',
                                                    'missing_ingredients':['мука','яйцо']})}}
        result=c.clarify(data,Tokenizer(),request)
        self.assertEqual(result['action'],'ask')
        self.assertEqual(result['ingredients'],['мука','яйцо'])

    def test_question_requires_named_ingredients(self):
        for value in ({'action':'ask','question':'Можно?','missing_ingredients':[]},
                      {'action':'ask','question':'','missing_ingredients':['мука']},
                      {'action':'ask','question':'Можно?','missing_ingredients':['мука','мука']}):
            with self.assertRaises(ValueError):
                c.clarify({'query':'Пирог'},Tokenizer(),lambda *args:{'message':{'content':json.dumps(value)}})

if __name__=='__main__':unittest.main()
