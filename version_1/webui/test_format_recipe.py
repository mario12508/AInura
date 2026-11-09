import copy
import json
import re
import unittest
import format_recipe as f
from test_recipe_chat import Tokenizer

class FormatTests(unittest.TestCase):
    def setUp(self):
        self.result={'query':'Ужин','source_recipes':[{'id':'a','title':'Мясо','text':'Мясо, 500 г.\nЗапечь 40 минут. Подавать горячим.'}]}
        self.docs=f.documents(self.result)
        self.layout={'documents':[{'id':'a','sections':[{'heading':'Приготовление',
                       'block_ids':list(range(len(self.docs[0]['blocks'])))}]}]}

    def test_source_text_preserved_and_model_only_selects_layout(self):
        original=copy.deepcopy(self.result)
        record=f.format_result(self.result,Tokenizer(),lambda *args:{'message':{'content':json.dumps(self.layout)}})
        self.assertEqual(record['status'],'formatted')
        text=' '.join(b['text'] for section in record['documents'][0]['sections'] for b in section['blocks'])
        normalize=lambda text:re.sub(r'\s+',' ',text).strip()
        self.assertEqual(normalize(text),normalize(self.result['source_recipes'][0]['text']))
        self.assertEqual(self.result,original)

    def test_missing_reordered_duplicated_and_invented_text_rejected(self):
        for ids in ([0],[1,0,2],[0,0,1,2],[0,1,2,100]):
            layout=copy.deepcopy(self.layout);layout['documents'][0]['sections'][0]['block_ids']=ids
            with self.subTest(ids=ids),self.assertRaises(ValueError):f.validate_layout(layout,self.docs)
        layout=copy.deepcopy(self.layout);layout['documents'][0]['sections'][0]['text']='Мясо, 900 г'
        with self.assertRaises(ValueError):f.validate_layout(layout,self.docs)

    def test_invalid_output_falls_back_without_losing_text(self):
        record=f.format_result(self.result,Tokenizer(),lambda *args:{'message':{'content':'{"documents":[]}'}})
        self.assertEqual(record['status'],'fallback')
        self.assertEqual(record['documents'],f.fallback(self.docs))
        self.assertEqual(record['llm_calls'],1)

    def test_source_heading_is_displayed_once_and_body_is_unchanged(self):
        result=copy.deepcopy(self.result)
        result['source_recipes'][0]['text']='Мясо\n'+result['source_recipes'][0]['text']
        formatted=f.fallback(f.documents(result))[0]
        body=' '.join(b['text'] for s in formatted['sections'] for b in s['blocks'])
        self.assertEqual(formatted['title'],'Мясо')
        self.assertFalse(body.startswith('Мясо Мясо'))
        normalize=lambda text:re.sub(r'\s+',' ',text).strip()
        self.assertEqual(normalize(formatted['title']+' '+body),normalize(result['source_recipes'][0]['text']))

    def test_structured_values_and_step_numbers_preserved(self):
        result={'recipe':{'title':'Рис','servings':2,'ingredients':[{'name':'рис','amount':100,'unit':'г'}],
                          'steps':['Варить 20 минут.','Подать.'],'notes':['Без молока.']}}
        docs=f.documents(result);formatted=f.fallback(docs)
        self.assertEqual(formatted[0]['servings'],2)
        self.assertEqual(formatted[0]['sections'][0]['blocks'][0]['text'],'рис: 100 г')
        steps=formatted[0]['sections'][1]['blocks']
        self.assertEqual([b['number'] for b in steps],[1,2])
        self.assertEqual([b['text'] for b in steps],result['recipe']['steps'])

if __name__=='__main__':unittest.main()
