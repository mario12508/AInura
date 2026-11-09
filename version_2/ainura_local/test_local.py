import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import answer_recipe as answer
import evaluate_retrieval as ev
import retrieval as r


class FakeStage:
    active = 0
    events = []

    def __init__(self, model, kind, device='auto', revision=None, max_tokens=8192, local_only=False):
        assert FakeStage.active == 0, 'Two models loaded simultaneously'
        FakeStage.active += 1
        FakeStage.events.append(('load',kind))
        self.kind = kind
        self.info = {'model':model,'revision_resolved':'test_commit',
                     'revision_requested':revision,'max_tokens':max_tokens,'device':'cpu'}

    def predict(self, texts, batch_size=2, queries=None, progress=False):
        if self.kind == 'embedding':
            mapping = {'рис':[1,0], 'суп':[0,1], 'паштет':[0.8,0.6]}
            return np.array([mapping.get(text,[1,0]) for text in texts],dtype=np.float32)
        return np.array([{'рис':2,'суп':5,'паштет':0}[text] for text in texts],dtype=np.float32)

    def close(self):
        FakeStage.active -= 1
        FakeStage.events.append(('close',self.kind))


class TinyTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        # Deterministic token budget adapter for tests, NOT a model token estimate.
        return [0] * (sum(len(m['content']) for m in messages)//4 + 20)


def build_args(chunks,index):
    return argparse.Namespace(chunks=str(chunks),index=str(index),model=r.EMBED_MODEL,
                              revision=None,device='cpu',batch_size=2,max_tokens=8192,
                              force=False,local_only=True)


def search_args(index,output,no_rerank=False):
    return argparse.Namespace(index=str(index),query='ужин',device='cpu',local_only=True,
        retrieve_k=3,top_k=3,reranker=r.RERANK_MODEL,reranker_revision=None,max_tokens=8192,
        rerank_batch_size=2,no_rerank=no_rerank,output=str(output))


def gen_args(output,search=None,mode='rag'):
    return argparse.Namespace(search=str(search) if search else None,mode=mode,
        query=None if mode=='rag' else 'ужин',model=answer.MODEL,tokenizer=answer.TOKENIZER,
        num_ctx=8192,num_predict=1536,context_recipes=3,output=str(output),temperature=0.2,
        seed=42,local_only=True,ollama_url='http://localhost:11434',timeout=60)


def recipe(source_ids):
    return {'title':'Блюдо','servings':2,'ingredients':[{'name':'Рис','amount':100,'unit':'г'}],
            'steps':['Сварить рис.'],'used_source_ids':source_ids,'notes':[]}


class LocalTests(unittest.TestCase):
    def test_duplicate_results_cannot_inflate_retrieval_scores(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ev.metrics(['a','a'], {'a'}, 2)
        for relevant, k in ((set(), 1), ({'a'}, 0)):
            with self.assertRaises(ValueError):
                ev.metrics(['a'], relevant, k)

    def test_measure_word_is_not_a_product_name(self):
        value = {'title':'Рис', 'servings':2,
                 'ingredients':[{'name':'рис','amount':100,'unit':'г'}],
                 'steps':['Сварить рис.'], 'used_source_ids':[], 'notes':[]}
        answer.validate_recipe(value, set())
        for name in ('ст. л.', 'стакан', ' Ч. Л. ', 'кг'):
            value['ingredients'][0]['name'] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'вместо продукта'):
                answer.validate_recipe(value, set())

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.d=Path(self.tmp.name)
        self.chunks=self.d/'chunks.jsonl'
        rows=[{'id':str(i),'title':name,'text':name,'book':'Книга',
               'sources':[{'file':'book.fb2'}]} for i,name in enumerate(['рис','суп','паштет'])]
        self.chunks.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows),encoding='utf-8')
        self.index=self.d/'index'
        FakeStage.active=0;FakeStage.events=[]
        with contextlib.redirect_stdout(io.StringIO()):
            r.build(build_args(self.chunks,self.index),stage_cls=FakeStage)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_roundtrip_alignment_and_cosine(self):
        index=r.DenseIndex(self.index)
        found=index.retrieve(np.array([1,0]),3)
        self.assertEqual([x['text'] for x in found],['рис','паштет','суп'])
        self.assertAlmostEqual(found[1]['dense_score'],0.8,places=6)
        self.assertEqual(found[0]['sources'],[{'file':'book.fb2'}])

    def test_corruption_and_duplicate_ids_rejected(self):
        with (self.index/'chunks.jsonl').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'changed or corrupted'):
            r.DenseIndex(self.index)
        self.chunks.write_text('{"id":"a","text":"x"}\n{"id":"a","text":"y"}')
        with self.assertRaisesRegex(ValueError,'duplicate id'):
            r.load_chunks(self.chunks)

    def test_reranker_reorders_full_records_sequentially(self):
        FakeStage.events=[]
        with contextlib.redirect_stdout(io.StringIO()):
            value=r.search(search_args(self.index,self.d/'search.json'),stage_cls=FakeStage)
        self.assertEqual([x['text'] for x in value['results']],['суп','рис','паштет'])
        self.assertEqual([x['dense_rank'] for x in value['results']],[3,1,2])
        self.assertEqual(FakeStage.events,[('load','embedding'),('close','embedding'),
                                           ('load','reranker'),('close','reranker')])
        self.assertEqual(FakeStage.active,0)
        saved=json.loads((self.d/'search.json').read_text())
        self.assertEqual(len(saved['dense_candidates']),3)

    def test_dense_only_does_not_load_reranker(self):
        FakeStage.events=[]
        with contextlib.redirect_stdout(io.StringIO()):
            value=r.search(search_args(self.index,self.d/'dense.json',True),stage_cls=FakeStage)
        self.assertEqual(value['results'][0]['text'],'рис')
        self.assertEqual(FakeStage.events,[('load','embedding'),('close','embedding')])

    def test_batch_reduction_restores_original_order(self):
        class OOM(Exception):pass
        stage=r.HFStage.__new__(r.HFStage)
        stage.device='cpu';stage.kind='embedding';stage.info={}
        stage.torch=SimpleNamespace(OutOfMemoryError=OOM,cuda=SimpleNamespace(empty_cache=lambda:None))
        stage.tokenize=lambda texts,queries=None:[{'input_ids':[i]*n} for i,n in enumerate([5,1,3,2])]
        def forward(rows):
            if len(rows)>2:raise OOM('test')
            return np.array([[x['input_ids'][0],1] for x in rows])
        stage._forward=forward
        with contextlib.redirect_stderr(io.StringIO()):
            result=stage.predict(['a','b','c','d'],batch_size=4)
        self.assertEqual(result[:,0].tolist(),[0,1,2,3])
        self.assertEqual(stage.info['effective_batch_size'],2)

    def test_baseline_and_rag_save_response_and_sources(self):
        with contextlib.redirect_stdout(io.StringIO()):
            r.search(search_args(self.index,self.d/'search.json'),stage_cls=FakeStage)
            rag=answer.generate(gen_args(self.d/'rag.json',self.d/'search.json'),TinyTokenizer(),
                request_fn=lambda *args:{'message':{'content':json.dumps(recipe(['1']))},'eval_count':100})
            base=answer.generate(gen_args(self.d/'base.json',mode='baseline'),TinyTokenizer(),
                request_fn=lambda *args:{'message':{'content':json.dumps(recipe([]))},'eval_count':100})
        self.assertEqual(len(rag['selected_sources']),3)
        self.assertEqual(base['selected_sources'],[])
        self.assertEqual(rag['request']['options'],base['request']['options'])
        self.assertEqual(rag['request']['keep_alive'],0)
        self.assertEqual(rag['request']['model'],base['request']['model'])

    def test_unknown_citation_saves_raw_failure(self):
        out=self.d/'failure.json'
        with self.assertRaises(RuntimeError):
            answer.generate(gen_args(out,mode='baseline'),TinyTokenizer(),
              request_fn=lambda *args:{'message':{'content':json.dumps(recipe(['unknown']))}})
        saved=json.loads(out.read_text())
        self.assertEqual(saved['status'],'error')
        self.assertIn('ollama_response',saved)

    def test_no_chunk_truncation_context_skips_whole_recipe(self):
        candidate={'id':'long','title':'Блюдо','text':'x'*50000}
        with self.assertRaisesRegex(ValueError,'Ни один целый рецепт'):
            answer.select_context('ужин',[candidate],TinyTokenizer(),8192,1536,3)
        self.assertEqual(len(candidate['text']),50000)

    def test_query_mismatch_prevents_generation(self):
        p=self.d/'search.json';p.write_text(json.dumps({'query':'one','results':[{'id':'1','text':'x'}]}))
        args=gen_args(self.d/'answer.json',p);args.query='two'
        with self.assertRaisesRegex(ValueError,'Запрос отличается'):
            answer.generate(args,TinyTokenizer(),request_fn=lambda *args:self.fail('Should not call LLM'))

    def test_retrieval_metrics_and_saved_evaluation(self):
        actual=ev.metrics(['a','b','c'],{'b','c'},3)
        self.assertEqual(actual['recall@3'],1)
        self.assertEqual(actual['mrr@3'],0.5)
        expected=(1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3))
        self.assertAlmostEqual(actual['ndcg@3'],expected)
        with contextlib.redirect_stdout(io.StringIO()):
            r.search(search_args(self.index,self.d/'search.json'),stage_cls=FakeStage)
        labels=self.d/'labels.jsonl';labels.write_text(json.dumps({'query':'ужин','relevant_ids':['1']}))
        result=ev.evaluate(labels,[str(self.d/'search.json')],self.index,[1,3])
        self.assertEqual(result['averages']['dense']['metrics']['mrr@1'],0)
        self.assertEqual(result['averages']['reranked']['metrics']['mrr@1'],1)

    def test_unlabeled_queries_not_counted_as_failures(self):
        labels=self.d/'labels.jsonl';labels.write_text(json.dumps({'query':'ужин','relevant_ids':[]}))
        with self.assertRaisesRegex(ValueError,'No labeled queries'):
            ev.evaluate(labels,[],self.index,[1])


if __name__=='__main__':
    unittest.main()
