import unittest
from score_reviews import aggregate

class ReviewTests(unittest.TestCase):
    def row(self, **changes):
        return dict(case_id='a',mode='rag',seed=42,reviewer='r',constraints=True,
                    consistency=False,executable=True,source_grounding=None,**changes)

    def test_failures_count_and_unknown_grounding_is_excluded(self):
        result=aggregate([self.row()])['modes']['rag']['metrics']
        self.assertEqual(result['consistency'],dict(passed=0,total=1,rate=0))
        self.assertEqual(result['source_grounding'],dict(passed=0,total=0,rate=None))

    def test_duplicate_and_empty_reviews_are_rejected(self):
        for rows in ([],[self.row(),self.row()]):
            with self.assertRaises(ValueError):
                aggregate(rows)

    def test_non_boolean_scores_are_rejected(self):
        row=self.row(); row['constraints']='yes'
        with self.assertRaises(ValueError):
            aggregate([row])
