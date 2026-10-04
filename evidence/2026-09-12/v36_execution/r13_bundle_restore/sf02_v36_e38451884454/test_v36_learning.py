import unittest
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from v32_features import word_tokens
from v36_learning import FrequencyTfidf, FactEncoder, fit_aggregated, PARAMETERS, risk_threshold, risk_metrics, repeat_weights


class Learning(unittest.TestCase):
    def test_repeat_weight_uses_feature_equality_not_input_ids(self):
        x=sparse.csr_matrix([[0,1],[0,1],[1,0],[0,0]])
        ids=np.array([0]*100+[1]*100+[2]*2+[3])
        w=repeat_weights(x,ids)
        self.assertAlmostEqual(float(w.mean()),1)
        self.assertTrue(np.all((w>=.2)&(w<=5)))
        self.assertEqual(w[0],w[100])
        self.assertGreater(w[-1],w[0])

    def test_frequency_tfidf_matches_real_expanded_rows(self):
        docs=['failure code bad credential','success code valid credential','network deny tcp / 443','']
        frequency=np.array([9,4,3,5])
        expanded=[doc for doc,n in zip(docs,frequency) for _ in range(n)]
        for limit in [100000,5]:
            reference=TfidfVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,
                ngram_range=(1,2),min_df=1,max_features=limit,dtype=np.float64).fit(expanded)
            candidate=FrequencyTfidf(max_features=limit).fit(docs,frequency)
            self.assertEqual(reference.get_feature_names_out().tolist(),candidate.names().tolist())
            np.testing.assert_allclose(reference.idf_,candidate.idf,atol=1e-12)
            np.testing.assert_allclose(reference.transform(docs).toarray(),candidate.transform(docs).toarray(),atol=1e-12)

    def test_unseen_evaluation_text_cannot_change_fit_vocab_or_idf(self):
        v=FrequencyTfidf().fit(['train failure','train success'],[2,3])
        before=v.idf.copy();names=v.names().tolist()
        x=v.transform(['previouslyunseenfield'])
        self.assertEqual(x.nnz,0)
        np.testing.assert_array_equal(v.idf,before)
        self.assertEqual(v.names().tolist(),names)

    def test_aggregated_loss_matches_expanded_logistic_predictions(self):
        rng=np.random.RandomState(12)
        x=sparse.csr_matrix(rng.normal(size=(15,6)))
        inverse=np.repeat(np.arange(15),np.arange(1,16))
        labels=np.arange(len(inverse))%3
        a=LogisticRegression(C=.1,solver='lbfgs',tol=PARAMETERS['tol'],max_iter=1000).fit(x[inverse],labels)
        b,info=fit_aggregated(x,inverse,labels,.1)
        np.testing.assert_allclose(a.predict_proba(x),b.predict_proba(x),atol=1e-8)
        self.assertEqual(info['sum_weights'],len(inverse))

    def test_numeric_zero_and_unobserved_are_distinct(self):
        f=FactEncoder().fit([{'src_port':0},{'src_port':443},{}])
        x=f.transform([{'src_port':0},{}])
        self.assertGreater((x[0]-x[1]).nnz,0)

    def test_fact_categories_fit_only(self):
        f=FactEncoder().fit([{'outcome':'failure','duration_seconds':1}])
        scale=f.scale.copy();names=f.names().tolist()
        f.transform([{'outcome':'novel_target','duration_seconds':99999}])
        np.testing.assert_array_equal(f.scale,scale)
        self.assertEqual(f.names().tolist(),names)
        self.assertFalse(any('novel_target' in x for x in names))

    def test_empty_facts_remain_empty_features(self):
        f=FactEncoder().fit([{}])
        self.assertEqual(f.transform([{},{}]).shape,(2,0))

    def test_threshold_ties_respect_budget(self):
        risk=np.array([.8]*20+[.1]*80,dtype=np.float64)
        t=risk_threshold(risk,.01)
        p=np.column_stack([1-risk,risk/2,risk/2])
        self.assertEqual(risk_metrics(np.zeros(100,dtype=int),p,t)['class_alert_counts'][0],0)


if __name__=='__main__':unittest.main(verbosity=2)
