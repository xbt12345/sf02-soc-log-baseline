"""Count-preserving TF-IDF and aggregate repeated logistic losses, Python 3.8+."""
import json
import warnings

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from v32_features import word_tokens

PARAMETERS = {'C':[0.1,1.0], 'tie_macro_f1':0.001, 'solver':'lbfgs', 'tol':1e-6,
              'max_iter':1000, 'class_weight':None, 'row_weight':1,
              'max_features':100000, 'ngram_range':[1,2], 'dtype':'float64',
              'optimizer_note':'Same regularized multinomial logistic objective; all three views use LBFGS and exact multiplicities. Historical SAGA scores are not directly reused.'}


class FrequencyTfidf:
    def __init__(self, max_features=100000):
        self.max_features = max_features

    def fit(self, texts, frequencies):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if len(texts) != len(frequencies) or np.any(frequencies<=0):
            raise ValueError('Positive allowed-fit row multiplicities required')
        counter = CountVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,
                                  ngram_range=(1,2),min_df=1,dtype=np.int64)
        x = counter.fit_transform(texts)
        names = counter.get_feature_names_out()
        frequency = np.asarray(x.T.dot(frequencies)).ravel()
        if len(names)>self.max_features:
            chosen = np.sort(np.argsort(-frequency)[:self.max_features])
        else:
            chosen = np.arange(len(names))
        chosen_names = names[chosen]
        self.counter = CountVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,
                 ngram_range=(1,2),vocabulary={str(v):i for i,v in enumerate(chosen_names)},dtype=np.float64)
        presence = x[:,chosen].astype(np.float64)
        presence.data[:] = 1
        df = np.asarray(presence.T.dot(frequencies)).ravel()
        self.idf = np.log((1+frequencies.sum())/(1+df))+1
        self.fit_rows = int(frequencies.sum())
        return self

    def transform(self, texts):
        x = self.counter.transform(texts).astype(np.float64)
        x = x.multiply(self.idf).tocsr()
        normalize(x,norm='l2',copy=False)
        return x

    def names(self):
        return self.counter.get_feature_names_out()


def fact_dictionary(value):
    data = json.loads(value) if isinstance(value,str) else value
    result = {}
    for k,v in data.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):
            if not np.isfinite(v) or v<0:
                raise ValueError('Invalid observed numeric fact')
            result['numeric:'+k] = float(np.log1p(v))
            result['observed:'+k] = 1.0
        elif isinstance(v,str):
            result['category:'+k] = v
    return result


class FactEncoder:
    def fit(self, values):
        self.vectorizer = DictVectorizer(sparse=True,dtype=np.float64)
        x = self.vectorizer.fit_transform([fact_dictionary(v) for v in values])
        self.scale = np.maximum(np.asarray(abs(x).max(axis=0).toarray()).ravel(),1) if x.shape[1] else np.array([])
        return self

    def transform(self, values):
        x=self.vectorizer.transform([fact_dictionary(v) for v in values])
        return x.multiply(1/self.scale).tocsr()

    def names(self):
        return self.vectorizer.get_feature_names_out()


def fit_aggregated(x_unique, inverse, labels, C, row_weights=None):
    inverse=np.asarray(inverse,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    if set(np.unique(labels))!={0,1,2}:
        raise ValueError('All three fit classes required')
    weights = np.ones(len(labels),dtype=np.float64) if row_weights is None else np.asarray(row_weights,dtype=np.float64)
    if len(weights)!=len(labels) or np.any(weights<=0):
        raise ValueError('Invalid training weights')
    grouped=np.bincount(inverse*3+labels,weights=weights,minlength=x_unique.shape[0]*3).reshape(-1,3)
    feature_row,target=np.nonzero(grouped)
    sample_weight=grouped[feature_row,target]
    model=LogisticRegression(C=C,solver='lbfgs',tol=PARAMETERS['tol'],max_iter=PARAMETERS['max_iter'],class_weight=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(x_unique[feature_row],target,sample_weight=sample_weight)
    warning_text=[str(w.message) for w in caught]
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    if not converged:
        raise RuntimeError('Logistic fit did not converge: '+repr(warning_text))
    assert model.classes_.tolist()==[0,1,2]
    return model,{'fit_rows':len(labels),'aggregated_rows':len(target),'sum_weights':float(weights.sum()),
                  'iterations':model.n_iter_.tolist(),'warnings':warning_text,'converged':True}


def cm_metrics(labels, predictions, weights=None):
    labels=np.asarray(labels,dtype=int);predictions=np.asarray(predictions,dtype=int)
    cm=np.bincount(labels*3+predictions,weights=weights,minlength=9).reshape(3,3)
    support=cm.sum(1);predcount=cm.sum(0)
    recall=np.divide(cm.diagonal(),support,out=np.zeros(3),where=support>0)
    precision=np.divide(cm.diagonal(),predcount,out=np.zeros(3),where=predcount>0)
    f1=np.divide(2*cm.diagonal(),support+predcount,out=np.zeros(3),where=(support+predcount)>0)
    return {'rows':len(labels),'macro_f1':float(f1.mean()) if (support>0).all() else None,
            'confusion_matrix':cm.tolist(),'class_support':support.tolist(),
            'class_recall':[float(recall[i]) if support[i] else None for i in range(3)],
            'class_precision':[float(precision[i]) if predcount[i] else None for i in range(3)],
            'class_f1':[float(f1[i]) if support[i] else None for i in range(3)]}


def risk_threshold(risk,budget):
    if not len(risk):return None
    descending=np.sort(np.asarray(risk,dtype=np.float64))[::-1]
    allowed=int(np.floor(len(risk)*budget))
    return float(np.nextafter(descending[min(allowed,len(risk)-1)],np.inf))


def risk_metrics(y,p,threshold):
    if threshold is None:return {'supported':False}
    y=np.asarray(y);alarm=(1-np.asarray(p,dtype=np.float64)[:,0])>=threshold
    return {'supported':True,'threshold':threshold,'class_support':np.bincount(y,minlength=3).tolist(),
            'class_alert_counts':[int(alarm[y==c].sum()) for c in range(3)],
            'class_alert_rates':[float(alarm[y==c].mean()) if np.any(y==c) else None for c in range(3)]}
