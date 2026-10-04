"""v3.7 unit-row-weight fitting and context-bound feature encoding."""
import json
import re
import numpy as np
from scipy import sparse
from scipy.stats import binom
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score,roc_auc_score
import v36_learning as control

PARAMETERS=dict(control.PARAMETERS)
PARAMETERS.update(row_weight=1,class_weight=None,repeat_weighting=False)
cm_metrics=control.cm_metrics
risk_threshold=control.risk_threshold
risk_metrics=control.risk_metrics
fit_aggregated=control.fit_aggregated
WORD=re.compile(r'(?u)\w+')
CONTEXT_OP=re.compile(r'(\w+)\s*(&&|\|\||>>|[=|<>])\s*(\w+)')

def word_tokens(text):
    words=WORD.findall(text.casefold())
    # Preserve operator relationships, never a standalone count such as op_equals.
    return words+['syntax:'+a+op+b for a,op,b in CONTEXT_OP.findall(text.casefold())]

class FrequencyTfidf:
    def __init__(self,max_features=100000):self.max_features=max_features
    def fit(self,texts,frequencies):
        frequencies=np.asarray(frequencies,dtype=np.float64)
        if len(texts)!=len(frequencies) or np.any(frequencies<=0):raise ValueError('Invalid fit multiplicities')
        cv=CountVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,ngram_range=(1,2),dtype=np.int64)
        try:x=cv.fit_transform(texts)
        except ValueError as e:
            if 'empty vocabulary' not in str(e):raise
            self.counter=None;self.idf=np.array([]);self.fit_rows=int(frequencies.sum());return self
        names=cv.get_feature_names_out();counts=np.asarray(x.T.dot(frequencies)).ravel()
        chosen=np.sort(np.argsort(-counts,kind='stable')[:self.max_features]) if len(names)>self.max_features else np.arange(len(names))
        selected=names[chosen]
        self.counter=CountVectorizer(tokenizer=word_tokens,token_pattern=None,lowercase=False,ngram_range=(1,2),
                                    vocabulary={str(v):i for i,v in enumerate(selected)},dtype=np.float64)
        presence=x[:,chosen].astype(float);presence.data[:]=1
        df=np.asarray(presence.T.dot(frequencies)).ravel()
        self.idf=np.log((1+frequencies.sum())/(1+df))+1;self.fit_rows=int(frequencies.sum())
        return self
    def transform(self,texts):
        if self.counter is None:return sparse.csr_matrix((len(texts),0),dtype=np.float64)
        x=self.counter.transform(texts).multiply(self.idf).tocsr();normalize(x,norm='l2',copy=False);return x
    def names(self):return self.counter.get_feature_names_out() if self.counter else np.array([],dtype=object)

def fact_dictionary(value):
    data=json.loads(value) if isinstance(value,str) else value
    result={}
    for key,value in data.items():
        if key in ('src_port','dst_port'):raise ValueError('Unprojected ordinal port')
        if isinstance(value,str):result['category:'+key]=value
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            if not np.isfinite(value) or value<0:raise ValueError('Invalid numeric fact')
            if key in ('http_status','icmp_type','icmp_code','logontype'):
                result['category:'+key]=str(value)
            else:result['numeric:'+key]=float(np.log1p(value))
    return result

class FactEncoder:
    def fit(self,values):
        self.vectorizer=DictVectorizer(sparse=True,dtype=np.float64)
        x=self.vectorizer.fit_transform([fact_dictionary(v) for v in values])
        self.scale=np.maximum(np.asarray(abs(x).max(axis=0).toarray()).ravel(),1) if x.shape[1] else np.array([])
        return self
    def transform(self,values):
        x=self.vectorizer.transform([fact_dictionary(v) for v in values])
        return x.multiply(1/self.scale).tocsr()
    def names(self):return self.vectorizer.get_feature_names_out()

def risk_ranking(y,p):
    y=np.asarray(y);positive=y!=0;risk=1-np.asarray(p)[:,0]
    if len(np.unique(positive))!=2:return {'supported':False}
    return {'supported':True,'average_precision':float(average_precision_score(positive,risk)),
            'roc_auc':float(roc_auc_score(positive,risk)),
            'positive_classes_observed':np.unique(y[positive]).tolist()}

def np_threshold(normal_scores,alpha=.001,delta=.05):
    a=np.sort(np.asarray(normal_scores,dtype=float));n=len(a)
    minimum=int(np.ceil(np.log(delta)/np.log1p(-alpha)))
    result={'alpha':alpha,'delta':delta,'normal_rows':n,'minimum_iid_normal_rows':minimum,
            'iid_assumed_not_verified':True,'new_domain_guaranteed':False,'formal_guarantee_accepted':False}
    if n<minimum:return dict(result,computable=False,reason='insufficient_normal_rows')
    # P(Binomial(n,1-alpha)>=k) <= delta; strict greater than order statistic.
    lo,hi=1,n
    while lo<hi:
        mid=(lo+hi)//2
        if binom.sf(mid-1,n,1-alpha)<=delta:hi=mid
        else:lo=mid+1
    return dict(result,computable=True,rank=lo,threshold=float(np.nextafter(a[lo-1],np.inf)),
                binomial_tail=float(binom.sf(lo-1,n,1-alpha)))

def gated_predictions(p,threshold):
    p=np.asarray(p);pred=np.zeros(len(p),dtype=np.uint8)
    alarm=1-p[:,0]>=threshold
    pred[alarm]=1+p[alarm,1:].argmax(1)
    return pred

def calibration_policies(y,p,products,groups,alpha=.001):
    y=np.asarray(y);normal=y==0;risk=1-p[:,0];products=np.asarray(products);groups=np.asarray(groups)
    source={}
    for name in sorted(set(products[normal])):
        m=normal&(products==name)
        source[str(name)]={'rows':int(m.sum()),'groups':int(len(np.unique(groups[m]))),
                          'threshold':risk_threshold(risk[m],alpha)}
    return {'alpha':alpha,'empirical_pooled':risk_threshold(risk[normal],alpha),
            'empirical_worst_source_global':max(s['threshold'] for s in source.values()),
            'calibration_sources':source,'np_iid_diagnostic':np_threshold(risk[normal],alpha),
            'formal_FPR_guarantee':False,'reason':'correlated logs and future source distribution not verified'}

