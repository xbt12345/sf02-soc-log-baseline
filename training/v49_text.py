"""Fixed word + local-character feature union; only fit rows build vocabulary.

The unchanged word channel retains original tokens and operator relations.
The added channel shares local spelling fragments, not presumed attack meaning.
"""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from run_v38_train import text_encoder

CHARACTER_WEIGHT=0.5
MAX_FEATURES=10000

def readable_compounds(text):
    return text.casefold().replace('_',' ')

class WordCharacter:
    def fit(self,texts,ids,fit):
        import pandas as pd
        self.word=text_encoder(texts,ids,fit)
        codes,unique=pd.factorize(np.asarray(texts,dtype=object),sort=False)
        frequency=np.bincount(codes[ids[fit]],minlength=len(unique));take=frequency>0
        t=list(unique[take]);frequency=frequency[take]
        counter=CountVectorizer(analyzer='char_wb',ngram_range=(3,5),preprocessor=readable_compounds,lowercase=False,dtype=np.int64)
        x=counter.fit_transform(t);names=counter.get_feature_names_out()
        counts=np.asarray(x.T.dot(frequency)).ravel()
        chosen=np.sort(np.argsort(-counts,kind='stable')[:MAX_FEATURES])
        self.character=CountVectorizer(analyzer='char_wb',ngram_range=(3,5),preprocessor=readable_compounds,lowercase=False,
            vocabulary={str(s):i for i,s in enumerate(names[chosen])},dtype=np.float64)
        present=x[:,chosen].astype(float);present.data[:]=1
        df=np.asarray(present.T.dot(frequency)).ravel()
        self.idf=np.log((1+frequency.sum())/(1+df))+1
        self.fit_rows=int(frequency.sum())
        return self

    def transform(self,texts):
        word=self.word.transform(texts)
        char=self.character.transform(texts).multiply(self.idf).tocsr()
        normalize(char,norm='l2',copy=False)
        return sparse.hstack([word,char*CHARACTER_WEIGHT],format='csr')

    def names(self):
        return np.concatenate([self.word.names(),np.array(['char:'+s for s in self.character.get_feature_names_out()])])
