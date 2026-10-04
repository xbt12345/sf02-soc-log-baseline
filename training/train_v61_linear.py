"""A/C: fixed original-row CE logistic controls; no outer prediction in this entry."""
import argparse
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from v61_common import FIELDS, fit_vocab, encode_facts, save, sha
from v61_runtime import load_data, score, predictions, source_receipt


def features(df,ctx,bundle=None):
    fit=df.role.eq('fit').to_numpy()
    facts=df[FIELDS].astype(str).to_numpy()
    if bundle is None:
        vocab=fit_vocab(facts[fit]);encoded=encode_facts(facts,vocab)
        word=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,dtype=np.float32)
        char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32)
        word.fit(df.text[fit]);char.fit(df.text[fit])
        onehot=OneHotEncoder(categories=[np.arange(len(v)) for v in vocab],handle_unknown='ignore',dtype=np.float32)
        onehot.fit(encoded[fit]);scale=StandardScaler().fit(ctx['stats'][fit])
        bundle={'vocab':vocab,'word':word,'char':char,'onehot':onehot,'scale':scale}
    else:encoded=encode_facts(facts,bundle['vocab'])
    x=sparse.hstack([bundle['word'].transform(df.text),bundle['char'].transform(df.text),bundle['onehot'].transform(encoded)],format='csr')
    xc=sparse.hstack([x,sparse.csr_matrix(bundle['scale'].transform(ctx['stats']))],format='csr')
    return x,xc,bundle


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=20260915)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    df,ctx,config=load_data(a.data);fit=np.flatnonzero(df.role.eq('fit'));dev=np.flatnonzero(df.role.eq('selection'))
    print('Fitting text and categorical transforms on fit membership only.',flush=True)
    x,xc,bundle=features(df,ctx)
    for arm,xx in [('A',x),('C',xc)]:
        target=a.out/(arm+'_'+str(a.seed));assert not target.exists();target.mkdir()
        params={'arm':arm,'seed':a.seed,'C':1.0,'max_iter':1000,'tol':1e-4,'solver':'lbfgs',
                'loss':'unweighted original-row multinomial cross entropy','class_weight':None,
                'word_ngram':[1,2],'char_ngram':[3,5],'fit_rows':len(fit),'selection_rows':len(dev),
                'data_configuration_sha256':sha(a.data/'configuration.json'),
                'source_sha256':source_receipt(['train_v61_linear.py','v61_common.py','v61_runtime.py'])}
        save(target/'preregistered.json',params);t=time.monotonic()
        model=LogisticRegression(C=1,max_iter=1000,tol=1e-4,solver='lbfgs',random_state=a.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(xx[fit],df.label.iloc[fit])
        assert model.classes_.tolist()==[0,1,2]
        probs=model.predict_proba(xx[dev]);fprobs=model.predict_proba(xx[fit])
        joblib.dump({'model':model,'features':bundle,'arm':arm},target/'model.joblib',compress=3)
        predictions(target/'selection.parquet',df,dev,probs)
        save(target/'selection.json',{'arm':arm,'seed':a.seed,'status':'inner_selection_only',
            'fit':score(df.label.iloc[fit],fprobs),'selection':score(df.label.iloc[dev],probs),
            'elapsed_seconds':time.monotonic()-t,'convergence_warnings':[str(w.message) for w in caught],
            'iterations':model.n_iter_.tolist(),'features':xx.shape[1],
            'model_sha256':sha(target/'model.joblib'),'predictions_sha256':sha(target/'selection.parquet')})
        print(arm+' fitted and inner selection saved; outer predictions remain unopened.',flush=True)


if __name__=='__main__':main()
