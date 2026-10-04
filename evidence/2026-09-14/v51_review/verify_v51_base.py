"""Compare learned base state; explicitly distinguish sklearn process-ID cache."""
import argparse
import sys
from pathlib import Path
import joblib

def main(root,out):
    sys.path.insert(0,str(out/'runtime'))
    from run_v51 import reference
    from run_v48 import save,sha
    checks=[]
    for name,fold in [('pressure',None),('fold_0',0),('fold_1',1),('fold_2',2)]:
        new=joblib.load(out/name/'model.joblib')['base'];old=joblib.load(reference(root,fold)/'model.joblib')
        equal={k:joblib.hash(new[k])==joblib.hash(old[k]) for k in ['fact_encoder','model']};assert all(equal.values())
        ta=vars(new['text_encoder']);tb=vars(old['text_encoder']);assert set(ta)==set(tb)
        for k in ta:
            if k!='counter':assert joblib.hash(ta[k])==joblib.hash(tb[k]),k
        a=vars(ta['counter']);b=vars(tb['counter']);assert set(a)==set(b)
        differences=[]
        for k in a:
            if joblib.hash(a[k])!=joblib.hash(b[k]):
                assert k=='_stop_words_id',k
                assert ta['counter'].stop_words is None and tb['counter'].stop_words is None
                differences.append({'field':k,'new':a[k],'old':b[k]})
        checks.append({'model':name,'learned_state_equal':True,'runtime_cache_differences':differences})
    source=root/'.venv/Lib/site-packages/sklearn/feature_extraction/text.py'
    text=source.read_text(encoding='utf-8');assert 'self._stop_words_id = id(self.stop_words)' in text
    evidence=root/'evidence/2026-09-14/v51_review'
    save(evidence/'base_identity.json',{'all_learned_states_equal':True,'models':checks,'source_sha256':sha(Path(__file__)),
        'sklearn_source_sha256':sha(source),'runtime_cache_explanation':'_stop_words_id caches id(None) for stop-word consistency checks (source lines 397,409,422); this can change between processes without changing vocabulary, IDF, parameters or outputs. No stored models were modified.'})
    (evidence/'verify_v51_base.py').write_bytes(Path(__file__).read_bytes())
    print('All four learned base states match references.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.out.resolve())
