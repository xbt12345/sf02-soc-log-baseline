"""Paired round summary and fit-side portable-evidence diagnostic. Zero fits."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

def main(root,out):
    if out.exists():raise FileExistsError(out)
    runtime=root/'artifacts/v49_word_character_20260914/runtime';sys.path.insert(0,str(runtime))
    from run_v48 import read,save,sha
    previous=root/'artifacts/v48_information_repair_r2_20260914'
    rows=pd.read_parquet(previous/'rows.parquet');p=pd.read_parquet(previous/'projections.parquet')
    fit=rows.inner_role!=2
    facts={int(i):json.loads(s) for i,s in p.facts.items()}
    invalid=[i for i,f in facts.items() if f.get('credential_check')=='invalid' and f.get('auth_result')=='failure']
    selected=rows[fit&rows.projection_id.isin(invalid)]
    assert len(selected)==314
    selected=selected.sort_values('row_position').reset_index(drop=True)
    active,ids=np.unique(selected.projection_id.to_numpy(),return_inverse=True)
    texts=p.iloc[active].text.tolist();original=[facts[int(i)] for i in active]
    portable_keys={'auth_result','credential_check','outcome'}
    reduced=[{k:v for k,v in f.items() if k in portable_keys} for f in original]
    # The same common observations are retained in this diagnostic. Removed
    # contextual facts may still be useful; these copies are NOT new truth labels.
    assert all(f=={'outcome':'failure','credential_check':'invalid','auth_result':'failure'} for f in reduced)
    models={'v48':previous/'pressure','observed':root/'artifacts/v49_observed_encoding_20260914/pressure',
            'word_character':root/'artifacts/v49_word_character_20260914/pressure'}
    results=[];bindings={}
    for name,folder in models.items():
        b=joblib.load(folder/'model.joblib')
        def predict(tt,ff):
            x=sparse.hstack([b['text_encoder'].transform(tt),b['fact_encoder'].transform(ff)],format='csr')
            return b['model'].predict_proba(x)[ids]
        full=predict(texts,original);limited=predict(['']*len(original),reduced)
        d=pd.read_parquet(folder/'evaluation.parquet');y=d.label_index.to_numpy()
        probs=d[['p_benign','p_malicious','p_suspicious']].to_numpy()
        report=read(folder/'report.json')
        metric=report['after']
        auth=d[d.route=='authentication']
        pattern_stats=[]
        for pid,g in auth.groupby('projection_id'):
            q=g[['p_benign','p_malicious','p_suspicious']].to_numpy()
            pattern_stats.append({'text':p.iloc[int(pid)].text,'rows':len(g),
                'labels_B_M_S':[int((g.label_index==k).sum()) for k in range(3)],
                'prediction_B_M_S':q[0].tolist(),'errors':int((q.argmax(1)!=g.label_index.to_numpy()).sum())})
        results.append({'model':name,'pressure_rows':len(d),'errors':metric['errors'],'macro_f1':metric['macro_f1'],
            'normal_errors':metric['normal_errors'],'authentication':pattern_stats,
            'fit_side_common_evidence_probe':{'rows':len(selected),'body_groups':int(selected.body_group.nunique()),
                'official_labels_B_M_S':[int((selected.label_index==k).sum()) for k in range(3)],
                'full_view_predictions_B_M_S':[int((full.argmax(1)==k).sum()) for k in range(3)],
                'reduced_view_predictions_B_M_S':[int((limited.argmax(1)==k).sum()) for k in range(3)],
                'reduced_probability_B_M_S':limited[0].tolist(),
                'scope':'Fit-side information-removal diagnostic, not out-of-sample performance or proof the removed facts were irrelevant.'}})
        for n in ['model.joblib','evaluation.parquet','report.json']:
            bindings[(folder/n).relative_to(root).as_posix()]=sha(folder/n)
    out.mkdir(parents=True)
    save(out/'summary.json',{'models':results,'new_classifier_fits':2,'classifier_promoted':False,
        'official_labels_unchanged':True,'masking_used_for_training':False,'source_bindings':bindings})
    selected[['row_position','body_group','projection_id','label_index','route']].to_parquet(out/'portable_probe_rows.parquet',index=False)
    shutil.copy2(__file__,out/'summarize_v49.py')
    print(json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())
