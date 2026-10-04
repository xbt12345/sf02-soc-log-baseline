"""Descriptive diagnostics after the fixed probe; no fitting or threshold edits."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(root, run):
    assert not (run/'analysis.json').exists()
    spec=importlib.util.spec_from_file_location('v44_frozen',run/'probe_v44_ready_models.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    r,x,b,ab=mod.data(root)
    e=pd.read_parquet(run/'evaluation.parquet')
    y=e.label_index.to_numpy();bp=e.B_pred.to_numpy()
    stress=json.loads((run/'information_removal_stress.json').read_text(encoding='utf-8'))
    result={}
    for name,cols in [('hide_source',['src_port_fixed','src_port_range']),('hide_destination',['dst_port_fixed','dst_port_range'])]:
        ix=x[cols[0]]!=mod.MISSING
        xx=x.loc[ix].copy();xx[cols]=mod.MISSING
        d=r.loc[ix,['fold','label_index']].copy()
        d['key']=[json.dumps(v) for v in xx.to_numpy().tolist()]
        ct=pd.crosstab([d.fold,d.key],d.label_index).reindex(columns=[1,2],fill_value=0)
        res={'rows':len(d),'fold_specific_descriptive_minimum_errors':int(ct.min(axis=1).sum()),'not_a_model_score':True}
        for n in ['LR','CAT']:
            items=[v for v in stress if v['model']==n and v['scope']==name]
            res[n]={'before_errors':sum(v['before']['errors'] for v in items),
                    'after_errors':sum(v['after']['errors'] for v in items),
                    'after_confusion_M_S':np.sum([v['after']['confusion_M_S'] for v in items],axis=0).tolist()}
        result[name]=res
    receipt=json.loads((run/'removal_information_floor.json').read_text())
    assert all(result[k]['fold_specific_descriptive_minimum_errors']==v['fold_specific_descriptive_minimum_errors'] for k,v in receipt.items())
    bodies={}
    for n in ['LR','CAT']:
        delta=(e[n+'_head_pred'].to_numpy()!=y).astype(int)-(bp!=y).astype(int)
        bd=pd.Series(delta).groupby(e.body_group).sum()
        bodies[n]={'net_improved_body_keys':int((bd<0).sum()),'net_worsened_body_keys':int((bd>0).sum())}
    bprob=b[['p_benign','p_malicious','p_suspicious']].to_numpy()
    target=b.route.to_numpy()=='asa'
    overlays={}
    for n in ['LR','CAT','MASK']:
        p=bprob.copy();q=e[n+'_qM'].to_numpy();mass=bprob[target,1]+bprob[target,2]
        base=bprob[target].copy();part=base.copy();part[:,1]=mass*q;part[:,2]=mass*(1-q)
        fallback=(base.argmax(1)==0)|(part.argmax(1)==0);part[fallback]=base[fallback];p[target]=part
        assert np.array_equal(p[~target],bprob[~target])
        assert np.array_equal(p[:,0],bprob[:,0])
        assert np.array_equal(p.argmax(1)==0,bprob.argmax(1)==0)
        assert np.array_equal(p[target].argmax(1),e[n+'_protected_pred'].to_numpy())
        overlays[n]={'total_rows':len(p),'outside_ASA_rows':int((~target).sum()),
                     'outside_probability_changes':0,'benign_probability_changes':0,'normal_boundary_changes':0}
    natural_missing={}
    for k in ['src_port_fixed','dst_port_fixed']:
        ix=(x[k]==mod.MISSING).to_numpy()
        natural_missing[k]={'rows':int(ix.sum()),'malicious':int((ix&(y==1)).sum()),'suspicious':int((ix&(y==2)).sum())}
    mod.save(run/'analysis.json',{'information_removal':result,'body_key_effects':bodies,
        'protected_probability_reconstruction':overlays,'natural_missing_label_counts':natural_missing,
        'limits':'Removal is synthetic loss of genuine observations, not a label-preserving nuisance test or verified real missingness distribution. Body groups are not verified incidents.',
        'script_sha256':mod.sha(__file__),'model_fits':0})
    print(json.dumps({'information_removal':result,'natural_missing':natural_missing,'overlays':overlays}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();main(Path(a.root).resolve(),Path(a.run).resolve())
