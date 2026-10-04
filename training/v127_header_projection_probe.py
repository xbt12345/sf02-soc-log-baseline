"""No-fit cost and scope of applying a verified header projection to the frozen base."""
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v126_frozen_audit import ROOT, PARENT, read, sha, save
from v125_evaluate import TRACE, VIEW
from v125_model import Expert, probabilities
from v124_header import HEADER, transform, old_text
from v127_mechanism_probe import OUT


def main():
    target=OUT/'header_projection.json'
    if target.exists():raise FileExistsError(target)
    hp=ROOT/'artifacts/v124_header_trial_20260929'
    prereq=read(hp/'input_preflight.json')
    for name,h in prereq['output_sha256'].items():
        if sha(hp/name)!=h:raise ValueError('Header source differs '+name)
    for rel,h in prereq['source_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Header dependency differs '+rel)
    d=pd.read_parquet(TRACE,columns=['row_position','local','fold','truth','raw_message'])
    xo=sparse.load_npz(VIEW);xh=sparse.load_npz(hp/'B_header_ASA.npz')
    full_checks=0;source_forms_changed=0
    for raw in d.raw_message:
        m=HEADER.match(raw)
        if m is None:raise ValueError('Unknown header')
        body=raw[m.end():]
        projected='<164>Jan 01 2000 01:02:03: '+body
        sanitized='<164>Dec 31 2001 USER-CRED-45: '+body
        a=transform(raw)[0]
        if a!=old_text(projected) or a!=transform(sanitized)[0]:raise ValueError('Projection does not close variants')
        source_forms_changed+=a!=old_text(raw);full_checks+=1
    row_local=d.local.to_numpy();y=d.truth.to_numpy();folds=d.fold.to_numpy()
    original=np.empty(len(d),dtype=np.int8);canonical=original.copy();results=[]
    for fold in range(3):
        model=Expert('A',10201).cuda().eval()
        state=torch.load(PARENT/f'fold{fold}_A/epoch25_model.pt',map_location='cpu',weights_only=True)
        model.base.load_state_dict(state['base'])
        po=probabilities(model,xo,device='cuda');ph=probabilities(model,xh,device='cuda')
        saved=np.load(PARENT/f'fold{fold}_A/epoch25_prob.npy')
        if not np.allclose(po,saved,atol=2e-6,rtol=2e-6) or not np.array_equal(po.argmax(1),saved.argmax(1)):
            raise ValueError('Base replay mismatch')
        ix=folds==fold;original[ix]=po[row_local[ix]].argmax(1);canonical[ix]=ph[row_local[ix]].argmax(1)
        np.save(OUT/f'fold{fold}_A_header_probability.npy',ph)
        results.append({'fold':fold,'prediction_flips':int((original[ix]!=canonical[ix]).sum()),
                        'original_errors':int((original[ix]!=y[ix]).sum()),'canonical_errors':int((canonical[ix]!=y[ix]).sum())})
    counts={}
    for c in (1,2):
        ix=y==c
        counts[str(c)]={'rows':int(ix.sum()),'original_errors':int((original[ix]!=c).sum()),
                        'canonical_errors':int((canonical[ix]!=c).sum()),
                        'repaired':int((ix&(original!=c)&(canonical==c)).sum()),
                        'regressed':int((ix&(original==c)&(canonical!=c)).sum())}
    report={'status':'header_projection_replayed_no_fit','full_ASA_variant_equivalence_rows':full_checks,
            'changed_raw_inputs':source_forms_changed,'per_class':counts,'by_fold':results,
            'base_weights_unchanged':True,'classifier_fits':0,'optimizer_steps':0,
            'quality_acceptance':False,'model_promoted':False,
            'source_sha256':sha(__file__),'header_input_sha256':sha(hp/'B_header_ASA.npz'),
            'scope':'Only the registered ASA syslog header forms; body/parsed facts retained. Invariance does not establish class improvement or arbitrary format robustness.'}
    save(target,report)
    print(__import__('json').dumps(report),flush=True)


if __name__=='__main__':main()
