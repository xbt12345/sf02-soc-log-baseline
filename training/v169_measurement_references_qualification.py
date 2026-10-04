"""Saved actual q/logq exact-reference and one-ulp differing-value preservation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,sha
from v169_measurement_output_references_v2 import save,load

OUT=ROOT/'artifacts/v169_measurement_references_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();actual=ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0';qfile=actual/'OOF_q.npy';lfile=actual/'OOF_logq.npy';rowfile=actual/'OOF_original_rows.parquet'
    q,lp=np.load(qfile),np.load(lfile);rr=pd.read_parquet(rowfile);ids=np.sort(rr.local.unique())[:2048];point=OUT/'complete_outputs.npz';np.savez_compressed(point,OOF_q=q,OOF_logq=lp)
    cases=[]
    for name,changed in [('exact_full_chunk_reference',False),('one_ulp_kept_actual_values',True)]:
        folder=OUT/name;folder.mkdir();value=dict(q=q[ids].copy(),logq=lp[ids].copy(),margin=-0.)
        if changed:value['q'][0,0]=np.nextafter(value['q'][0,0],np.inf)
        for repetition in [0,1]:
            save(folder,repetition,value,point,'OOF',ids);qq,ll,margin=load(folder,repetition)
            assert qq.tobytes()==value['q'].tobytes() and ll.tobytes()==value['logq'].tobytes() and margin.hex()=='-0x0.0p+0'
            meta=json.loads((folder/f'repeat{repetition}_outputs.json').read_text(encoding='utf-8'))
            assert meta['exact_reference']==(not changed)
            assert (folder/f'repeat{repetition}_different_outputs.npz').exists()==changed
        cases.append(dict(case=name,outputs_exact=True,negative_zero_scalar_preserved=True,chunk_indices_saved_once=True))
    # Source corruption is rejected; retain a distinct fixture, not old results.
    fixture=OUT/'changed_source.npz';np.savez_compressed(fixture,OOF_q=q,OOF_logq=lp);folder=OUT/'changed_source_refused';folder.mkdir();value=dict(q=q[ids],logq=lp[ids],margin=0.)
    save(folder,0,value,fixture,'OOF',ids);changed=q.copy();changed[ids[0],0]=np.nextafter(changed[ids[0],0],np.inf);np.savez_compressed(fixture,OOF_q=changed,OOF_logq=lp)
    try:load(folder,0)
    except ValueError:pass
    else:raise AssertionError('Changed point source must refuse')
    files=[Path(__file__).resolve(),ROOT/'training/v169_measurement_output_references_v2.py',ROOT/'training/v169_prior_pair_training_entry_v7.py',qfile,lfile,rowfile]
    report=dict(status='V169_exact_same_point_references_one_ulp_fallback_and_source_change_refusal_qualified',cases=cases,source_corruption_refused=True,no_close_based_reference_deduplication=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],official_calls=0)))

if __name__=='__main__':main()
