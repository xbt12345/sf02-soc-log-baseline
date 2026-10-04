"""Preserve the frozen gate result, then audit its missing matched-arm comparison."""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from run_v75 import ROOT, save, sha
from v104_phase_b import ASA_IDS, FID, PREP
from v107_matched_training import DEST


def main():
    target=DEST/'adversarial_decision.json'
    assert not target.exists()
    e=json.loads((DEST/'evaluation.json').read_text(encoding='utf-8'))
    assert e['status']=='twelve_fits_evaluated_developmental'
    assert e['continue_replication'] and all(e['frozen_gate_checks'].values())
    d=pd.read_parquet(DEST/'OOF_ASA_ledger.parquet')
    old=pd.read_parquet(ROOT/'artifacts/v106_frozen_audit_20260928/paired_OOF_predictions.parquet',
                        columns=['row_position','root','nested_pattern'])
    assert np.array_equal(d.row_position.to_numpy(),old.row_position.to_numpy())
    y=d.truth.to_numpy(); a=d.N1_TabM25.to_numpy(); b=d.N2_TabM25.to_numpy()
    ac=a==y;bc=b==y
    repairs=(~ac)&bc;regressions=ac&(~bc)
    assert int(repairs.sum())==e['comparisons']['N2_TabM25']['vs_same_arch_N1']['repaired']
    assert int(regressions.sum())==e['comparisons']['N2_TabM25']['vs_same_arch_N1']['regressed']
    nested=old.nested_pattern.to_numpy(dtype=bool)
    c1=e['variants']['N1_TabM25'];c2=e['variants']['N2_TabM25']
    ids=np.load(ASA_IDS)
    fid=np.load(FID,mmap_mode='r')
    lookup=np.full(int(fid.max())+1,-1,np.int32)
    lookup[ids]=np.arange(len(ids))
    local=lookup[fid[d.row_position.to_numpy()]]
    assert (local>=0).all()
    n1=sparse.load_npz(PREP/'N1_ASA.npz')
    n2=sparse.load_npz(DEST/'N2_ASA.npz')
    changed_unique=np.asarray((n1!=n2).getnnz(axis=1)>0)
    changed=changed_unique[local]
    assert int(changed_unique.sum())==json.loads((DEST/'registration.json').read_text(encoding='utf-8'))['N2_changed_unique_ASA_inputs']
    def count(mask):
        return {'rows':int(mask.sum()),'N1_errors':int((mask&(~ac)).sum()),
                'N2_errors':int((mask&(~bc)).sum()),
                'N1_to_N2_repairs':int((mask&repairs).sum()),
                'N1_to_N2_regressions':int((mask&regressions).sum()),
                'M_N1_correct':int((mask&(y==1)&ac).sum()),
                'M_N2_correct':int((mask&(y==1)&bc).sum()),
                'S_N1_correct':int((mask&(y==2)&ac).sum()),
                'S_N2_correct':int((mask&(y==2)&bc).sum())}
    root_frame=pd.DataFrame({'root':d.root.to_numpy(),'truth':y,
                             'S_lost':(y==2)&regressions,
                             'S_gained':(y==2)&repairs})
    root_loss=root_frame.groupby('root',sort=False).agg(
        S_rows=('truth',lambda z:int((z==2).sum())),
        S_lost=('S_lost','sum'),S_gained=('S_gained','sum'))
    root_loss=root_loss[root_loss.S_lost>0].sort_values('S_lost',ascending=False)
    masks={'all':np.ones(len(d),bool),'changed_N2_input':changed,'unchanged_N2_input':~changed,
           'nested_wrapper':nested,'no_nested_wrapper':~nested}
    result={'status':'first_wave_adversarial_stop','training_fits':12,
      'source_sha256':sha(__file__),'evaluation_sha256':sha(DEST/'evaluation.json'),
      'OOF_sha256':sha(DEST/'OOF_ASA_ledger.parquet'),
      'frozen_gate_outcome':{'all_passed':True,'continue_replication_recorded':True,
                             'why_insufficient':'Compared N2_TabM25 to N1 linear teacher, not to matched N1_TabM25; easier reference masks harm from collapsing the wrapper.'},
      'matched_arm':{'N1_TabM25_ASA_errors':c1['all_ASA']['errors'],
                     'N2_TabM25_ASA_errors':c2['all_ASA']['errors'],
                     'extra_ASA_errors':c2['all_ASA']['errors']-c1['all_ASA']['errors'],
                     'N1_MS_F1':c1['all_ASA']['MS_equal_F1'],
                     'N2_MS_F1':c2['all_ASA']['MS_equal_F1'],
                     'S_correct_N1':c1['all_ASA']['class']['suspicious']['correct'],
                     'S_correct_N2':c2['all_ASA']['class']['suspicious']['correct'],
                     'M_correct_N1':c1['all_ASA']['class']['malicious']['correct'],
                     'M_correct_N2':c2['all_ASA']['class']['malicious']['correct'],
                     'repairs':int(repairs.sum()),'regressions':int(regressions.sum())},
      'slices':{name:count(mask) for name,mask in masks.items()},
      'per_fold':{str(k):count(d.fold.to_numpy()==k) for k in (0,1,2)},
      'historic_major_roots':{str(k):count(old.root.to_numpy()==k) for k in (637660,2868,2300)},
      'top_new_roots_for_S_loss':[{**{'root':int(k)},**{p:int(v) for p,v in row.items()}}
                                  for k,row in root_loss.head(12).iterrows()],
      'S_groups':{'N1_zero':c1['group_S']['zero_recall_groups'],
                  'N2_zero':c2['group_S']['zero_recall_groups'],
                  'total':c1['group_S']['groups']},
      'no_wrapper_MS_F1':{'N1':c1['slices']['no_nested_wrapper']['MS_equal_F1'],
                          'N2':c2['slices']['no_nested_wrapper']['MS_equal_F1']},
      'full_task':{'N1_composite_errors':e['full_task']['N1_TabM25_composite']['errors'],
                   'N2_composite_errors':e['full_task']['N2_TabM25_composite']['errors'],
                   'N1_benign_false_alerts':e['full_task']['N1_TabM25_composite']['class']['benign']['missed'],
                   'N2_benign_false_alerts':e['full_task']['N2_TabM25_composite']['class']['benign']['missed']},
      'decision':{'replicate_N2_seeds':False,'promote_N2':False,'promote_N1':False,
                  'reason':'Primary same-architecture input comparison fails severely and S groups remain sparse. Both first-wave results are development OOF, not an unseen environment.'},
      'limitations':['Observed body source is not certified real actor identity.',
                     'A N1 gain can depend on artificial wrapper; it cannot be called transfer success.',
                     'No private answer, external source, or platform score was used.']}
    assert result['matched_arm']['extra_ASA_errors']>0
    assert result['matched_arm']['S_correct_N2']<result['matched_arm']['S_correct_N1']
    assert result['no_wrapper_MS_F1']['N2']<result['no_wrapper_MS_F1']['N1']
    save(target,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
