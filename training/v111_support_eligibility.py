"""No-fit support eligibility, paired errors, and exact-behavior case recount."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT,save,sha
from v110_layer_probes import LEDGER,DEST as V110,check_registration

DEST=ROOT/'artifacts/v111_root_review_20260929'


def main():
    check_registration()
    target=DEST/'support_eligibility.json';assert not target.exists()
    d=pd.read_parquet(LEDGER)
    o=pd.read_parquet(V110/'OOF_probe_comparison.parquet')
    q=pd.read_parquet(DEST/'P2_margin_decomposition.parquet')
    assert np.array_equal(d.row_position,o.row_position) and np.array_equal(d.row_position,q.row_position)
    r={'status':'no_fit_support_eligibility_diagnosis','classifier_fits':0,'calibration_fits':0,
       'source_sha256':sha(__file__),'input_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [LEDGER,V110/'OOF_probe_comparison.parquet',DEST/'P2_margin_decomposition.parquet']},
       'models':{},'support':{},'exact_udp514_train_and_target':[]}
    for model in ('N1','N2','P1','P2'):
        pred=d[model+'_TabM25'].to_numpy() if model.startswith('N') else o[model+'_prediction'].to_numpy()
        err=pred!=d.truth.to_numpy()
        r['models'][model]={name:{'errors':int((err&(d.truth==c)).sum()),
          'error_roots':int(d.loc[err&(d.truth==c),'root'].nunique()),
          'error_unique_inputs':int(d.loc[err&(d.truth==c),'local'].nunique())} for c,name in [(1,'M'),(2,'S')]}
    support_table=d[['row_position','fold','root','truth']].copy()
    for key in ('behavior','coarse'):
        countM=np.zeros(len(d),int);countS=np.zeros(len(d),int)
        for k in (0,1,2):
            tr=d[(d.fold!=k)&d[key].notna()]
            counts=tr.groupby([key,'truth']).root.nunique().to_dict()
            mask=d.fold.to_numpy()==k
            countM[mask]=[counts.get((v,1),0) for v in d.loc[mask,key]]
            countS[mask]=[counts.get((v,2),0) for v in d.loc[mask,key]]
        support_table[key+'_M_training_roots']=countM
        support_table[key+'_S_training_roots']=countS
        for minimum in (1,2,3):
            ok=(countM>=minimum)&(countS>=minimum)&d[key].notna().to_numpy()
            summary={}
            for c,name in ((1,'M'),(2,'S')):
                m=ok&(d.truth.to_numpy()==c)
                summary[name]={'eligible_rows':int(m.sum()),'eligible_roots':int(d.loc[m,'root'].nunique()),
                    'N1_errors_in_eligible':int((m&(d.N1_TabM25.to_numpy()!=c)).sum()),
                    'N2_errors_in_eligible':int((m&(d.N2_TabM25.to_numpy()!=c)).sum()),
                    'P2_errors_in_eligible':int((m&(o.P2_prediction.to_numpy()!=c)).sum())}
            r['support'][key+'_at_least_'+str(minimum)+'_train_roots_per_class']=summary
    b=d.loc[d.root==2868,'behavior'].iloc[0]
    for root in (2868,29,11083,216921):
        z=d[(d.root==root)&(d.behavior==b)]
        if z.empty:continue
        # The OOF margin for training roots comes from a different fold. Do not
        # mistake it for the fold-1 fitted margin used by root 2868.
        r['exact_udp514_train_and_target'].append({'root':root,'fold':int(z.fold.iloc[0]),
            'M_rows':int((z.truth==1).sum()),'S_rows':int((z.truth==2).sum()),
            'unique_inputs':int(z.local.nunique()),'N1_OOF_errors':int((z.N1_TabM25!=z.truth).sum()),
            'N2_OOF_errors':int((z.N2_TabM25!=z.truth).sum())})
    r['scope']='Training-side source-proxy support only. Three sources per class is a necessary coverage screen for fit/weight-target/selection separation, not a proven sufficient rule, and is not selected for performance.'
    r['limitations']=['Behavior and coarse keys omit possible security context; exact-key support is not the full population support.',
        'Group source proxies are not confirmed independent environments.',
        'Eligibility criteria use training labels in aggregate, never held-out true class; error analysis is retrospective and never an inference route.',
        'No claim that meta-reweighting is already effective or that missing support can be fixed by duplicating rows.']
    support_table.to_parquet(DEST/'train_side_support.parquet',index=False)
    r['support_table_sha256']=sha(DEST/'train_side_support.parquet')
    save(target,r)
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
