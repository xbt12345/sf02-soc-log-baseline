"""Classwise paired errors and exact representation/support diagnosis."""
import collections
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v78_boundary import DEST, data, masks
from run_v75 import ROOT, OUT, BATCH, save, sha, features
from v75_views import matrix_hashes


def main():
    r,m=data();role=masks(r);y=r.label_index.to_numpy();a=np.load(DEST/'O_sgd_logits.npy').argmax(1);z=np.load(DEST/'O_lbfgs_logits.npy');b=z.argmax(1)
    records=[]
    for name,mask in role.items():
        for route in sorted(r.route.unique()):
            for label in [0,1,2]:
                take=mask&r.route.eq(route).to_numpy()&(y==label)
                records.append({'role':name,'route':route,'label_index':label,'rows':int(take.sum()),
                    'fixed':int((take&(a!=y)&(b==y)).sum()),'broken':int((take&(a==y)&(b!=y)).sum()),
                    'remaining_errors':int((take&(b!=y)).sum())})
    pd.DataFrame(records).to_csv(DEST/'optimizer_paired_cells.csv',index=False)
    wanted=np.flatnonzero(r.route.eq('asa'));hashes={}
    for start in range(0,len(wanted),BATCH):
        ix=wanted[start:start+BATCH]
        hashes.update(zip(ix,matrix_hashes(features(r,ix,m,'new'))))
    fit_counts=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for i in wanted[role['fit'][wanted]]:fit_counts[hashes[i]][y[i]]+=1
    out=[]
    for name in ['calibration','evaluation']:
        for i in wanted[role[name][wanted]]:
            if b[i]==y[i]:continue
            c=fit_counts.get(hashes[i],np.zeros(3,dtype=np.int64))
            reason='unseen_exact_representation' if c.sum()==0 else ('mixed_fit_labels' if (c>0).sum()>1 else ('seen_only_other_label' if c[y[i]]==0 else 'seen_same_label_but_wrong'))
            out.append({'row_position':int(i),'role':name,'component':int(r.component.iat[i]),'body_group':int(r.body_group.iat[i]),'label':int(y[i]),'pred':int(b[i]),
                'M_minus_S_logit':float(z[i,1]-z[i,2]),'exact_fit_B':int(c[0]),'exact_fit_M':int(c[1]),'exact_fit_S':int(c[2]),'reason':reason,'feature_sha256':hashes[i].hex()})
    d=pd.DataFrame(out);d.to_parquet(DEST/'asa_error_ledger.parquet',index=False)
    grouped=d.groupby(['role','component','label','pred','reason']).size().rename('errors').reset_index().sort_values('errors',ascending=False)
    grouped.to_csv(DEST/'asa_error_components.csv',index=False)
    fit_mixed=sum(int(v.sum()-v.max()) for v in fit_counts.values())
    summary={'new_fits':0,'input_scope':'Actual corrected R0 byte/fact/metadata row equality, original unit frequencies.',
        'ASA_fit_empirical_conflict_error_floor':fit_mixed,'error_reasons':d.groupby(['role','reason']).size().rename('rows').reset_index().to_dict('records'),
        'top_components':grouped.head(12).to_dict('records'),'source_sha256':sha(__file__),
        'warning':'Exact unseen does not mean no behavior support. Conflicting observed labels do not identify wrong labels or prove absent raw evidence.'}
    save(DEST/'error_audit.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
