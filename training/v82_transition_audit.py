"""Paired real error changes; label-sensitive diagnostics after selection freeze."""
import json
import joblib
import numpy as np
import pandas as pd
from run_v75 import save, sha, load_sparse
from v79_execute import rows
from v82_capacity import DEST, LAST


def main():
    if (DEST/'paired_changes.json').exists():raise FileExistsError('Frozen audit exists')
    r=rows();fid=np.load(LAST/'row_feature_id.npy');y=r.label_index.to_numpy()
    names={arm:json.loads((DEST/('primary_'+arm+'_fit.json')).read_text())['states'][-1]['name'] for arm in ['linear','nonlinear']}
    predictions={arm:np.load(DEST/(name+'_prediction.npy'))[fid] for arm,name in names.items()}
    a,b=predictions['linear'],predictions['nonlinear'];fit=~r.fold.isin([0,2]).to_numpy()
    masks={'fit':fit,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    table=[]
    for role,mask in masks.items():
        for route in ['ALL']+sorted(r.route.unique()):
            for cl in range(3):
                take=mask&(y==cl)
                if route!='ALL':take &= r.route.eq(route).to_numpy()
                repaired=int((take&(a!=y)&(b==y)).sum());regressed=int((take&(a==y)&(b!=y)).sum())
                table.append({'role':role,'route':route,'class':cl,'support':int(take.sum()),
                              'repaired':repaired,'new_errors':regressed,'net_repair':repaired-regressed})
    pd.DataFrame(table).to_csv(DEST/'paired_changes.csv',index=False)
    cohort=np.load(DEST/'fit_457_cohort.npy');ids=fid[cohort]
    x=load_sparse(LAST/'X');k=np.load(DEST/'primary_kernel.npy',mmap_mode='r')
    models={arm:joblib.load(DEST/(name+'.joblib')) for arm,name in names.items()}
    margins={}
    for arm,model in models.items():
        z=x[ids]@model['coef']+model['intercept'];extra=np.zeros(len(ids))
        if arm=='nonlinear':
            contribution=k[ids]@model['kernel_coef'];z+=contribution;extra=contribution[:,1]-contribution[:,2]
        margins[arm]={'M_minus_S_quantiles':np.quantile(z[:,1]-z[:,2],[0,.25,.5,.75,1]).tolist(),
                      'kernel_M_minus_S_quantiles':np.quantile(extra,[0,.25,.5,.75,1]).tolist(),
                      'correct':int((predictions[arm][cohort]==2).sum()),'rows':len(cohort)}
    out={'names':names,'actual_paired_role_changes':[x for x in table if x['route']=='ALL'],
         'fit_457_margins':margins,'source_sha256':sha(__file__),
         'scope':'Fixed converged states after selection; attribution does not justify choosing another H-best model or pseudo-labeling.'}
    save(DEST/'paired_changes.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
