"""Read-only V128 mechanism summary; does not train, select or relabel."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

from experiment_review import sha
from v128_experiment_review import read, save
from v128_train_r3 import OUT, BASE, role_frame, score_role, TRACE


def summarize(z, rows):
    ids=rows.local.to_numpy(np.int64);truth=rows.truth.to_numpy(np.int64)
    v=z[ids]
    member_logp=(v-logsumexp(v,axis=-1)[:,:,None]).mean(1)
    p=softmax(v,axis=-1).mean(1)
    pred=p.argmax(1)
    result={}
    for c in (1,2):
        mask=truth==c
        truep=p[mask,c];wrong=pred[mask]!=c
        result[str(c)]={'original_rows':int(mask.sum()),'wrong_original_rows':int(wrong.sum()),
            'unique_inputs':int(rows.loc[mask,'local'].nunique()),
            'roots':int(rows.loc[mask,'root'].nunique()),
            'member_CE':float(-member_logp[mask,c].mean()),
            'ensemble_CE':float(-np.log(np.maximum(truep,1e-12)).mean()),
            'true_class_probability_quantiles':np.quantile(truep,[0,.1,.25,.5,.75,.9,1]).tolist()}
    return result


def main():
    target=OUT/'score_role_postmortem.json'
    if target.exists():raise FileExistsError(target)
    rows=[]
    for f in range(3):
        roles=role_frame(f)
        outer=np.load(BASE/f'fold{f}_base_member_logits.npy')
        item={'fold':f,'original_fit_rows':len(roles),'unique_fit_inputs':roles.local.nunique(),
              'fit_roots':roles.root.nunique(),'outer_base_on_fit':summarize(outer,roles)}
        for role in ('IS','CF'):
            z,c,provenance=score_role(f,role)
            item[role]={'score':summarize(z,roles),'teacher_provenance':provenance,
                'endpoint_K_bias':read(OUT/f'fold{f}_K_{role}/fit.json')['bias'],
                'endpoint_P_train_role':read(OUT/f'fold{f}_P_{role}/checkpoints.json')[-1]['train_role'],
                'endpoint_P_outer_held':read(OUT/f'fold{f}_P_{role}/checkpoints.json')[-1]['outer_held']}
        rows.append(item)
    groups=pd.read_csv(OUT/'source_group_changes.csv')
    for arm in ('P_IS','P_CF'):
        groups[arm+'_net_errors_vs_A0']=groups[arm+'_errors']-groups.A0_errors
    leaders={arm:groups.sort_values(arm+'_net_errors_vs_A0',ascending=False).head(12)[
        ['root','truth','fold','rows','A0_errors',arm+'_errors',arm+'_net_errors_vs_A0']
        ].to_dict('records') for arm in ('P_IS','P_CF')}
    gains={arm:groups.sort_values(arm+'_net_errors_vs_A0').head(12)[
        ['root','truth','fold','rows','A0_errors',arm+'_errors',arm+'_net_errors_vs_A0']
        ].to_dict('records') for arm in ('P_IS','P_CF')}
    result={'status':'read_only_postfit_diagnostic','source_sha256':sha(__file__),
        'training_run_seal_sha256':sha(OUT/'run_seal.json'),
        'score_roles_by_fold':rows,'largest_regression_groups':leaders,
        'largest_repair_groups':gains,
        'interpretation_boundary':'Role scores are from smaller, uneven inner teachers; paired controls do not isolate every teacher-size or support shift. No promotion or new threshold is chosen here.'}
    save(target,result)
    brief=[{'fold':x['fold'],
            'IS':{c:v['wrong_original_rows'] for c,v in x['IS']['score'].items()},
            'CF':{c:v['wrong_original_rows'] for c,v in x['CF']['score'].items()}}
           for x in rows]
    print({'stage':result['status'],'fold_role_M_S_errors':brief})


if __name__=='__main__':main()
