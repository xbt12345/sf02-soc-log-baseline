"""Verify completed v6.2 artifacts and frozen v6.1 evidence; never retrain."""
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from train_v62_heads import ROOT,OUT,Cache,Head,evaluate,group_metrics
from v61_common import read,save,sha,metrics


def main():
    torch.set_num_threads(4)
    dest=ROOT/'evidence/2026-09-19/v62_capacity';dest.mkdir(parents=True,exist_ok=True)
    old=ROOT/'evidence/2026-09-15/v61_execution/delivery.json';prior=read(old)
    failures=[name for name,digest in prior['bindings'].items() if sha(ROOT/name)!=digest]
    assert not failures,failures
    hp=read(OUT/'head_protocol.json');op=read(OUT/'objective_protocol.json')
    for name,digest in hp['sources'].items():assert sha(ROOT/'training'/name)==digest,name
    assert op['source_sha256']==sha(ROOT/'training/train_v62_objective.py')
    assert op['head_source_sha256']==sha(ROOT/'training/train_v62_heads.py')
    assert op['cache_receipt_sha256']==sha(OUT/'cache_receipt.json')
    assert op['pre_objective_analysis_sha256']==sha(OUT/'pre_objective_analysis.json')
    for arm,info in op['weights'].items():assert info['file_sha256']==sha(OUT/(arm+'_fit_weights.parquet'))
    data=Cache();arms=['meanmax','attention','class_balanced','source_balanced'];checks={}
    for arm in arms:
        r=read(OUT/arm/'result.json');assert r['model_sha256']==sha(OUT/arm/'model.pt')
        model=Head(arm=='attention').cuda()
        model.load_state_dict(torch.load(OUT/arm/'model.pt',map_location='cuda',weights_only=True))
        checks[arm]={}
        for role,idx in [('fit',data.fit),('selection',data.dev)]:
            actual=evaluate(model,data,idx)
            saved=pd.read_parquet(OUT/arm/(role+'.parquet'))
            assert np.array_equal(saved.row_position,data.df.iloc[idx].row_position)
            assert np.array_equal(saved.label,data.df.iloc[idx].label)
            p=saved[['p_B','p_M','p_S']].to_numpy()
            assert np.isfinite(actual).all() and np.isfinite(p).all()
            maximum=float(np.max(np.abs(actual-p)));assert maximum<2e-6
            assert np.array_equal(actual.argmax(1),p.argmax(1))
            y=saved.label.to_numpy();mask=y>0
            norm=p.astype(np.float64);norm/=norm.sum(1,keepdims=True)
            observed=metrics(y[mask],norm[mask]);reported=r['selected_'+role]['ASA']
            assert observed['confusion_B_M_S']==reported['confusion_B_M_S']
            assert observed['errors']==reported['errors']
            assert abs(observed['macro_f1_M_S']-reported['macro_f1_M_S'])<1e-12
            g=group_metrics(data.df.iloc[idx],p)
            assert abs(g['M_S_class_balanced_source_recall']-r['selected_source_'+role]['M_S_class_balanced_source_recall'])<1e-12
            checks[arm][role]={'rows':len(y),'reloaded_argmax_identical':True,
                'maximum_probability_difference':maximum,'maximum_probability_sum_error':float(abs(p.sum(1)-1).max()),
                'confusion_errors_macro_and_source_metrics_recomputed':True}
        del model;torch.cuda.empty_cache()
        print(arm+' reload and metric recomputation passed.',flush=True)
    report=read(OUT/'capacity_analysis.json');assert not any(g['all_passed'] for g in report['continuation_gates'].values())
    files=list(OUT.rglob('*'))+list((ROOT/'training').glob('*v62*.py'))+[ROOT/'docs/V62_MODEL_CAPACITY_REVIEW.md']
    bindings={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files if p.is_file()}
    receipt={'status':'capacity_and_objective_diagnostics_completed_no_model_promoted',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'platform_used':False,
        'new_head_fits':4,'epochs_per_fit':12,'seed_count':1,
        'backbone':'Same frozen D epoch-1 text and fact encoder for all four new heads',
        'metric_replay':checks,'prior_v61_bound_files_verified':len(prior['bindings']),
        'prior_v61_delivery_sha256':sha(old),'prior_v61_failures':failures,
        'continuation_gates':report['continuation_gates'],'quality_acceptance':False,
        'validation_scope':'Adaptive fit/selection development; not blind or external validation',
        'probabilities_normalized_only_for_metric_recomputation_not_decisions':True,
        'bindings':bindings}
    save(dest/'delivery.json',receipt)
    print({'prior_v61_bound_files_verified':len(prior['bindings']),'new_bound_files':len(bindings),
        'reloads_passed':len(checks),'quality_acceptance':False})


if __name__=='__main__':main()
