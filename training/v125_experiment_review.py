"""V125 versioned execution gates; historical fixed-input checker is untouched."""
import hashlib
import json
from pathlib import Path

from v124_experiment_review import ROOT, sha, read, check_bindings, ReviewError


def review_plan(plan,preflight):
    t=plan['training'];branch=plan['new_branch'];data=plan['data'];accept=plan['primary_promotion_eligibility']
    flags={
        'version_and_scope':plan['version']=='V125' and plan['latest_actual_training']=='V124' and plan['status'].startswith('design_only'),
        'three_arm_identity':set(plan['arms'])=={'A','B','C'} and set(map(tuple,t['order']))=={(f,a) for f in range(3) for a in 'ABC'},
        'schedule':t['primary_fits']==9 and t['epochs']==25 and t['seed']==10201 and t['expected_steps_per_arm']==[1825,800,1825] and t['expected_primary_steps']==13350,
        'objective':t['loss']=='original-row-frequency three-class mean-member CE' and not t['class_or_group_reweighting'] and not t['dynamic_error_routing'] and not t['heldout_label_updates'],
        'fixed_checkpoint':t['selector'].startswith('fixed epoch25') and t['checkpoints']==[1,2,5,10,15,20,25],
        'data':data['ASA_original_rows']==112807 and data['full_scoring_rows']==2056871 and data['official_labels_unchanged'] and data['original_row_frequency_unchanged'] and not data['external_training_data'],
        'new_branch':branch['vocabulary'].startswith('256 bytes') and branch['dimension']==64 and branch['transformer_layers']==2 and branch['attention_heads']==4 and branch['feed_forward_dimension']==128 and branch['additional_parameters_max']==250000 and branch['truncation'] is False,
        'preflight':preflight['status']=='input_prepared_before_fit' and preflight['rows']==112807 and preflight['old_locals']==preflight['fine_body_groups']==22546 and preflight['old_local_splits']==0 and preflight['truncated_bytes']==0 and preflight['C_byte_multiset_equal_B'] and preflight['C_exact_equal_B_rows']==0 and preflight['cross_fold_old_input_keys']==0,
        'quality':(accept['ASA_M_errors_max'],accept['ASA_S_errors_max'],accept['ASA_total_errors_max'])==(318,2094,2170) and accept['at_least_two_improving_folds'] and accept['paired_A_M_S_recall_and_F1_no_regression'] and accept['full_B_M_S_class_protection'] and accept['S_zero_recall_group_count_no_increase'],
        'confirmation':plan['confirmation']['only_after_all_primary_gates'] and plan['confirmation']['seeds']==[10202,10203] and plan['confirmation']['additional_fits']==12 and not plan['confirmation']['promotion_automatic'],
        'review_not_training':plan['classifier_fits_this_review']==0 and not plan['model_promoted'],
    }
    for rel,h in preflight['data_sha256'].items():flags['data_sha256_'+rel]=sha(ROOT/rel)==h
    for name,h in preflight['files_sha256'].items():flags['input_sha256_'+name]=sha(ROOT/'artifacts/v125_order_trial_20260929'/name)==h
    if not all(flags.values()):raise ReviewError('V125 plan/preflight blocked: '+','.join(k for k,v in flags.items() if not v))
    return {'status':'eligible_for_bounded_trial','passed':True,'checks':flags,'scope':'Pre-update design/input only; no classification claim.'}


def seal_run(plan_path,trainer_path,source_paths,out):
    if out.exists():raise FileExistsError(out)
    plan=read(plan_path);preflight=read(ROOT/'artifacts/v125_order_trial_20260929/input_preflight.json')
    review_plan(plan,preflight)
    paths={Path(plan_path),Path(trainer_path),Path(__file__),ROOT/'training/experiment_review.py'}|set(map(Path,source_paths))
    bindings={p.resolve().relative_to(ROOT).as_posix():sha(p) for p in paths}
    obj={'status':'sealed_before_fit','plan_path':Path(plan_path).resolve().relative_to(ROOT).as_posix(),
         'trainer_path':Path(trainer_path).resolve().relative_to(ROOT).as_posix(),
         'source_sha256':dict(sorted(bindings.items())),'review_checks':list(review_plan(plan,preflight)['checks']),
         'quality_acceptance':False,'model_promoted':False}
    out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def require_run_seal(seal_path,trainer_path):
    seal=read(seal_path)
    if seal['status']!='sealed_before_fit' or (ROOT/seal['trainer_path']).resolve()!=Path(trainer_path).resolve():raise ReviewError('Wrong V125 execution seal')
    check_bindings(seal['source_sha256'])
    plan=read(ROOT/seal['plan_path'])
    review_plan(plan,read(ROOT/'artifacts/v125_order_trial_20260929/input_preflight.json'))
    return plan


def require_checkpoint(plan,metadata):
    if metadata['status']!='fit_executed' or metadata['completed_epochs']!=plan['training']['epochs'] or metadata['prediction_epoch']!=25:raise ReviewError('Unregistered or incomplete V125 endpoint')
