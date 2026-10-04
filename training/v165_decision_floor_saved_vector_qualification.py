"""Real saved-vector regression and confidence-independence; zero model calls."""
import json,traceback
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v165_decision_floor_restoration import propose

TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
PRIOR=ROOT/'artifacts/v164_independent_decision_floor_counterfactual_20261002'
OUT=ROOT/'artifacts/v165_decision_floor_saved_vector_qualification_20261002'
POLICY_FIELDS=['local_protection_target','actual_origin_margins_for_identity_only','actual_argmax_all_original_rows_and_retention_guards_still_required','confidence_restoration_not_required_by_this_local_policy']

def save(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();OUT.mkdir(); report=read(PRIOR/'review.json');assert len(report['results'])==6
    sources={Path(__file__).resolve(),ROOT/'training/v165_decision_floor_restoration.py',ROOT/'training/v163_one_sided_joint_restoration.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py',PRIOR/'review.json',PRIOR/'source_bindings.json'}; inputs=[]
    for row in report['results']:
        folder=ROOT/row['actual_source'];point=folder.parent;cached=PRIOR/f'{folder.parent.parent.name}_{point.name}_{folder.name}';refs=read(folder/'active_normal_references.json')
        normal_paths=[ROOT/ref['gradient'] for ref in refs.values()];gradient_paths=[point/f'class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]]
        sources.update([folder/'active_normal_references.json',*(folder/name for name in ['current_displacement.npy','base_margins.npy','actual_current_margins.npy']),*normal_paths,*gradient_paths,*(cached/name for name in ['counterfactual_original_unit_review.json','displacement.npy','correction.npy'])]);inputs.append((folder,cached,normal_paths,gradient_paths))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));results=[];qps=0
    for folder,cached,normal_paths,gradient_paths in inputs:
        u=np.load(folder/'current_displacement.npy');a=np.stack([np.load(p) for p in normal_paths]);b=np.load(folder/'base_margins.npy');c=np.load(folder/'actual_current_margins.npy');gm,gs=[np.load(p) for p in gradient_paths]
        original=read(cached/'counterfactual_original_unit_review.json');result=propose(u,a,b,c,gm,gs);qps+=int('optimizer_iterations' in result)
        core={k:v for k,v in result.items() if k not in ['displacement','correction',*POLICY_FIELDS]};assert core==original
        for key in ['displacement','correction']:assert np.array_equal(result[key],np.load(cached/f'{key}.npy'))
        # Increasing old confidence changes only identity diagnostics, never the
        # mathematical decision-floor target or returned full displacement.
        altered=propose(u,a,b*1e24,c,gm,gs);qps+=int('optimizer_iterations' in altered)
        for key in ['displacement','correction']:assert np.array_equal(result[key],altered[key])
        assert {k:v for k,v in altered.items() if k not in ['displacement','correction',*POLICY_FIELDS]}==core
        results.append(dict(source=folder.relative_to(ROOT).as_posix(),full_parameters=len(u),normal_functions=len(a),actual_prior_counterfactual_reproduced_bit_exact=True,confidence_magnitude_does_not_change_local_target_or_full_displacement=True,local_class_descent_and_original_unit_constraints=core['class_reviews'],classification_gain_still_unknown=True))
    invalid=False
    try:propose(np.ones(3),np.ones((1,3)),[-1.],[-1.],np.ones(3),np.ones(3))
    except ValueError:invalid=True
    assert invalid;check_bindings(bindings);save(OUT/'qualification.json',dict(status='saved_full_parameter_decision_floor_core_and_confidence_independence_qualified',cases=results,negative_origin_rejected=True,actual_CPU_QP_solves=qps,official_model_calls=0,official_gradients=0,fits=0,permanent_updates=0,finite_forward_qualification=False,no_training_authority=True,source_sha256=bindings));print(json.dumps(dict(status='V165_decision_floor_saved_vector_qualification_passed',actual_CPU_QP_solves=qps,official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
