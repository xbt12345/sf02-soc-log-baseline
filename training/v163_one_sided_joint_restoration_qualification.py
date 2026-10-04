"""Qualify the new inequality method on original saved and synthetic vectors."""
import json,traceback
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient
from v163_one_sided_joint_restoration import propose
OUT=ROOT/'artifacts/v163_one_sided_joint_restoration_qualification_20261002'
SOURCE=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002/role0/restoration3'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role0'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir();files={Path(__file__).resolve(),ROOT/'training/v163_one_sided_joint_restoration.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'docs/V162_INDEPENDENT_RESULTS_AND_TARGETED_NEXT_TRAINING_PLAN_20261002.md'}|{p for p in SOURCE.rglob('*') if p.is_file()}
    files|={DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0))
    u=np.load(SOURCE/'current_displacement.npy');a=np.load(SOURCE/'active_complete_margin_normals.npy');b=np.load(SOURCE/'base_margins.npy');c=np.load(SOURCE/'actual_current_margins.npy');gs=[np.load(DIAG/f'baseline_error_class{k}_repeat0/complete_fixed_error_target_gradient.npy') for k in [1,2]]
    actual=propose(u,a,b,c,*gs);assert actual['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard';again=propose(u,a,b,c,*gs);assert repeat_gradient(actual['displacement'],again['displacement'])['passed']
    np.save(OUT/'saved_role0_displacement.npy',actual['displacement']);np.save(OUT/'saved_role0_correction.npy',actual['correction']);save(OUT/'saved_role0_local_certificate.json',{k:v for k,v in actual.items() if k not in ['displacement','correction']})
    cases=[]
    for width in [3,1060832]:
        def embed(v):out=np.zeros(width);out[-3:]=v;return out
        displacement=embed([1.,-1.,-1.]);normals=np.stack([embed([1.,0.,0.]),embed([0.,1.,0.])]);base=np.array([.1,.1]);current=np.array([-.1,.2]);g1=embed([0.,0.,1.]);g2=embed([0.,1.,1.]);plain=propose(displacement,normals,base,current,g1,g2);assert plain['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard'
        for units in [np.array([1.,1.]),np.array([1e-24,1e24]),np.array([1e24,1e-24])]:
            scaled=propose(displacement,normals*units[:,None],base*units,current*units,g1,g2);assert scaled['status']==plain['status'] and repeat_values(scaled['displacement'],plain['displacement'],'vector')['passed'];cases.append(dict(width=width,units=units.tolist(),status=scaled['status']))
        ordered=propose(displacement,normals[::-1].copy(),base[::-1].copy(),current[::-1].copy(),g1,g2);assert ordered['status']==plain['status'] and repeat_values(ordered['displacement'],plain['displacement'],'vector')['passed']
        tiny=propose(displacement*1e-30,normals,base*1e-30,current*1e-30,g1*1e-30,g2*1e-30);assert tiny['status']==plain['status'] and repeat_values(tiny['displacement']/1e-30,plain['displacement'],'vector')['passed']
        opposed=propose(displacement,np.stack([embed([1.,0.,0.]),embed([-1.,0.,0.])]),np.ones(2),-np.ones(2),g1,g2);assert opposed['status']=='local_inequality_or_common_descent_unqualified_stop' and opposed['no_global_infeasibility_claim']
        wrong=propose(displacement,normals,base,current,-g1,-g2);assert wrong['status']=='local_inequality_or_common_descent_unqualified_stop'
    check_bindings(bindings);save(OUT/'qualification.json',dict(status='one_sided_joint_original_vector_saved_and_synthetic_numeric_qualification_passed',actual_saved_role0_local_pass=True,same_input_complete_gradient_repeat=True,unit_order_and_tiny_common_scale_cases=cases,opposed_constraints_rejected_no_global_claim=True,unresolved_class_descent_rejected=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,no_actual_model_safety_or_classification_claim=True,source_sha256=bindings));print(json.dumps(dict(status='V163_one_sided_joint_saved_and_synthetic_qualified',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
