"""Direct original-space recovery, units, rank and second guard qualification."""
import json,traceback
from pathlib import Path
import numpy as np
from experiment_review import ROOT,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_gradient,repeat_values,finite_step_review
from v162_multi_function_finite_restoration_v2 import propose
OUT=ROOT/'artifacts/v162_multi_restoration_synthetic_qualification_v2_20261002'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002/role2'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir()
    files={Path(__file__).resolve(),ROOT/'training/v162_multi_function_finite_restoration_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'artifacts/v162_finite_restoration_saved_qualification_20261002/actual_saved_displacement.npy'}|{DIAG/'round0/polished_direction.npy',DIAG/'round0/raw_margin_normals.npy',DIAG/'baseline_error_class1_repeat0/OOF_logq.npy',DIAG/'round0/probe4/OOF_logq.npy'}|{DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]}
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_bindings.json',dict(source_sha256=binding,official_calls=0))
    a=np.load(DIAG/'round0/raw_margin_normals.npy');u=np.load(DIAG/'round0/polished_direction.npy')/16;gs=[np.load(DIAG/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
    lp0=np.load(DIAG/'baseline_error_class1_repeat0/OOF_logq.npy');lp1=np.load(DIAG/'round0/probe4/OOF_logq.npy');b=np.array([lp0[21050,2]-lp0[21050,1]]);c=np.array([lp1[21050,2]-lp1[21050,1]])
    actual=propose(u,a,b,c,*gs);assert 'requires_full_actual_finite_guard' in actual['status'];assert repeat_gradient(actual['displacement'],np.load(ROOT/'artifacts/v162_finite_restoration_saved_qualification_20261002/actual_saved_displacement.npy'))['passed']
    np.save(OUT/'actual_saved_displacement.npy',actual['displacement']);cases=[]
    for width in [3,1060832]:
        def emb(v):z=np.zeros(width);z[-3:]=v;return z
        u=emb([.1,0,0]);a=np.stack([emb([0,1,0]),emb([0,0,1])]);b=np.array([1e-12,2e-12]);c=b-np.array([.27,.13])*.1**2;gm=emb([-1,.01,.02]);gs=emb([-.5,.02,.01])
        r=propose(u,a,b,c,gm,gs);assert 'requires_full_actual' in r['status'];x,y,z=r['displacement'][-3:];assert np.all(b+np.array([y-.27*x*x,z-.13*x*x])>0)
        for units in [[1.,1.],[1e-24,1e24],[1e12,1e-12]]:
            units=np.array(units);rr=propose(u,a*units[:,None],b*units,c*units,gm,gs)
            assert 'requires_full_actual' in rr['status'] and repeat_values(r['displacement'],rr['displacement'],'displacement')['passed']
        repeat=propose(u,a,b,c,gm,gs)
        if width==1060832:assert repeat_gradient(r['displacement'],repeat['displacement'])['passed']
        # Opposed recovery equalities cannot silently select one protected row.
        opposed=np.stack([a[0],-a[0]]);bad=propose(u,opposed,b,c,gm,gs);assert bad['status']=='local_restoration_residual_or_common_descent_failed_stop'
        # A one-function correction can hurt a second, unmodelled protected row.
        one=propose(u,a[:1],b[:1],c[:1],gm,gs);second_margin=b[1]-.5*one['displacement'][-2]
        assert second_margin<0;gate=finite_step_review([1.,1.],[.9,.9],[-.1,-.1],1,1,'B',1.,False);assert not gate['accepted']
        cases.append(dict(width=width,two_actual_nonlinear_toy_guards_restored=True,normal_margin_units_preserved=True,opposed_equalities_rejected_without_global_infeasibility_claim=True,unmodelled_second_guard_actual_classification_rejected=True))
    check_bindings(binding);save(OUT/'qualification.json',dict(status='multi_original_space_restoration_saved_vector_units_rank_and_all_guard_counterexamples_passed',actual_saved_candidate={k:v for k,v in actual.items() if k not in ['displacement','correction']},cases=cases,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,actual_corrected_model_not_yet_executed=True,source_sha256=binding))
    print(json.dumps(dict(status='V162_multi_restoration_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
