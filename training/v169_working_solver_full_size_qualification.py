"""Actual full parameter CPU QPs for64 current functions, old arithmetic unchanged."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
from pathlib import Path
import numpy as np
from experiment_review import ROOT,sha
import v166_coverage_joint_restoration as old
import v169_working_joint_restoration as new
from v169_dynamic_trial_restoration import propose

OUT=ROOT/'artifacts/v169_working_solver_full_size_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();original=(ROOT/'training/v166_coverage_joint_restoration.py').read_text(encoding='utf-8');revised=(ROOT/'training/v169_working_joint_restoration.py').read_text(encoding='utf-8')
    assert revised==original.replace('MAX_NORMALS=25','MAX_NORMALS=64')
    results=[];QP_calls=0;iterations=0
    def callback(event,value):
        nonlocal QP_calls,iterations
        if event=='call':QP_calls+=1
        else:iterations+=int(value.nit) if value is not None else 0
    for arm,width in [('A',1060832),('B',1060833)]:
        u=np.zeros(width);u[:2]=1.;gm=np.zeros(width);gs=np.zeros(width);gm[0]=-1.;gs[1]=-1.
        if arm=='B':gm[-1]=.1;gs[-1]=.2
        a=np.zeros((64,width));a[np.arange(64),np.arange(2,66)]=1.;b=np.full(64,.2);c=np.full(64,-.1);tau=np.full(64,16*np.finfo(np.float64).eps)
        result=propose(callback,u,a,b,c,gm,gs,tau)
        assert result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard' and len(result['inequality_reviews'])==66
        assert all(r['passed'] for r in result['inequality_reviews']) and all(r['resolved_negative'] for r in result['class_reviews'])
        assert not result['finite_step_authority'] and not result['QP_optimality_claim']
        np.save(OUT/f'{arm}_full_displacement.npy',result['displacement']);np.save(OUT/f'{arm}_full_correction.npy',result['correction'])
        results.append(dict(arm=arm,complete_parameters=width,current_working_functions=64,status=result['status'],optimizer_iterations=result['optimizer_iterations'],all_original_unit_inequalities_passed=True))
        # Old <=25 function arithmetic is untouched in the new resource copy.
        z=old.propose(u,a[:2],b[:2],c[:2],gm,gs);n=new.propose(u,a[:2],b[:2],c[:2],gm,gs)
        assert z['status']==n['status'] and np.array_equal(z['displacement'],n['displacement']) and np.array_equal(z['correction'],n['correction'])
        try:new.propose(u,np.vstack([a,a[:1]]),np.r_[b,b[0]],np.r_[c,c[0]],gm,gs)
        except ValueError:pass
        else:raise AssertionError('Current65 must refuse before solve')
        del a
    assert QP_calls==2
    files=[Path(__file__).resolve(),ROOT/'training/v166_coverage_joint_restoration.py',ROOT/'training/v169_working_joint_restoration.py',ROOT/'training/v169_dynamic_trial_restoration.py']
    report=dict(status='V169_full_size_A_B64_current_function_CPU_QPs_and_old_arithmetic_identity_qualified',results=results,actual_new_full_size_CPU_QP_calls=2,additional_old_vs_new_comparison_CPU_QP_calls=4,optimizer_iterations_observed_by_callback=iterations,no_global_feasibility_or_actual_SOC_finite_acceptance_claim=True,current65_resource_bound_refused=True,union_history_not_solver_resource_count=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],actual_CPU_QPs=6,official_calls=0)))

if __name__=='__main__':main()
