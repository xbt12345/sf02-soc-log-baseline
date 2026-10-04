"""Audit all original gold, full inequalities, QP work and finite evidence."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v162_independent_actual_restoration_review import actual_proposal,rows_review,close,parameter_hash
from v160_independent_saved_direction_certificate import dot,norm
TRIAL=ROOT/'artifacts/v163_fixed_endpoint_one_sided_restoration_20261002'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v163_saved_joint_restoration_actual_audit_20261002'
EPS=float(np.finfo(float).eps)

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir();goldpath=ROOT/'data/official/train.parquet';planpath=ROOT/'training/review_policy/v163_fixed_endpoint_one_sided_restoration_contract.json';plan=read(planpath)
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),goldpath,planpath,ROOT/'training/v162_independent_actual_restoration_review.py',ROOT/'training/v160_independent_fixed_diagnostic_review.py',ROOT/'training/v161_independent_all_finite_results_review.py',ROOT/'training/v160_independent_saved_direction_certificate.py'}
    for role in range(3):
        files.add(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt');files.add(ROOT/f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/fixed_pure_error_targets.parquet');files|={DIAG/f'role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();reports=[]
    for role in range(3):
        folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');spec=plan['roles'][role];assert diag['exception'] is None and all(diag[k]==0 for k in ['new_fits','permanent_updates','new_fixed_error_target_gradients','new_full_original_class_gradients'])
        state=torch.load(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',map_location='cpu',weights_only=True)['state'];assert parameter_hash(state)==diag['initial_parameter_sha256']==diag['restored_parameter_sha256'] and diag['restored_joint_TRAIN_retention']['passed']
        baseline={s:pd.read_parquet(folder/'baseline'/f'{s}_original_rows.parquet') for s in ['OOF','deployment']}
        for scope,f in baseline.items():
            rows_review(folder/'baseline',scope,f,gold);restored,_,_,_=rows_review(folder/'restored',scope,f,gold);assert np.array_equal(restored.pred,f.pred);close(restored[['p0','p1','p2']].to_numpy(),f[['p0','p1','p2']].to_numpy());close(restored[['logp0','logp1','logp2']].to_numpy(),f[['logp0','logp1','logp2']].to_numpy())
        targets=pd.read_parquet(ROOT/f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/fixed_pure_error_targets.parquet').row_position;gs=[np.load(DIAG/f'role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]];iterations=[];qp=nit=proposals=0
        for path in sorted(folder.glob('restoration*/original_unit_restoration_review.json')):
            result=read(path);solver=read(path.parent/'restoration_solver_context.json');entered='optimizer_iterations' in result;assert solver['entered_SLSQP_QP']==entered
            if entered:qp+=1;nit+=result['optimizer_iterations'];assert 0<result['optimizer_iterations']<=256
            assert solver['QP_solves_this_role']==qp and solver['optimizer_iterations_this_role']==nit
            u=np.load(path.parent/'current_displacement.npy');a=np.load(path.parent/'active_complete_margin_normals.npy');b=np.load(path.parent/'base_margins.npy');c=np.load(path.parent/'actual_current_margins.npy');item=dict(path=path.relative_to(ROOT).as_posix(),status=result['status'],entered_QP=entered,optimizer_iterations=result.get('optimizer_iterations',0),active_protection_functions=len(a))
            if (path.parent/'displacement.npy').exists():
                candidate=np.load(path.parent/'displacement.npy');e=np.load(path.parent/'correction.npy');assert np.array_equal(candidate,u+e);matrix=np.vstack([a,-gs[0],-gs[1]]);rhs=np.r_[b-c,dot(gs[0],u)[0],dot(gs[1],u)[0]]
                for row,target,review in zip(matrix,rhs,result['inequality_reviews']):
                    value,error=dot(row,e);limit=max(error,16*EPS*max(abs(float(target)),norm(row)*norm(e)));assert value==review['linear_recovery'] and float(target)==review['target'] and limit==review['residual_limit'] and error==review['arithmetic_error'];assert (value-target>=-limit)==review['passed']
                for g,review in zip(gs,result['class_reviews']):
                    value,error=dot(g,candidate);resolution=16*EPS*norm(g)*norm(candidate);assert value==review['linear_change'] and error==review['arithmetic_error'] and resolution==review['resolution'] and (value<-max(error,resolution))==review['resolved_negative']
                certified=result['optimizer_success_reported'] and all(r['passed'] for r in result['inequality_reviews']) and all(r['resolved_negative'] for r in result['class_reviews']);assert certified==(result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard');item.update(all_original_inequalities_passed=all(r['passed'] for r in result['inequality_reviews']),class_reviews=result['class_reviews'],correction_norm=norm(e))
            probepath=path.parent/'finite_probe/probe.json'
            if probepath.exists():
                assert result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard';proposals+=1;item['actual']=actual_proposal(probepath.parent,baseline,targets,gs,state,gold)
            iterations.append(item)
        assert qp==diag['QP_solves'] and nit==diag['optimizer_iterations'] and len(iterations)==diag['restoration_solves']<=6 and proposals==diag['finite_proposals']<=6
        assert any(r.get('actual',{}).get('accepted',False) for r in iterations)==diag['actual_finite_restoration_pass']
        calls=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={};assert all(c['kind'] in ['head','feature','full_parameter_margin_gradient'] for c in calls)
        for kind,prefix in [('head','head'),('feature','feature'),('full_class_gradient','gradient'),('full_parameter_margin_gradient','margin')]:
            for event,key in [('attempt','attempts'),('completed','completed')]:
                events=[c for c in calls if c['kind']==kind and c['event']==event];assert [c['ordinal'] for c in events]==list(range(1,len(events)+1));counts[prefix+'_'+key]=len(events)
        assert counts==diag['counts'] and counts['margin_attempts']==counts['margin_completed']<=spec['fresh_margin_gradient_cap'] and counts['head_attempts']==(2+proposals)*(spec['OOF_chunks']+12)+counts['margin_attempts']<=spec['head_cap']
        reports.append(dict(role=role,status=diag['status'],actual_finite_pass=diag['actual_finite_restoration_pass'],counts=counts,QP_solves=qp,optimizer_iterations=nit,iterations=iterations,parameter_hash_restored=True,original_joint_retention_passed=True))
    check_bindings(bindings);heads=sum(r['counts']['head_attempts'] for r in reports);margins=sum(r['counts']['margin_attempts'] for r in reports);save(OUT/'audit.json',dict(status='saved_joint_inequality_restoration_original_gold_complete_vectors_guards_QP_iterations_and_costs_passed',reports=reports,all_three_actual_finite_pass=all(r['actual_finite_pass'] for r in reports),actual_new_heads=heads,actual_new_features=heads,actual_new_margin_gradients=margins,actual_QP_solves=sum(r['QP_solves'] for r in reports),actual_optimizer_iterations=sum(r['optimizer_iterations'] for r in reports),cumulative_heads=17334+heads,cumulative_all_complete_derivatives=422+margins,official_audit_calls=0,new_fits=0,permanent_updates=0,quality_acceptance=False,source_sha256=bindings));print(json.dumps(dict(status='V163_saved_joint_actual_audit_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
