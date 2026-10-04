"""Class-constrained direction and fixed functional perturbation qualification.

The six fixed counterfactual parameter evaluations are disclosed explicitly.
No optimizer, mutable parameter update, candidate fit or checkpoint selection.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json
import numpy as np
import pandas as pd
import torch
from v142_runtime import ROOT,OUT as BASE,read,save,sha,require_run_seal
from v142_train import tensors
from v138_train import configure,quick_stats
from v135_runtime import load_data
from v135_model import tensor_hash
from v138_readout import probabilities,full_gradient
from v144_pair_objective import build_pairs,gradients
from v145_class_direction import class_gradients,project_vector,flatten,unflatten,DESCENT_SLACK,FIXED_DIAGNOSTIC_STEP
from experiment_review import check_bindings
from v142_retention_check import check

OUT=ROOT/'artifacts/v145_class_direction_qualification_20261001'


def main():
    if OUT.exists():raise FileExistsError('No projection/step search or qualification overwrite')
    require_run_seal(ROOT/'training/v142_train.py');configure();_,d=load_data()
    trace=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    facts=pd.read_parquet(trace,columns=['facts_json']).facts_json;evidence=read(ROOT/'artifacts/v144_aux_gradient_evidence_20261001/probe.json')
    paths=[ROOT/'training/v145_class_direction.py',ROOT/'training/v145_class_direction_probe.py',ROOT/'training/v144_pair_objective.py',trace,
        ROOT/'artifacts/v144_aux_gradient_evidence_20261001/probe.json',BASE/'verified_TRAIN_mastery_registry.json']
    binding={str(p.relative_to(ROOT)):sha(p) for p in paths}
    OUT.mkdir();save(OUT/'pre_registered_probe.json',dict(status='fixed_before_gradients_and_functional_perturbations',new_optimizer_steps=0,new_mutable_parameter_updates=0,
        fixed_functional_parameter_evaluations=6,roles=[0,1,2],directions=['class_constrained_CE','class_constrained_CE_plus_fixed_ratio_auxiliary'],
        slack=DESCENT_SLACK,fixed_L2_step=FIXED_DIAGNOSTIC_STEP,temperature=0.1,auxiliary_gradient_ratio=0.1,
        coefficient_source='unchanged V144 legal TRAIN fixed initial values',source_sha256=binding,
        no_coefficient_step_or_slack_scan=True,no_HELD_labels_or_errors=True))
    reports=[];bundles={arm:[] for arm in ['A','B']};originals=[];failures=[]
    for f in range(3):
        frame,c,pure,ids,p,capacity=build_pairs(d,facts,f);h,ff,m=tensors(f)
        m.load_state_dict(torch.load(BASE/f'fold{f}_S2/endpoint.pt',map_location='cpu',weights_only=True)['state']);identity=tensor_hash(m.state_dict())
        params=tuple(m.parameters());names=[n for n,_ in m.named_parameters()];mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
        ce,mass_seen=full_gradient(m,h,ff,mass,ids);direct=tuple(v.grad.detach().clone() for v in params)
        risks,cgs,combined_ce,class_seen=class_gradients(m,h,ff,mass,ids)
        gap=float((flatten(direct)-flatten(combined_ce)).abs().max())
        if gap>1e-12:raise ValueError('Original-frequency CE gradient recomposition differs')
        value,ag=gradients(m,h,p);lam=evidence['folds'][f]['fixed_initial_auxiliary_coefficient']
        q=probabilities(m,h,ff,np.arange(len(c)));originals.append(frame[['row_position','truth']].assign(training_role=f,pred=q[frame.local].argmax(1)))
        details=[]
        for arm in ['A','B']:
            gradient=flatten(combined_ce if arm=='A' else tuple(a+lam*b for a,b in zip(combined_ce,ag)))
            try:direction,projection=project_vector(gradient,[flatten(v) for v in cgs])
            except ValueError as exc:
                failures.append(dict(fold=f,arm=arm,reason=str(exc)));details.append(dict(arm=arm,projected_nonzero=False));continue
            step=unflatten(direction,params);trial={name:v.detach()-FIXED_DIAGNOSTIC_STEP*z for name,v,z in zip(names,params,step)}
            allq=[];trial_risks=np.zeros(3)
            # Only legal TRAIN input IDs are evaluated; no held outputs are scored.
            with torch.no_grad():
                for start in range(0,len(ids),1024):
                    take=ids[start:start+1024];z=torch.func.functional_call(m,trial,(h[take],ff[take]));lp=torch.log_softmax(z,-1)
                    allq.append(lp.exp().mean(1).cpu().numpy());trial_risks+=(-lp.mean(1).cpu().numpy()*c[take]).sum(0)/np.maximum(c.sum(0),1)
            trialq=np.zeros_like(q);trialq[ids]=np.concatenate(allq)
            stats=quick_stats(frame,trialq,pure,q[frame.local].argmax(1),[22,6,28][f])
            rows=frame[['row_position','truth']].assign(training_role=f,pred=trialq[frame.local].argmax(1));bundles[arm].append(rows)
            finite_class_descent=all(trial_risks[y]<risks[y-1] for y in [1,2])
            if not stats['mastered'] or stats['old_correct_pure_regressions'] or not finite_class_descent:failures.append(dict(fold=f,arm=arm,reason='Fixed functional perturbation failed class-risk descent or actual classification guard'))
            details.append(dict(arm=arm,projected_nonzero=True,projection=projection,initial_M_S_member_CE=risks,
                functional_trial_M_S_member_CE=trial_risks[1:].tolist(),strict_finite_class_risk_descent=finite_class_descent,
                actual_TRAIN_classification=stats,trial_parameter_sha256=tensor_hash(trial),objective_descent_derivative=-float(gradient@direction),
                class_descent_derivatives=[-float(flatten(g)@direction) for g in cgs]))
            rows.to_parquet(OUT/f'fold{f}_{arm}_functional_trial_TRAIN_rows.parquet',index=False)
        if tensor_hash(m.state_dict())!=identity:raise ValueError('Qualification altered original model parameters/buffers')
        reports.append(dict(fold=f,capacity=capacity,original_full_member_CE=ce,original_class_mass_seen=mass_seen,class_gradient_mass_seen=class_seen,
            original_CE_gradient_recomposition_max_gap=gap,lambda_fixed=lam,auxiliary_loss=value,parameter_sha256=identity,parameter_identity_unchanged=True,directions=details))
        print(json.dumps(dict(stage='class_direction_qualified_role',report=reports[-1]),ensure_ascii=False),flush=True)
        del m,h,ff,params,direct,combined_ce,cgs,ag;gc.collect();torch.cuda.empty_cache()
    baseline=check(pd.concat(originals));guards={arm:check(pd.concat(rows)) for arm,rows in bundles.items() if len(rows)==3}
    passed=not failures and len(guards)==2 and baseline['passed'] and all(g['passed'] for g in guards.values())
    check_bindings(binding)
    save(OUT/'qualification.json',dict(status='class_direction_and_fixed_functional_trial_completed',latest_actual_training='V142',qualification_passed=bool(passed),
        new_classifier_fits=0,new_optimizer_steps=0,new_mutable_parameter_updates=0,fixed_functional_parameter_evaluations=6,
        full_original_CE_gradients=3,class_risk_gradients=6,auxiliary_gradients=3,folds=reports,baseline_guard=baseline,fixed_trial_guards=guards,failures=failures,
        second_issue_solved=False,model_promoted=False,training_entry_registered=False,evidence_bindings=binding,
        limits=['Six fixed gradient-informed functional parameter trials are preparation diagnostics, not trained candidate endpoints; explicitly disclosed, not hidden as zero computation.',
            'A single fixed small perturbation passing guard does not guarantee a full run or finite larger steps.',
            'No coefficient, slack or step was searched; failed roles are not replaced.',
            'Known fine behavior gaps and direct 578-S-error coverage zero remain; no new labels or input deletion.']))
    print(dict(qualification_passed=passed,failures=failures,new_optimizer_steps=0,fixed_functional_parameter_evaluations=6),flush=True)


if __name__=='__main__':main()
