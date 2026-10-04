"""True legal-TRAIN gradients and bounded functional steps; no model fitting."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import check_bindings, sha
from v135_model import tensor_hash
from v135_runtime import load_data, fit_context
from v138_readout import full_gradient, probabilities
from v138_train import configure, quick_stats
from v142_retention_check import check
from v146_train import initialize
from v146_runtime import require
from v148_relation_loss import prepare, relation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v148_factual_relation_qualification_20261001'
GRAD_RATIO=.1
MAX_TRIALS=20
INITIAL_STEP=.01
MIN_STEP=1e-8
ARMIJO=1e-4


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


@torch.no_grad()
def values(m,h,ff,c,ids,graph,parameters,coefficient):
    q=np.zeros((len(c),3));class_losses=np.zeros(3)
    for start in range(0,len(ids),2048):
        take=ids[start:start+2048]
        logits=torch.func.functional_call(m,parameters,(h[take],ff[take]),strict=False)
        lp=torch.log_softmax(logits,-1)
        q[take]=lp.exp().mean(1).cpu().numpy()
        class_losses+=(-lp.mean(1).cpu().numpy()*c[take]).sum(0)
    ce=float(class_losses.sum()/c.sum());aux=relation(m,h,graph,parameters=parameters)
    return dict(original_member_CE=ce,factual_relation_loss=aux,lambda_effective=coefficient,
        objective=ce+coefficient*aux,original_class_mass_seen=c.sum(0).tolist(),
        per_class_member_CE=(class_losses/np.maximum(c.sum(0),1)).tolist()),q


def main():
    target=OUT/'gradient_probe.json'
    assert not target.exists() and not (OUT/'functional_trials.jsonl').exists()
    require(ROOT/'training/v146_train.py');configure()
    capacity=read(OUT/'capacity.json');check_bindings(capacity['source_sha256'])
    _,d=load_data();reports=[];rows={a:[] for a in ['A','B']};all_trials=[]
    CE_gradients=aux_gradients=0
    for fold in range(3):
        info=capacity['folds'][fold]
        assert sha(OUT/f'fold{fold}_edges.parquet')==info['edge_sha256'] and sha(OUT/f'fold{fold}_nodes.parquet')==info['node_sha256']
        graph=prepare(pd.read_parquet(OUT/f'fold{fold}_edges.parquet'))
        frame,c,pure,_,ids=fit_context(d,fold);h,ff,m=initialize(fold)
        before=tensor_hash(m.state_dict());names=[n for n,_ in m.named_parameters()]
        mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
        ce,seen=full_gradient(m,h,ff,mass,ids);CE_gradients+=1;cg=[p.grad.clone() for p in m.parameters()]
        repeated,seen2=full_gradient(m,h,ff,mass,ids);CE_gradients+=1
        assert ce==repeated and seen==seen2 and all(torch.equal(g,p.grad) for g,p in zip(cg,m.parameters()))
        m.zero_grad(set_to_none=True);aux=relation(m,h,graph,backward_scale=1.);aux_gradients+=1
        ag=[p.grad.clone() for p in m.parameters()]
        m.zero_grad(set_to_none=True);again=relation(m,h,graph,backward_scale=1.);aux_gradients+=1
        assert aux==again and all(torch.equal(g,p.grad) for g,p in zip(ag,m.parameters()))
        cn=float(sum(g.square().sum() for g in cg).sqrt());an=float(sum(g.square().sum() for g in ag).sqrt())
        assert np.isfinite(an) and an>0 and all(torch.isfinite(g).all() for g in ag)
        coefficient=GRAD_RATIO*cn/an
        old=probabilities(m,h,ff,np.arange(len(c)));oldpred=old[frame.local].argmax(1)
        base_parameters={n:p.detach().clone() for n,p in m.named_parameters()}
        arm_reports=[]
        # lambda is fixed above, before any finite probe; no HELD score is read.
        for arm in ['A','B']:
            lam=0. if arm=='A' else coefficient
            gradient=cg if arm=='A' else [a+lam*b for a,b in zip(cg,ag)]
            norm=float(sum(g.square().sum() for g in gradient).sqrt())
            base,_=values(m,h,ff,c,ids,graph,base_parameters,lam)
            assert abs(base['objective']-(ce+lam*aux))<1e-12
            step=INITIAL_STEP;accepted=None;proposals=[]
            for k in range(MAX_TRIALS):
                if step<MIN_STEP:break
                parameters={n:base_parameters[n]-step*g/norm for n,g in zip(names,gradient)}
                val,q=values(m,h,ff,c,ids,graph,parameters,lam)
                s=quick_stats(frame,q,pure,oldpred,[22,6,28][fold])
                guard=s['mastered'] and s['new_errors_vs_start']==0
                bound=base['objective']-ARMIJO*step*norm
                passed=bool(guard and val['objective']<base['objective'] and val['objective']<=bound)
                item=dict(fold=fold,arm=arm,trial=k+1,step=step,base_objective=base['objective'],Armijo_bound=bound,
                    classification_guard=bool(guard),accepted=passed,stats=s,**val)
                proposals.append(item);all_trials.append(item)
                if passed:
                    accepted=item
                    r=frame[['row_position','truth']].assign(training_role=fold,pred=q[frame.local].argmax(1))
                    rows[arm].append(r);r.to_parquet(OUT/f'fold{fold}_{arm}_functional_TRAIN_rows.parquet',index=False)
                    break
                step*=.5
            assert tensor_hash(m.state_dict())==before
            arm_reports.append(dict(arm=arm,finite_qualification_passed=accepted is not None,
                full_gradient_norm=norm,finite_trials=len(proposals),initial_values=base,accepted_functional_point=accepted))
        reports.append(dict(fold=fold,CE_gradient_norm=cn,fact_gradient_norm=an,factual_relation_loss=aux,
            original_member_CE=ce,fixed_initial_coefficient=coefficient,fixed_gradient_ratio=GRAD_RATIO,
            repeated_gradients_exact=True,model_state_unchanged=True,initial_parameter_sha256=before,arms=arm_reports))
        print(json.dumps(dict(stage='actual_fact_gradient_and_functional_probe',**reports[-1]),ensure_ascii=False),flush=True)
        del m,h,ff,mass;gc.collect();torch.cuda.empty_cache()
    guards={a:check(pd.concat(rows[a],ignore_index=True)) if len(rows[a])==3 else dict(passed=False,reason='no complete feasible functional bundle') for a in ['A','B']}
    qualification=all(z['finite_qualification_passed'] for f in reports for z in f['arms']) and all(z['passed'] for z in guards.values())
    with (OUT/'functional_trials.jsonl').open('w',encoding='utf-8') as stream:
        for z in all_trials:stream.write(json.dumps(z,ensure_ascii=False)+'\n')
    result=dict(status='actual_factual_gradient_and_finite_objective_guard_qualification_only',latest_actual_training='V146',
        new_fits=0,new_optimizer_updates=0,new_persistent_parameter_updates=0,
        full_CE_gradient_evaluations=CE_gradients,factual_auxiliary_gradient_evaluations=aux_gradients,
        functional_parameter_evaluations=len(all_trials),qualification_passed=qualification,folds=reports,retention=guards,
        next_training_registered=False,issue_solved=False,quality_acceptance=False,
        limits=['Nonzero factual gradient and guarded local descent are not new threat-class evidence or source-generalization acceptance.',
            'No trained endpoint/checkpoint is produced or selected; V146 remains the latest actual classifier trial.',
            'The auxiliary target shares already observable components within protocol/role context; classifier information is not increased.',
            'HELD classes/errors are not used for weights, graph, coefficient, finite steps or checkpoint choice.'],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92),'/'):sha(p) for p in [Path(__file__),ROOT/'training/v148_relation_loss.py',
            ROOT/'training/v148_observed_relation.py',ROOT/'training/test_v148_observed_relation.py',OUT/'capacity.json',OUT/'functional_trials.jsonl']})
    save(target,result);print(json.dumps(dict(status=result['status'],qualification_passed=qualification,gradients=CE_gradients+aux_gradients,
        functional_parameter_evaluations=len(all_trials),new_fits=0,new_updates=0),ensure_ascii=False))


if __name__=='__main__':main()
