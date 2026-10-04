"""Fixed legal TRAIN differential probe; no parameter mutation or fitting."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json
import numpy as np
import pandas as pd
import torch
from v142_runtime import ROOT,OUT as RUN,read,save,sha,require_run_seal
from v142_train import tensors
from v138_train import configure
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v138_readout import full_gradient,probabilities
from experiment_review import check_bindings
from v142_retention_check import check
from v144_pair_objective import build_pairs,gradients,AUX_GRADIENT_RATIO,TEMPERATURE

OUT=ROOT/'artifacts/v144_aux_gradient_evidence_20261001'


def norm(g):return float(sum(v.square().sum() for v in g).sqrt())


def main():
    if OUT.exists():raise FileExistsError('No diagnostic overwrite or unregistered search')
    require_run_seal(ROOT/'training/v142_train.py');configure();_,d=load_data()
    trace=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    facts=pd.read_parquet(trace,columns=['facts_json']).facts_json
    binding={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'training/v144_aux_gradient_probe.py',ROOT/'training/v144_pair_objective.py',trace,RUN/'verified_TRAIN_mastery_registry.json']}
    for f in range(3):binding[str((RUN/f'fold{f}_S2/endpoint.pt').relative_to(ROOT))]=sha(RUN/f'fold{f}_S2/endpoint.pt')
    OUT.mkdir();save(OUT/'pre_registered_probe.json',dict(status='registered_before_model_gradients',folds=[0,1,2],new_fits_max=0,new_parameter_updates_max=0,
        temperature=TEMPERATURE,auxiliary_gradient_ratio=AUX_GRADIENT_RATIO,coefficient_source='fixed 0.1 norm ratio from legal full-frequency TRAIN CE and restricted auxiliary gradients, no grid or held selection',
        full_CE_gradients_max=3,auxiliary_gradients_max=6,jvp_directions=['unit_auxiliary_descent','unit_combined_CE_plus_fixed_ratio_descent'],jvp_chunk=512,
        source_sha256=binding,scope='Derivative qualification only, not a registered training objective or proof of support/transfer.'))
    check_bindings(binding);reports=[];allrows=[];replayed=[];jvps=0
    for f in range(3):
        frame,c,pure,ids,p,capacity=build_pairs(d,facts,f);h,ff,m=tensors(f)
        st=torch.load(RUN/f'fold{f}_S2/endpoint.pt',map_location='cpu',weights_only=True);m.load_state_dict(st['state']);identity=tensor_hash(m.state_dict())
        q=probabilities(m,h,ff,np.arange(len(c)))
        if not np.array_equal(q,np.load(RUN/f'fold{f}_S2/sealed_all_prob.npy')):raise ValueError('Actual frozen model differs before probe')
        ce,mass=full_gradient(m,h,ff,torch.as_tensor(c,device='cuda',dtype=torch.float64),ids);cg=tuple(v.grad.detach().clone() for v in m.parameters())
        aux,ag=gradients(m,h,p);repeat,rg=gradients(m,h,p)
        if aux!=repeat or any(not torch.equal(a,b) for a,b in zip(ag,rg)):raise ValueError('Restricted auxiliary gradient does not repeat exactly')
        cn,an=norm(cg),norm(ag)
        if min(cn,an)<=1e-15:raise ValueError('Zero gradient prevents meaningful fixed norm-ratio probe')
        coeff=AUX_GRADIENT_RATIO*cn/an;combined=tuple(a+coeff*b for a,b in zip(cg,ag));combined_norm=norm(combined)
        directions={'aux':tuple(-a/an for a in ag),'combined':tuple(-a/combined_norm for a in combined)}
        params=tuple(m.parameters());names=[n for n,_ in m.named_parameters()];q_replay=[];derivs={name:[] for name in directions};ce_derivs={name:np.zeros(3) for name in directions}
        for start in range(0,len(c),512):
            take=np.arange(start,min(start+512,len(c)))
            def function(*values):return torch.func.functional_call(m,dict(zip(names,values)),(h[take],ff[take]))
            for name,dir_ in directions.items():
                z,dz=torch.func.jvp(function,params,dir_);jvps+=1
                prob=torch.softmax(z,-1);dlogp=dz-(prob*dz).sum(-1,keepdim=True);dq=(prob*dlogp).mean(1)
                if name=='aux':q_replay.append(prob.mean(1).detach().cpu().numpy())
                derivs[name].append(dq.detach().cpu().numpy())
                ce_derivs[name]+=(-dlogp.mean(1).detach().cpu().numpy()*c[take]).sum(0)/np.maximum(c.sum(0),1)
            if start%10240==0:print(json.dumps(dict(stage='TRAIN_only_JVP',fold=f,numeric_inputs_through=int(take[-1]+1))),flush=True)
        actual=np.concatenate(q_replay)
        if not np.array_equal(actual.argmax(1),q.argmax(1)) or np.max(np.abs(actual-q))>2e-6:raise ValueError('Functional JVP evaluation changed base probabilities')
        row=frame[['row_position','local','root','truth','canonical_key']].copy();row['training_role']=f;row['pred']=q[frame.local].argmax(1);row['pure']=pure[frame.local].astype(bool)
        loc=row.local.to_numpy();y=row.truth.to_numpy();other=q[loc].copy();other[np.arange(len(row)),y]=-1;alt=other.argmax(1)
        margin=q[loc,y]-q[loc,alt];row['base_true_margin']=margin
        summaries=[]
        for name,arrays in derivs.items():
            dq=np.concatenate(arrays);dm=dq[loc,y]-dq[loc,alt];row[name+'_margin_derivative']=dm
            radius=np.where((margin>0)&(dm<0),margin/np.maximum(-dm,1e-300),np.inf);row[name+'_linear_guard_radius']=radius
            s=[]
            for cl in [1,2]:
                mask=(y==cl)&row.pure.to_numpy();valid=(margin>0)&(y==cl)
                s.append(dict(truth=cl,pure_original_rows=int(mask.sum()),positive_pure_margin_slopes=int((dm[mask]>0).sum()),negative_pure_margin_slopes=int((dm[mask]<0).sum()),
                    full_original_member_CE_derivative=float(ce_derivs[name][cl]),
                    mean_pure_original_margin_derivative=float(dm[mask].mean()),minimum_original_correct_margin=float(margin[valid].min()),
                    linearized_minimum_correct_margin_radius=None if not np.isfinite(radius[valid]).any() else float(radius[valid].min())))
            summaries.append(dict(direction=name,classes=s))
        replayed.append(row[['row_position','training_role','truth','pred']]);allrows.append(row)
        if tensor_hash(m.state_dict())!=identity:raise ValueError('Diagnostic modified parameter/buffer state')
        reports.append(dict(fold=f,capacity=capacity,original_member_CE=ce,original_class_mass_seen=mass,auxiliary_loss=aux,gradients_repeat_exact=True,
            full_CE_gradient_norm=cn,auxiliary_gradient_norm=an,fixed_initial_auxiliary_coefficient=coeff,
            CE_auxiliary_gradient_cosine=float(sum((a*b).sum() for a,b in zip(cg,ag)))/(cn*an),
            combined_full_CE_descent_derivative=-float(sum((a*b).sum() for a,b in zip(cg,combined)))/combined_norm,
            margin_differentials=summaries,parameter_identity_unchanged=True,parameter_sha256=identity))
        print(json.dumps(dict(stage='probe_role_complete',report=reports[-1]),ensure_ascii=False),flush=True)
        del m,h,ff,params,cg,ag,rg,combined,directions;gc.collect();torch.cuda.empty_cache()
    guard=check(pd.concat(replayed,ignore_index=True))
    if not guard['passed']:raise ValueError('Baseline classification protection contradicted')
    table=pd.concat(allrows,ignore_index=True);table.to_parquet(OUT/'legal_TRAIN_margin_differentials.parquet',index=False)
    check_bindings(binding)
    report=dict(status='TRAIN_only_auxiliary_gradient_probe_completed',latest_actual_training='V142',new_fits=0,new_updates=0,
        full_CE_gradient_evaluations=3,full_auxiliary_gradient_evaluations=6,numeric_chunk_JVP_evaluations=jvps,
        auxiliary_temperature=TEMPERATURE,auxiliary_gradient_ratio=AUX_GRADIENT_RATIO,folds=reports,baseline_mastery_guard=guard,
        baseline_all_correct_protected_original_role_rows=225558,second_issue_solved=False,third_issue_solved=False,new_fit_registered=False,
        evidence_bindings=binding,output_sha256=sha(OUT/'legal_TRAIN_margin_differentials.parquet'),
        limitations=['Local derivatives are not finite-update classification protection or source transfer.',
        'Existing same-context positives do not create unseen fine behavior/category evidence; known 578 S errors still have no direct supported-pair-key coverage.',
        'Root prototype/class/key weighting is an explicit auxiliary choice, not a claim to preserve original CE weights in the auxiliary term.',
        'HELD labels/errors never enter pair construction, coefficient or derivatives; no coefficient scan or model selection.',
        'No parameter update, optimizer step or new model was produced; all original TRAIN classification rows remain intact.'])
    save(OUT/'probe.json',report);print('TRAIN auxiliary differential evidence complete; no fit, no update',flush=True)


if __name__=='__main__':main()
