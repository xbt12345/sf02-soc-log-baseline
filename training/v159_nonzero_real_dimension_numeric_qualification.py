"""Predetermined synthetic full-width CSR nonzero-state numeric tests, no gold."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from experiment_review import ROOT,sha,read,check_bindings
from v159_current_input_boundary_v3 import CurrentInputBoundary
from v159_float64_repeat_policy import repeat_values,repeat_gradient,repeat_direction,finite_step_review,EPS

OUT=ROOT/'artifacts/v159_nonzero_real_dimension_numeric_qualification_20261002'
COUNTS=dict(synthetic_head_attempts=0,synthetic_head_completed=0,synthetic_feature_attempts=0,synthetic_feature_completed=0,synthetic_complete_class_gradients=0,synthetic_manual_states=0,official_heads=0,official_features=0,official_gradients=0,official_fits=0,official_updates=0)

def main():
    assert not OUT.exists();OUT.mkdir();t=time.monotonic()
    files=[Path(__file__).resolve(),ROOT/'training/v159_float64_repeat_policy.py',ROOT/'training/v159_class_direction.py',ROOT/'training/v159_current_input_boundary_v3.py',ROOT/'training/v159_current_input_boundary.py',ROOT/'training/v159_mgda_synthetic_qualification.py',ROOT/'artifacts/v159_gradient_diagnostic_saved_vector_review_v2_20261002/review.json',ROOT/'artifacts/v159_independent_gradient_repeat_diagnostic_audit_20261002/audit.json']
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    specification=dict(status='registered_before_synthetic_data_and_any_head_construction',shape=[18543,66287],hidden=16,chunk=2048,nnz_per_row=177,seed=15902,manual_output_scales=[0.,.125,8.],repeat_eps=8,finite_step_eps=16,parameter_segment_bound_and_relative_L2=True,gradient_absolute_floor=0,per_row_argmax_exact=True,official_calls=0,source_sha256=binding)
    (OUT/'pre_synthetic_bindings.json').write_text(json.dumps(specification,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(15902);n,width=18543,66287;k=177
    # Full-width distinct sorted columns; all values are synthetic float32.
    starts=rng.integers(0,width-k,n);indices=(starts[:,None]+np.arange(k)).astype(np.int32).ravel()
    values=rng.uniform(-2.,2.,n*k).astype(np.float32);x=csr_matrix((values,indices,np.arange(n+1,dtype=np.int32)*k),shape=(n,width))
    raw=rng.uniform(.03,1.,(n,16,3));prob=raw/raw.sum(-1,keepdims=True)
    labels=np.arange(n)%3==0;counts=np.zeros((n,3),np.int64);counts[np.arange(n),np.where(labels,2,1)]=rng.integers(1,8,n);mass=counts.sum(0)
    model=CurrentInputBoundary().cuda();journal=(OUT/'synthetic_calls.jsonl').open('w',encoding='utf-8',buffering=1)
    def event(kind,phase):
        key='synthetic_'+kind+'_'+phase;COUNTS[key]+=1;journal.write(json.dumps(dict(kind=kind,phase=phase,ordinal=COUNTS[key]))+'\n')
    hooks=[model.register_forward_pre_hook(lambda *a:event('head','attempts')),model.register_forward_hook(lambda *a:event('head','completed')),model.opinions.register_forward_pre_hook(lambda *a:event('feature','attempts')),model.opinions.register_forward_hook(lambda *a:event('feature','completed'))]
    def evaluate(cls):
        model.zero_grad(set_to_none=True);q=np.empty((n,3));lp=np.empty_like(q);num=np.zeros(3)
        for start in range(0,n,2048):
            end=min(start+2048,n);s=x[start:end]
            tx=torch.sparse_csr_tensor(torch.tensor(s.indptr,device='cuda',dtype=torch.int64),torch.tensor(s.indices,device='cuda',dtype=torch.int64),torch.tensor(s.data,device='cuda',dtype=torch.float64),size=s.shape,device='cuda')
            p=torch.tensor(prob[start:end],dtype=torch.float64,device='cuda');c=torch.tensor(counts[start:end],dtype=torch.float64,device='cuda')
            qq,ll,_=model(tx,p);loss=-(c[:,cls]*ll[:,cls]).sum()/float(mass[cls]);loss.backward()
            q[start:end]=qq.detach().cpu().numpy();lp[start:end]=ll.detach().cpu().numpy();num+=(-(c*ll).sum(0)).detach().cpu().numpy()
        g=np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()]);COUNTS['synthetic_complete_class_gradients']+=1
        return dict(q=q,lp=lp,risk=num[1:]/mass[1:],gradient=g)
    states=[];output_template=torch.tensor(rng.normal(size=(16,3)),dtype=torch.float64,device='cuda')
    for scale in specification['manual_output_scales']:
        with torch.no_grad():model.output_weight.copy_(scale*output_template)
        COUNTS['synthetic_manual_states']+=1;reps=[]
        for rep in range(2):
            current=[]
            for cls in [1,2]:
                result=evaluate(cls);prefix=OUT/f'scale{scale}_rep{rep}_class{cls}'
                for key,value in result.items():np.save(str(prefix)+'_'+key+'.npy',value)
                current.append(result)
            reps.append(current)
        reference=reps[0][0];values_review=[]
        for repetition in reps:
            for r in repetition:
                values_review.append({key:repeat_values(reference[key],r[key],'probability' if key=='q' else key) for key in ['q','lp','risk']})
        grad=[repeat_gradient(reps[0][j]['gradient'],reps[1][j]['gradient']) for j in [0,1]]
        dirs=[repeat_direction([r['gradient'] for r in reps[0]],[r['gradient'] for r in reps[1]],int(mass[1]),int(mass[2]),arm) for arm in ['A','B']]
        report=dict(output_scale=scale,complete_parameter_gradient_nonzero=bool(all(np.linalg.norm(r['gradient'])>0 for r in reps[0])),hidden_gradient_nonzero=bool(all(np.linalg.norm(r['gradient'][:-48])>0 for r in reps[0])),same_point_values=values_review,gradient_comparisons=grad,direction_comparisons=dirs)
        states.append(report);(OUT/f'scale{scale}_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        assert all(v['passed'] for r in values_review for v in r.values()) and all(g['passed'] for g in grad) and all(d['passed'] for d in dirs),report
        assert report['complete_parameter_gradient_nonzero'] and (scale==0 or report['hidden_gradient_nonzero'])
        print(json.dumps(dict(event='synthetic_real_dimension_repeat_passed',output_scale=scale,head_calls=COUNTS['synthetic_head_attempts'],gradients=COUNTS['synthetic_complete_class_gradients'])),flush=True)
    # Deterministic policy rejection fixtures, no extra head or gradient calls.
    fixtures={}
    g=np.zeros(1060832);g[0]=1e-30;bad=g.copy();bad[0]+=1e-20
    fixtures['tiny_gradient_not_swallowed_by_absolute_floor']=not repeat_gradient(g,bad)['passed']
    near=np.zeros_like(g);near[-1]=1.;near_bad=near.copy();near_bad[0]=1e-25
    fixtures['zero_hidden_segment_must_remain_exact_zero']=not repeat_gradient(near,near_bad)['passed']
    nan=g.copy();nan[-1]=np.nan;fixtures['nonfinite_gradient_rejected']=not repeat_gradient(g,nan)['passed']
    fixtures['probability_argmax_flip_rejected_within_float_envelope']=not repeat_values([[.5,.5-EPS,0]],[[.5-EPS,.5,0]],'probability')['passed']
    fixtures['risk_difference_over_envelope_rejected']=not repeat_values([1.,2.],[1.+64*EPS,2.])['passed']
    for arm in ['A','B']:
        fixtures[arm+'_unchanged_risks_rejected']=not finite_step_review([1.,2.],[1.,2.],[-1.,-1.],2,1,arm,2**-40,True)['accepted']
        fixtures[arm+'_subresolution_drop_rejected']=not finite_step_review([1.,2.],[1.-EPS,2.-EPS],[-1.,-1.],2,1,arm,2**-40,True)['accepted']
        fixtures[arm+'_actual_finite_descent_accepted']=finite_step_review([1.,2.],[.9,1.9],[-1.,-1.],2,1,arm,1.,True)['accepted']
        fixtures[arm+'_classification_regression_rejected']=not finite_step_review([1.,2.],[.9,1.9],[-1.,-1.],2,1,arm,1.,False)['accepted']
        fixtures[arm+'_Armijo_shortfall_rejected']=not finite_step_review([1.,2.],[1.-1e-5,2.-1e-5],[-1.,-1.],2,1,arm,1.,True)['accepted']
    fixtures['B_class_sacrifice_rejected']=not finite_step_review([1.,2.],[.1,2.001],[-1.,-1.],99,1,'B',1.,True)['accepted']
    assert all(fixtures.values()),fixtures
    for h in hooks:h.remove()
    journal.close();assert COUNTS['synthetic_head_attempts']==COUNTS['synthetic_head_completed']==COUNTS['synthetic_feature_attempts']==COUNTS['synthetic_feature_completed']==120 and COUNTS['synthetic_complete_class_gradients']==12
    check_bindings(binding)
    summary=dict(status='fixed_numeric_policy_passed_synthetic_full_dimension_nonzero_states_and_rejection_cases',specification=specification,states=states,fixtures=fixtures,counts=COUNTS,elapsed_seconds=time.monotonic()-t,policy_active_for_official_calls=False,first_training_issue_passed=False,quality_acceptance=False,source_sha256=binding)
    (OUT/'qualification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=summary['status'],counts=COUNTS,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),counts=COUNTS),indent=2)+'\n',encoding='utf-8')
        raise
