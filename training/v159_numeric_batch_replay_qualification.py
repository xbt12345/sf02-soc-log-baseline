"""Different row order/chunks and one fixed finite probe, synthetic inputs only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from experiment_review import ROOT,sha,read,check_bindings
from v159_current_input_boundary_v3 import CurrentInputBoundary
from v159_float64_repeat_policy_v2 import repeat_values,resolved_class_direction,finite_step_review,repeat_gradient,repeat_direction

OUT=ROOT/'artifacts/v159_numeric_batch_replay_qualification_20261002'
PREV=ROOT/'artifacts/v159_nonzero_real_dimension_numeric_qualification_v2_20261002'
calls=dict(synthetic_heads=0,synthetic_opinion_features=0,synthetic_manual_states=0,synthetic_manual_directional_probes=0,synthetic_gradients=0,official_heads=0,official_gradients=0,official_fits=0,official_updates=0)
def main():
    assert not OUT.exists();OUT.mkdir()
    files=[Path(__file__).resolve(),ROOT/'training/v159_current_input_boundary_v3.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_nonzero_real_dimension_numeric_qualification_v2.py',PREV/'qualification.json']+list(PREV.glob('*.npy'))
    diag=ROOT/'artifacts/v159_bound_gradient_repeat_diagnostic_20261002';files+=list(diag.glob('*_complete_gradient.npy'))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    (OUT/'pre_synthetic_bindings.json').write_text(json.dumps(dict(status='before_synthetic_reconstruction_or_any_new_forward',reverse_all_rows=True,batch=1024,fixed_synthetic_B_probe_step=.001,policy_change=False,source_sha256=bindings),indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(15902);n,width,k=18543,66287,177
    starts=rng.integers(0,width-k,n);indices=(starts[:,None]+np.arange(k)).astype(np.int32).ravel();values=rng.uniform(-2.,2.,n*k).astype(np.float32)
    x=csr_matrix((values,indices,np.arange(n+1,dtype=np.int32)*k),shape=(n,width));raw=rng.uniform(.03,1.,(n,16,3));p=raw/raw.sum(-1,keepdims=True)
    counts=np.zeros((n,3),np.int64);counts[np.arange(n),np.where(np.arange(n)%3==0,2,1)]=rng.integers(1,8,n);mass=counts.sum(0);template=torch.tensor(rng.normal(size=(16,3)),dtype=torch.float64,device='cuda')
    model=CurrentInputBoundary().cuda();model.register_forward_hook(lambda *a:calls.update(synthetic_heads=calls['synthetic_heads']+1));model.opinions.register_forward_hook(lambda *a:calls.update(synthetic_opinion_features=calls['synthetic_opinion_features']+1))
    def replay(ids,chunk):
        q=np.empty((n,3));lp=np.empty_like(q)
        with torch.no_grad():
            for start in range(0,n,chunk):
                ii=ids[start:start+chunk];s=x[ii]
                tx=torch.sparse_csr_tensor(torch.tensor(s.indptr,dtype=torch.int64,device='cuda'),torch.tensor(s.indices,dtype=torch.int64,device='cuda'),torch.tensor(s.data,dtype=torch.float64,device='cuda'),size=s.shape,device='cuda')
                qq,ll,_=model(tx,torch.tensor(p[ii],dtype=torch.float64,device='cuda'));q[ii]=qq.cpu().numpy();lp[ii]=ll.cpu().numpy()
        risks=(-(counts*lp).sum(0))[1:]/mass[1:]
        return q,lp,risks
    reviews=[]
    for scale in [0.,.125,8.]:
        with torch.no_grad():model.output_weight.copy_(scale*template)
        calls['synthetic_manual_states']+=1;qq,ll,risks=replay(np.arange(n-1,-1,-1),1024)
        reference={key:np.load(PREV/f'scale{scale}_rep0_class1_{key}.npy') for key in ['q','lp','risk']}
        metrics=dict(q=repeat_values(reference['q'],qq,'probability'),logp=repeat_values(reference['lp'],ll,'logp'),risk=repeat_values(reference['risk'],risks,'risk'))
        assert all(z['passed'] for z in metrics.values()),metrics
        reviews.append(dict(output_scale=scale,actual_different_batch_and_order=metrics))
    with torch.no_grad():model.output_weight.copy_(.125*template)
    gs=[np.load(PREV/f'scale0.125_rep0_class{c}_gradient.npy') for c in [1,2]];direction=resolved_class_direction(*gs,*mass[1:],'B');assert direction['status']=='direction_qualified_for_finite_guarded_proposal'
    with torch.no_grad():
        offset=0
        for parameter in model.parameters():
            parameter.add_(.001*torch.tensor(direction['direction'][offset:offset+parameter.numel()].reshape(parameter.shape),device='cuda'));offset+=parameter.numel()
    calls['synthetic_manual_directional_probes']+=1;qq,ll,risks=replay(np.arange(n),2048)
    base=np.load(PREV/'scale0.125_rep0_class1_risk.npy');finite=finite_step_review(base,risks,direction['class_slopes'],*mass[1:],'B',.001,True);assert finite['accepted'],finite
    np.save(OUT/'synthetic_fixed_directional_probe_risks.npy',risks)
    vectors=[[np.load(diag/f'repetition{r}_class{c}_complete_gradient.npy') for c in [1,2]] for r in [0,1]]
    real_gradient_reviews=[repeat_gradient(vectors[0][j],vectors[1][j]) for j in [0,1]];real_directions=[repeat_direction(*vectors,58840,33497,arm) for arm in ['A','B']]
    assert all(r['passed'] for r in real_gradient_reviews+real_directions) and all(r['descent_qualified'] for r in real_directions)
    assert calls['synthetic_heads']==calls['synthetic_opinion_features']==67
    check_bindings(bindings)
    result=dict(status='fixed_policy_actual_synthetic_batch_replays_and_finite_probe_plus_saved_real_vectors_passed',reviews=reviews,synthetic_real_dimension_finite_B_probe=finite,real_saved_gradient_reviews=real_gradient_reviews,real_saved_direction_reviews=real_directions,counts=calls,official_policy_active=False,first_training_issue_passed=False,quality_acceptance=False,source_sha256=bindings)
    (OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],counts=calls,finite_probe=finite)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),counts=calls),indent=2)+'\n',encoding='utf-8')
        raise
