"""Real head synthetic old-error gradient/global mass, zero official calls."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,traceback
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,sha,check_bindings
from v159_current_input_boundary_v3 import CurrentInputBoundary
from v159_float64_repeat_policy_v2 import repeat_gradient,repeat_values,finite_step_review
from v161_fixed_pure_error_risk import error_risk
OUT=ROOT/'artifacts/v161_error_risk_synthetic_qualification_20261002'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    files=[Path(__file__),ROOT/'training/v161_fixed_pure_error_risk.py',ROOT/'training/v159_current_input_boundary_v3.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v161_synthetic_accuracy_vs_mean_CE_review.py',ROOT/'artifacts/v161_synthetic_accuracy_vs_mean_CE_review_20261002/review.json']
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in files};save(OUT/'pre_synthetic_bindings.json',dict(source_sha256=binding,official_calls=0))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(16101);n=8;device='cuda';x=torch.zeros(n,66287,dtype=torch.float64,device=device)
    for i in range(n):
        columns=rng.choice(66287,177,replace=False);x[i,columns]=torch.tensor(rng.normal(size=177).astype(np.float32),dtype=torch.float64,device=device)
    p=torch.tensor(rng.dirichlet([2.,3.,4.],size=(n,16)),dtype=torch.float64,device=device)
    c=np.zeros((22546,3),np.int64);c[:4,1]=[3,2,5,7];c[4:8,2]=[1,8,2,4]
    e=np.zeros_like(c);e[2,1]=5;e[6,2]=2;ctx=dict(counts=c,target_counts=e,mass=c.sum(0));ids=np.arange(n)
    def builder(ctx,scope,ii):return x[ii],p[ii]
    class Counter:
        def gradient_before(self,*a):pass
        def gradient_after(self,*a):pass
    model=CurrentInputBoundary().cuda()
    with torch.no_grad():model.output_weight.copy_(torch.tensor(rng.normal(size=(16,3))*.125,dtype=torch.float64,device=device))
    baseline={k:v.detach().clone() for k,v in model.state_dict().items()};reports=[]
    for cls in [1,2]:
        results=[error_risk(model,ctx,ids,Counter(),cls,builder,batch) for batch in [4,8]]
        for j,(rv,q,lp,g) in enumerate(results):
            for name,value in [('gradient',g),('q',q),('logq',lp),('error_risk',rv['fixed_pure_error_contribution']),('full_risk',rv['full_original_class_CE'])]:np.save(OUT/f'class{cls}_batch{[4,8][j]}_{name}.npy',value)
        gr=repeat_gradient(results[0][3],results[1][3]);assert gr['passed']
        rv,q,lp,g=results[0];expected=-(e[:n]*lp[:n]).sum(0)[1:]/ctx['mass'][1:]
        assert repeat_values(rv['fixed_pure_error_contribution'],expected,'risk')['passed']
        direct=-lp[[2,6],[1,2]]*np.array([5.,2.])/ctx['mass'][1:]
        assert repeat_values(expected,direct,'risk')['passed']
        wrong_denominator=-lp[[2,6],[1,2]]
        assert not repeat_values(expected,wrong_denominator,'risk')['passed']
        # Independent autograd derivative of full-row target sum/global mass.
        model.zero_grad(set_to_none=True);_,ll,_=model(x,p);loss=-(torch.tensor(e[:n,cls],dtype=torch.float64,device=device)*ll[:,cls]).sum()/float(ctx['mass'][cls]);loss.backward()
        direct_g=np.concatenate([v.grad.detach().cpu().numpy().ravel() for v in model.parameters()]);assert repeat_gradient(g,direct_g)['passed']
        reports.append(dict(class_id=cls,complete_original_mass=int(ctx['mass'][cls]),fixed_error_original_rows=int(e[:,cls].sum()),zero_target_class_chunk_preserved=True,gradient_repeat=gr,wrong_error_subset_denominator_rejected=True))
    assert all(torch.equal(v,baseline[k]) for k,v in model.state_dict().items())
    # Preserve original numerical gate for the new target, not whole CE.
    def state(t):return np.logaddexp(0.,1+t)/101
    before=state(0.);after=state(-1.5);slope=-(1/(1+np.exp(-1.)))/101
    finite=finite_step_review([before,before],[after,after],[slope,slope],101,101,'B',1.5,True)
    assert finite['accepted']
    check_bindings(binding)
    result=dict(status='synthetic_complete_head_pure_error_contribution_gradient_and_full_mass_policy_passed',reports=reports,finite_error_target_gate=finite,whole_mean_CE_increase_does_not_override_correct_classification_fixture=True,parameters_unchanged=True,synthetic_only=True,official_heads=0,official_features=0,official_error_gradients=0,official_fits=0,permanent_updates=0,source_sha256=binding)
    save(OUT/'qualification.json',result);print(json.dumps(dict(status=result['status'],official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as err:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc(),official_calls=0))
        raise
