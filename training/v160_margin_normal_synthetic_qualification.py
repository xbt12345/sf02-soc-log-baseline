"""Actual unchanged head, synthetic inputs only; no project arrays or labels."""
import json,traceback,math
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from experiment_review import ROOT,sha,check_bindings
from v159_current_input_boundary_v3 import CurrentInputBoundary
from v159_float64_repeat_policy_v2 import repeat_gradient,repeat_values,SEGMENTS
from v160_margin_normal import measure,input_identity

OUT=ROOT/'artifacts/v160_margin_normal_synthetic_qualification_20261002'
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    files=[Path(__file__),ROOT/'training/v160_margin_normal.py',ROOT/'training/v159_current_input_boundary_v3.py',ROOT/'training/v159_current_input_boundary.py',ROOT/'training/v159_float64_repeat_policy_v2.py']
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in files};save(OUT/'pre_synthetic_bindings.json',dict(source_sha256=binding,official_calls=0))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(16001);n=8;width=66287
    indices=np.concatenate([np.sort(rng.choice(width,177,replace=False)) for _ in range(n)])
    values=rng.normal(size=len(indices)).astype(np.float32);x=csr_matrix((values,indices,np.arange(n+1)*177),shape=(n,width));x.sort_indices()
    p=rng.dirichlet([2.,3.,4.],size=(n,16)).astype(np.float64);ids=np.arange(n,dtype=np.int64)
    device='cuda';xx=torch.sparse_csr_tensor(torch.tensor(x.indptr,device=device),torch.tensor(x.indices,device=device),torch.tensor(x.data,dtype=torch.float64,device=device),size=x.shape,device=device);pp=torch.tensor(p,device=device)
    model=CurrentInputBoundary().cuda()
    with torch.no_grad():model.output_weight.copy_(torch.tensor(rng.normal(size=(16,3))*.125,dtype=torch.float64,device=device))
    base=tuple(v.detach().clone() for v in model.parameters());reports=[]
    for truth,rival in [(1,0),(1,2),(2,0),(2,1)]:
        measurements=[measure(model,xx,pp,2,truth,rival) for _ in range(2)]
        for k,r in enumerate(measurements):
            for name,value in r.items():np.save(OUT/f'truth{truth}_rival{rival}_repeat{k}_{name}.npy',value)
        gr=repeat_gradient(measurements[0]['gradient'],measurements[1]['gradient']);assert gr['passed']
        qr=repeat_values(measurements[0]['q'],measurements[1]['q'],'probability');lr=repeat_values(measurements[0]['logq'],measurements[1]['logq'],'log_probability');assert qr['passed'] and lr['passed']
        g=measurements[0]['gradient'];direction=np.zeros_like(g)
        # Excite all four real parameter groups, not only the zero-init output.
        for _,start,end in SEGMENTS:
            segment=g[start:end];assert np.count_nonzero(segment)>0
            direction[start:end]=segment/max(float(np.abs(segment).max()),np.finfo(float).tiny)
        analytic=math.fsum(g*direction);step=2**-18;observed=[]
        for sign in [1.,-1.]:
            with torch.no_grad():
                offset=0
                for v,b in zip(model.parameters(),base):
                    v.copy_(b+sign*step*torch.tensor(direction[offset:offset+v.numel()].reshape(v.shape),device=device));offset+=v.numel()
                _,lp,_=model(xx,pp);observed.append(float((lp[2,truth]-lp[2,rival]).cpu()))
        with torch.no_grad():
            for v,b in zip(model.parameters(),base):v.copy_(b)
        fd=(observed[0]-observed[1])/(2*step);relative=abs(fd-analytic)/abs(analytic);assert relative<1e-5
        reports.append(dict(truth=truth,rival=rival,gradient_repeat=gr,probability_repeat=qr,log_probability_repeat=lr,analytic_slope=analytic,central_difference=fd,FD_relative_error=relative,FD_scope='synthetic Jacobian correctness only, not official numerical acceptance'))
    identity=input_identity(0,'OOF',ids,x,p,2,1,2);assert identity==input_identity(0,'OOF',ids,x,p,2,1,2)
    variants=[]
    for role,scope,truth,rival in [(1,'OOF',1,2),(0,'deployment',1,2),(0,'OOF',2,1),(0,'OOF',1,0)]:variants.append(input_identity(role,scope,ids,x,p,2,truth,rival))
    variants.append(input_identity(0,'OOF',ids,x,p[:,::-1].copy(),2,1,2))
    changed=x.copy();changed.data[0]=np.nextafter(changed.data[0],np.float32(np.inf));variants.append(input_identity(0,'OOF',ids,changed,p,2,1,2))
    changed_ids=ids.copy();changed_ids[0]+=1;variants.append(input_identity(0,'OOF',changed_ids,x,p,2,1,2))
    assert identity not in variants and len(set(variants))==len(variants)
    rejected=0
    for truth,rival in [(1,1),(0,1),(2,3)]:
        try:measure(model,xx,pp,2,truth,rival)
        except ValueError:rejected+=1
    assert rejected==3 and all(torch.equal(v,b) for v,b in zip(model.parameters(),base))
    check_bindings(binding)
    save(OUT/'qualification.json',dict(status='unchanged_real_head_synthetic_margin_Jacobian_repeat_and_input_identity_passed',reports=reports,identity_distinct_variants=len(variants),invalid_queries_rejected=rejected,complete_parameter_width=1060832,parameters_restored=True,synthetic_heads=16,synthetic_margin_gradients=8,official_heads=0,official_features=0,official_class_gradients=0,official_margin_gradients=0,official_fits=0,official_updates=0,finite_SOC_qualification=False,source_sha256=binding))
    print(json.dumps(dict(status='synthetic_margin_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_calls=0))
        raise
