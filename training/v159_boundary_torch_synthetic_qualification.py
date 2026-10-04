"""Execute actual complete-head code on toy data only, with counted autograd."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,sha,check_bindings
from v159_current_input_boundary import CurrentInputBoundary,original_class_risk,clipped_probability_class_ce,flattened_class_gradient
from v159_mgda_synthetic_qualification import exact_two_gradient_direction

OUT=ROOT/'artifacts/v159_boundary_torch_synthetic_qualification_20261001'
COUNTS=dict(synthetic_head_constructions=0,synthetic_head_forward_calls=0,synthetic_class_risk_evaluations=0,synthetic_full_parameter_class_gradients=0,synthetic_probe_parameter_states=0)

def forward(model,x,z):
    COUNTS['synthetic_head_forward_calls']+=1
    return model(x,z)

def risk(lp,c,cls):
    COUNTS['synthetic_class_risk_evaluations']+=1
    return original_class_risk(lp,c,cls)

def gradient(r,model):
    COUNTS['synthetic_full_parameter_class_gradients']+=1
    return flattened_class_gradient(r,model)

def main():
    assert not OUT.exists();OUT.mkdir()
    sources=[Path(__file__).resolve(),ROOT/'training/v159_current_input_boundary.py',ROOT/'training/v159_mgda_synthetic_qualification.py',
        ROOT/'artifacts/v159_class_boundary_math_v2_20261001/qualification.json',
        ROOT/'artifacts/v159_saved_probability_numerical_audit_20261001/audit.json',
        ROOT/'artifacts/v159_zero_probability_gradient_qualification_20261001/qualification.json',
        ROOT/'docs/V159_NUMERICAL_INITIALIZATION_AND_GRADIENT_REVIEW.md']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    (OUT/'pre_synthetic_bindings.json').write_text(json.dumps(dict(status='bound_before_toy_head_construction_forward_and_autograd',source_sha256=bindings,python=sys.version,torch=torch.__version__),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    COUNTS['synthetic_head_constructions']+=1;model=CurrentInputBoundary(width=2,hidden=2)
    x=torch.tensor([[-1.,0.],[1.,.2]],dtype=torch.float64)
    member=torch.empty((16,3),dtype=torch.float64);member[:,0]=.05;member[:,1]=torch.linspace(.62,.80,16,dtype=torch.float64);member[:,2]=.95-member[:,1]
    z=member.log()[None,:,:].repeat(2,1,1)
    c=torch.tensor([[0.,9.,0.],[0.,0.,2.]],dtype=torch.float64)
    q,lp,delta=forward(model,x,z);base=z.softmax(-1).mean(1)
    assert torch.equal(q,base) and torch.equal(delta,torch.zeros_like(delta))
    initial=[risk(lp,c,cls) for cls in [1,2]];gs=[gradient(r,model) for r in initial]
    hidden_size=sum(p.numel() for p in list(model.parameters())[:-1])
    assert all(torch.count_nonzero(g[:hidden_size])==0 and torch.linalg.vector_norm(g[hidden_size:])>0 for g in gs)
    assert all(g.reshape(-1)[-6:].reshape(2,3)[:,0].abs().max()>0 for g in gs) # N remains in the denominator.
    alpha,d=exact_two_gradient_direction(*(g.detach().numpy() for g in gs))
    slopes=[float(g.numpy()@d) for g in gs];assert max(slopes)<0
    # One predetermined functional probe, not an official parameter update.
    original=[p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        cursor=0
        for p in model.parameters():
            p.add_(torch.from_numpy(d[cursor:cursor+p.numel()]).reshape_as(p)*(.01/np.linalg.norm(d)));cursor+=p.numel()
    COUNTS['synthetic_probe_parameter_states']+=1
    pq,plp,_=forward(model,x,z);probe=[float(risk(plp,c,k)) for k in [1,2]]
    assert all(a<b.item() for a,b in zip(probe,initial)) and pq[0].argmax()==1
    with torch.no_grad():
        for p,v in zip(model.parameters(),original):p.copy_(v)
        model.output_weight.copy_(torch.tensor([[.03,-.04,.02],[-.01,.02,-.03]],dtype=torch.float64))
    COUNTS['synthetic_probe_parameter_states']+=1
    _,lp,_=forward(model,x,z);gs_nonzero=[gradient(risk(lp,c,k),model) for k in [1,2]]
    fd=0.;step=1e-5
    for cls,g in zip([1,2],gs_nonzero):
        cursor=0
        for p in model.parameters():
            for j in range(p.numel()):
                v=float(p.detach().flatten()[j])
                with torch.no_grad():p.flatten()[j]=v+step
                _,pos,_=forward(model,x,z);rpos=float(risk(pos,c,cls))
                with torch.no_grad():p.flatten()[j]=v-step
                _,neg,_=forward(model,x,z);rneg=float(risk(neg,c,cls))
                with torch.no_grad():p.flatten()[j]=v
                fd=max(fd,abs((rpos-rneg)/(2*step)-float(g[cursor+j])))
            cursor+=p.numel()
    assert fd<1e-9
    dense_q,_,_=forward(model,x,z);sparse_q,_,_=forward(model,x.to_sparse_csr(),z)
    sparse_gap=float((dense_q-sparse_q).abs().max());assert sparse_gap<1e-14
    perm=torch.tensor([7,4,0,15,3,8,1,2,6,9,10,5,11,14,12,13]);permuted,_,_=forward(model,x,z[:,perm])
    permutation_gap=float((dense_q-permuted).abs().max());assert permutation_gap<1e-14
    with torch.no_grad():model.output_weight.zero_()
    huge=torch.tensor([0.,1000.,-1000.],dtype=torch.float64)[None,None,:].repeat(1,16,1)
    hc=torch.tensor([[0.,0.,5.]],dtype=torch.float64);hq,hlp,_=forward(model,x[:1],huge)
    hr=risk(hlp,hc,2);hg=gradient(hr,model)
    assert float(hr)==2000 and float(hq[0,2])==0 and torch.isfinite(hg).all() and torch.linalg.vector_norm(hg)>0
    historical=float(clipped_probability_class_ce(hq,hc,2));assert historical<691 and float(hr)>historical
    missing=False
    try:risk(hlp,hc,0)
    except ValueError:missing=True
    assert missing
    gpu=None
    if torch.cuda.is_available():
        model.to('cuda');gx=x.cuda();gz=z.cuda();gq,glp,_=forward(model,gx.to_sparse_csr(),gz)
        gpu_base=gz.softmax(-1).mean(1);gpu_zero_gap=float((gq-gpu_base).abs().max());assert torch.equal(gq,gpu_base)
        ggrad=[gradient(risk(glp,c.cuda(),k),model) for k in [1,2]]
        assert all(torch.isfinite(g).all() and torch.linalg.vector_norm(g)>0 for g in ggrad)
        uhq,uhlp,_=forward(model,gx[:1].to_sparse_csr(),huge.cuda());ug=gradient(risk(uhlp,hc.cuda(),2),model)
        assert float(uhq[0,2])==0 and torch.isfinite(ug).all() and torch.linalg.vector_norm(ug)>0
        gpu=dict(device=torch.cuda.get_device_name(0),toy_sparse_CUDA_origin_exact=True,origin_probability_gap=gpu_zero_gap,underflow_full_head_gradient_nonzero=True)
    check_bindings(bindings)
    result=dict(status='actual_candidate_torch_head_synthetic_autograd_qualified_not_official_runtime',latest_actual_training='V158',
        official_layout=dict(input=66287,hidden=16,opinion=11,members=16,outputs=3,parameters=1060832),toy_layout=dict(input=2,hidden=2,parameters=sum(p.numel() for p in model.parameters())),
        origin_mean_member_probability_exact=True,zero_hidden_gradient_expected=True,zero_output_gradient_nonzero=True,N_denominator_gradient_nonzero=True,
        full_parameter_class_gradient_finite_difference_max_error=fd,sparse_dense_probability_gap=sparse_gap,member_permutation_gap=permutation_gap,
        single_probe=dict(M_weight=alpha,slopes=slopes,class_risks_before=[float(r) for r in initial],class_risks_after=probe,initial_correct_M_retained=True),
        extreme_underflow=dict(stable_training_CE=float(hr),saved_probability_clipped_diagnostic_CE=historical,full_head_gradient_norm=float(torch.linalg.vector_norm(hg)),saved_true_probability=float(hq[0,2])),
        missing_class_rejected=missing,CUDA_toy=gpu,synthetic_work=COUNTS,
        official_classifier_calls=0,official_features_calls=0,official_gradient_calls=0,official_fits=0,official_parameter_updates=0,
        new_training_entry_registered=False,official_zero_step_replay=False,quality_acceptance=False,model_promoted=False,
        limits=['Toy CPU/CUDA execute the actual generic head, not official input population or imported training runtime.','Finite class risk descent does not prove classification learning or source transfer.','Step scale must be qualified against saved-logit range before budget activation.'],source_bindings_sha256=sha(OUT/'pre_synthetic_bindings.json'))
    (OUT/'qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
