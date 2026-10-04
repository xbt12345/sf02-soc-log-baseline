"""CPU toy of function-preserving gradient activation; no official data/model."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v160_synthetic_centered_initialization_review_20261002'
def main():
    assert not OUT.exists();torch.set_num_threads(2)
    reports=[];arrays={}
    for seed in [16001,16002,16003]:
        gen=torch.Generator().manual_seed(seed)
        x=torch.randn(257,64,generator=gen,dtype=torch.float64)/8
        p=torch.softmax(torch.randn(257,16,3,generator=gen,dtype=torch.float64),-1)
        truth=torch.arange(257)%2+1
        ent=-(p*p.log()).sum(-1,keepdim=True);mean=p.mean(1)
        common=torch.cat([mean,p.var(1,unbiased=False),ent.mean(1)],-1)
        features=torch.cat([p,common[:,None,:].expand(-1,16,-1),ent],-1)
        w0=torch.randn(64,16,generator=gen,dtype=torch.float64)/8
        u0=torch.randn(11,16,generator=gen,dtype=torch.float64)/math.sqrt(11)
        b0=torch.zeros(16,dtype=torch.float64)
        v0=torch.randn(16,3,generator=gen,dtype=torch.float64)/4
        old=[w0.clone().requires_grad_(),u0.clone().requires_grad_(),b0.clone().requires_grad_(),torch.zeros_like(v0,requires_grad=True)]
        new=[torch.zeros_like(t,requires_grad=True) for t in [w0,u0,b0,v0]]
        oldh=torch.tanh((x@old[0])[:,None,:]+features@old[1]+old[2])
        olddelta=(oldh@old[3]).mean(1)
        shared=(x@w0)[:,None,:]+features@u0+b0
        h0=torch.tanh(shared)
        h=torch.tanh(shared+(x@new[0])[:,None,:]+features@new[1]+new[2])
        newdelta=((h-h0)@v0+h@new[3]).mean(1)
        assert torch.equal(olddelta,newdelta) and torch.count_nonzero(newdelta)==0
        oldlp=torch.log_softmax(mean.log()+olddelta,-1)
        newlp=torch.log_softmax(mean.log()+newdelta,-1)
        assert torch.equal(oldlp,newlp)
        classes=[]
        for c in [1,2]:
            oldg=torch.autograd.grad(-oldlp[truth==c,c].mean(),old,retain_graph=True)
            newg=torch.autograd.grad(-newlp[truth==c,c].mean(),new,retain_graph=True)
            assert all(torch.count_nonzero(t)==0 for t in oldg[:3])
            assert all(torch.linalg.vector_norm(t)>0 for t in newg)
            of=np.concatenate([t.detach().numpy().ravel() for t in oldg])
            nf=np.concatenate([t.detach().numpy().ravel() for t in newg])
            arrays[f'seed{seed}_class{c}_old_gradient']=of
            arrays[f'seed{seed}_class{c}_centered_gradient']=nf
            classes.append(dict(class_id=c,old_segment_L2=[float(torch.linalg.vector_norm(t)) for t in oldg],centered_segment_L2=[float(torch.linalg.vector_norm(t)) for t in newg],old_nonzero_parameters=int(np.count_nonzero(of)),centered_nonzero_parameters=int(np.count_nonzero(nf))))
        reports.append(dict(seed=seed,initial_delta_bitwise_equal=True,initial_log_probability_bitwise_equal=True,classes=classes))
    report=dict(status='CPU_synthetic_centered_residual_preserves_origin_and_activates_all_gradient_segments',reports=reports,
      synthetic_rows=257,synthetic_input_width=64,hidden_width=16,members=16,
      official_data_used=False,official_classifier_calls=0,official_features=0,official_gradients=0,official_fits=0,official_updates=0,
      conclusion='Algebra and toy autograd only. Does not establish SOC-safe finite updates, convergence, classification gains or cross-source transfer.',
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.mkdir()
    for key,array in arrays.items():np.save(OUT/(key+'.npy'),array)
    (OUT/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','official_data_used','official_classifier_calls','official_fits','conclusion']},ensure_ascii=False))
if __name__=='__main__':main()
