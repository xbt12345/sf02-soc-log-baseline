"""Real-input repeated backward check; no optimizer and zero parameter updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import torch
import numpy as np
from v131_common import ROOT,load_data,fit_context
from v131_model import Classifier,batch,objective,tensor_hash

def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    x,d=load_data();fit,c,pure,totals,ids=fit_context(d,0)
    rng=np.random.default_rng(10201);ids=ids[rng.permutation(len(ids))][:256]
    result=[]
    for mode in [False,True]:
        torch.use_deterministic_algorithms(mode)
        torch.manual_seed(10201);m=Classifier(128).to('cuda');initial=tensor_hash(m.state_dict())
        grads=[];outputs=[];error=None
        try:
            for _ in range(4):
                m.zero_grad(set_to_none=True);z=batch(m,x,ids)
                loss,*_=objective(z,torch.as_tensor(c[ids],device='cuda',dtype=torch.float32),
                    torch.as_tensor(pure[ids],device='cuda'),float(c.sum()),totals,73,False)
                loss.backward();outputs.append(z.detach().cpu().numpy())
                grads.append({k:p.grad.detach().cpu().numpy().copy() for k,p in m.named_parameters()})
        except Exception as e:error=repr(e)
        delta={k:max(float(np.abs(g[k]-grads[0][k]).max()) for g in grads) for k in grads[0]} if grads else {}
        result.append({'deterministic':mode,'error':error,'max_gradient_difference':delta,
            'forward_identical':all(np.array_equal(outputs[0],z) for z in outputs),
            'parameter_unchanged':initial==tensor_hash(m.state_dict()),'optimizer_updates':0})
        del m;torch.cuda.empty_cache()
    target=ROOT/'artifacts/v135_repro_probe_20260930.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
