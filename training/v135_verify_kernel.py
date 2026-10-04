"""Repeated real TRAIN forward/backward and old algebra parity, zero updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,time
import numpy as np
import torch
from v131_common import ROOT,load_data,fit_context
from v131_model import Classifier as Old,batch,objective,tensor_hash
from v135_model import Classifier as New
from experiment_review import sha

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    x,d=load_data();records=[];start=time.monotonic()
    for fold in range(3):
        fit,c,pure,totals,used=fit_context(d,fold)
        selections=[used[np.random.default_rng(10201+fold).permutation(len(used))][:256],
            np.unique(np.concatenate([used[pure[used]==0],used[(c[used,2]>0)&(pure[used]>0)][:32]]))]
        for si,ids in enumerate(selections):
            for focused in [False,True]:
                outputs={};gradients={};states={};identical=[]
                for name,cls in [('old',Old),('new',New)]:
                    torch.manual_seed(10201);m=cls(128).to('cuda');before=tensor_hash(m.state_dict())
                    results=[];grads=[]
                    for _ in range(2):
                        m.zero_grad(set_to_none=True);z=batch(m,x,ids)
                        loss,*_=objective(z,torch.as_tensor(c[ids],device='cuda',dtype=torch.float32),
                            torch.as_tensor(pure[ids],device='cuda'),float(c.sum()),totals,int(np.ceil(len(used)/256)),focused)
                        loss.backward();results.append(z.detach().cpu().numpy());grads.append({k:p.grad.detach().cpu().numpy().copy() for k,p in m.named_parameters()})
                    outputs[name]=results[0];gradients[name]=grads[0];states[name]=before
                    if name=='new':
                        identical=[np.array_equal(results[0],results[1]),all(np.array_equal(grads[0][k],grads[1][k]) for k in grads[0]),before==tensor_hash(m.state_dict())]
                    del m;gc.collect();torch.cuda.empty_cache()
                gap=float(np.abs(outputs['old']-outputs['new']).max());gg=max(float(np.abs(gradients['old'][k]-gradients['new'][k]).max()) for k in gradients['old'])
                records.append({'fold':fold,'selection':si,'focused':focused,'inputs':len(ids),
                    'old_new_logit_max_abs':gap,'old_new_gradient_max_abs':gg,
                    'initial_states_identical':states['old']==states['new'],
                    'new_forward_bitwise_identical':identical[0],'new_backward_bitwise_identical':identical[1],
                    'parameter_unchanged':identical[2],
                    'passed':gap<=2e-6 and gg<=2e-6 and states['old']==states['new'] and all(identical)})
                print(json.dumps(records[-1]),flush=True)
    path=ROOT/'artifacts/v135_kernel_qualification_20260930.json'
    if path.exists():raise FileExistsError(path)
    report={'all_checks_passed':all(z['passed'] for z in records),'records':records,'optimizer_updates':0,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [__import__('pathlib').Path(__file__),ROOT/'training/v135_model.py',ROOT/'training/v131_model.py']},
        'seconds':time.monotonic()-start,'limits':'Same hardware/software real-input repeated gradients, not model quality.'}
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not report['all_checks_passed']:raise ValueError('Kernel qualification failed')
if __name__=='__main__':main()
