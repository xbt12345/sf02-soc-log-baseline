"""Frozen V107 member/gradient diagnosis; no optimizer and zero classifier fits."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

from run_v75 import ROOT, save, sha
from v75_views import BYTE_FEATURES
from v104_phase_b import SparseTabM, csr_tensor, DEVICE, K
from v108_root_evidence_audit import PREV, DEST as OLD, N1

DEST = ROOT/'artifacts/v109_plan_review_20260929'


def main():
    assert not DEST.exists(); DEST.mkdir()
    d=pd.read_parquet(OLD/'support_and_error_ledger.parquet')
    y=d.truth.to_numpy(); local=d.local.to_numpy(); fold=d.fold.to_numpy()
    all_results=[];receipts={};start=time.monotonic()
    for view,path in [('N1',N1),('N2',PREV/'N2_ASA.npz')]:
        x=sparse.load_npz(path);receipts[path.relative_to(ROOT).as_posix()]=sha(path)
        for k in range(3):
            folder=PREV/f'fold{k}_{view}_TabM25_seed10201'
            fit=json.loads((folder/'fit.json').read_text(encoding='utf-8'))
            assert sha(folder/'model.pt')==fit['model_sha256']
            assert sha(folder/'ASA_input_prob.npy')==fit['prob_sha256']
            receipts[(folder/'model.pt').relative_to(ROOT).as_posix()]=fit['model_sha256']
            receipts[(folder/'ASA_input_prob.npy').relative_to(ROOT).as_posix()]=fit['prob_sha256']
            model=SparseTabM().to(DEVICE)
            state=torch.load(folder/'model.pt',map_location=DEVICE,weights_only=False)
            model.load_state_dict(state['state_dict']);model.eval()
            train=fold!=k
            counts=np.bincount(local[train]*3+y[train],minlength=x.shape[0]*3).reshape(-1,3)
            n=int(counts.sum());out=[]; replay_max=0.; derivative_checked=False
            captured={}
            hook=model.second.register_forward_hook(lambda module,args,value: captured.update(h2=torch.relu(value)))
            grads={c:[np.zeros((K,128,3)),np.zeros((K,3)),np.zeros((x.shape[1]-BYTE_FEATURES,3))] for c in (1,2)}
            with torch.inference_mode():
                for s in range(0,x.shape[0],512):
                    xx=x[s:s+512];sp=csr_tensor(xx,DEVICE)
                    f=torch.as_tensor(xx[:,BYTE_FEATURES:].toarray(),device=DEVICE)
                    z=model(sp,f);h2=captured['h2']
                    reconstructed=model.head(h2)+model.facts_direct(f)[:,None,:]
                    replay_max=max(replay_max,float((z-reconstructed).abs().max()))
                    assert torch.equal(z,reconstructed)
                    prob=torch.softmax(z,-1)
                    out.append(z.cpu().numpy())
                    if not derivative_checked:
                        # Independent autograd check on the actual frozen activations.
                        with torch.inference_mode(False), torch.enable_grad():
                            hc=h2[:8].detach().clone().double();fc=f[:8].detach().clone().double()
                            wh=model.head.weight.detach().clone().double().requires_grad_()
                            bh=model.head.bias.detach().clone().double().requires_grad_()
                            wf=model.facts_direct.weight.detach().clone().double().T.requires_grad_()
                            zz=torch.einsum('bki,kic->bkc',hc,wh)+bh+ (fc@wf)[:,None,:]
                            mass=torch.arange(1,9,device=DEVICE,dtype=torch.float64)
                            cc=2;rr=(zz.softmax(-1)-torch.eye(3,device=DEVICE,dtype=torch.float64)[cc])*mass[:,None,None]/(n*K)
                            analytical=(torch.einsum('bki,bkc->kic',hc,rr),rr.sum(0),fc.T@rr.sum(1))
                            loss=(-zz.log_softmax(-1)[:,:,cc]*mass[:,None]).sum()/(n*K)
                            automatic=torch.autograd.grad(loss,(wh,bh,wf))
                            assert all(torch.allclose(a,b,atol=1e-12,rtol=1e-8) for a,b in zip(analytical,automatic))
                        derivative_checked=True
                    # Exact full-training CE derivatives restricted to the existing
                    # head and direct-fact parameters, grouped by true class.
                    pp=prob.double();hh=h2.double();ff=f.double()
                    for c in (1,2):
                        mass=torch.as_tensor(counts[s:s+xx.shape[0],c],device=DEVICE,dtype=torch.float64)
                        one=torch.zeros(3,device=DEVICE,dtype=torch.float64);one[c]=1
                        rr=(pp-one)*mass[:,None,None]/(n*K)
                        grads[c][0]+=torch.einsum('bki,bkc->kic',hh,rr).cpu().numpy()
                        grads[c][1]+=rr.sum(0).cpu().numpy()
                        grads[c][2]+=(ff.T@rr.sum(1)).cpu().numpy()
            hook.remove();logits=np.concatenate(out)
            t=torch.as_tensor(logits)
            pp=torch.softmax(t,-1).numpy()
            avg=pp.mean(1)
            saved=np.load(folder/'ASA_input_prob.npy')
            assert np.allclose(avg,saved,atol=2e-6,rtol=2e-6)
            assert np.array_equal(avg.argmax(1),saved.argmax(1))
            np.save(DEST/f'{view}_fold{k}_member_logits.npy',logits)
            row=pp[local];pred=avg[local].argmax(1);member=row.argmax(2)
            meanlogit=logits.mean(1)[local].argmax(1)
            details={}
            for name,mask in [('train',train),('heldout',~train)]:
                details[name]={}
                for c,label in [(1,'M'),(2,'S')]:
                    take=mask&(y==c);bad=take&(pred!=c)
                    anygood=(member==c).any(1);allbad=~anygood
                    details[name][label]={'rows':int(take.sum()),'mean_probability_errors':int(bad.sum()),
                        'errors_all_16_members_wrong':int((bad&allbad).sum()),
                        'errors_some_member_correct':int((bad&anygood).sum()),
                        'mean_logit_errors_diagnostic_only':int((take&(meanlogit!=c)).sum()),
                        'mean_logit_repairs':int((bad&(meanlogit==c)).sum()),
                        'mean_logit_regressions':int((take&(pred==c)&(meanlogit!=c)).sum()),
                        'S_probability_std_mean_on_errors':float(row[bad,:,2].std(1).mean()) if bad.any() else None}
            gm=np.concatenate([g.ravel() for g in grads[1]]);gs=np.concatenate([g.ravel() for g in grads[2]])
            total=gm+gs
            gg={'M_norm_original_frequency':float(np.linalg.norm(gm)),'S_norm_original_frequency':float(np.linalg.norm(gs)),
                'cosine_M_S':float(gm@gs/(np.linalg.norm(gm)*np.linalg.norm(gs))),
                'negative_total_gradient_directional_derivative_M':float(-gm@total),
                'negative_total_gradient_directional_derivative_S':float(-gs@total),
                'total_head_gradient_inf':float(np.abs(total).max()),
                'scope':'Fixed checkpoint head-only unregularized training CE derivative, not actual AdamW step, finite-step gain, or transfer proof.'}
            all_results.append({'view':view,'fold':k,'counts':details,'head_gradient_diagnostic':gg,
                                'frozen_prediction_replay_passed':True,'saved_probability_max_abs_difference':float(np.max(np.abs(avg-saved))),
                                'same_forward_head_reconstruction_max_abs_difference':replay_max,'analytic_gradient_autograd_check':derivative_checked})
            print(json.dumps({'view':view,'fold':k,'heldout':details['heldout'],'gradient':gg},ensure_ascii=False),flush=True)
            del model
            if DEVICE=='cuda':torch.cuda.empty_cache()
    result={'status':'frozen_model_diagnosis_no_fit','classifier_fits':0,'calibration_fits':0,
      'source_sha256':sha(__file__),'input_sha256':receipts,'ledger_sha256':sha(OLD/'support_and_error_ledger.parquet'),
      'model_source_sha256':sha(ROOT/'training/v104_phase_b.py'),'results':all_results,
      'elapsed_seconds':time.monotonic()-start,
      'limits':['Best-member access is oracle diagnosis; no learned selector or deployment change.',
        'Mean-logit is a retrospective control, not a selected new classifier.',
        'All folds were previously inspected; this is no new blind test.',
        'Gradient cancellation can be normal at an optimum; no causal claim that class imbalance alone caused failure.']}
    save(DEST/'frozen_member_audit.json',result)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
