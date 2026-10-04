"""Cross-check the no-fit problem ledger with explicit saved-weight formulas."""
import math
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST
from v85_verify import tensor_csr

DEST=ROOT/'artifacts/v85_protection_20260927'


def main():
    d=read(DEST/'problem_ledger.json');assert d['source_sha256']==sha(ROOT/'training/v85_failure_audit.py')
    r=rows();assert np.array_equal(r.row_position,np.arange(len(r)))
    y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X');folder=DEST/'fold1'
    chosen=np.load(folder/'selected_rows.npy');selected=np.zeros(len(r),bool);selected[chosen]=True
    old=np.load(folder/'teacher_prediction.npy');cc=np.bincount(fid[chosen]*3+y[chosen],minlength=x.shape[0]*3).reshape(-1,3)
    used=np.flatnonzero(cc.sum(1));pcount=cc[used,old[used]]
    floor=int((cc[used].sum(1)-cc[used].max(1)).sum())
    lockfloor=int(np.where(pcount>0,cc[used].sum(1)-pcount,cc[used].sum(1)-cc[used].max(1)).sum())
    assert (floor,lockfloor)==(d['fit_empirical_unrestricted_lookup_floor'],d['fit_empirical_protected_lookup_floor'])
    case=pd.read_parquet(DEST/'problem_casebook.parquet');expected=[];oldrow=old[fid]
    roles={'selected_fit':selected,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    for name in ['A_epoch060','B_epoch060','C_epoch060']:
        new=np.load(folder/(name+'_all_prediction.npy'))[fid]
        for role,mask in roles.items():
            for kind,ok in [('positive',(oldrow!=y)&(new==y)),('negative',(oldrow==y)&(new!=y)),('remaining_old_error',(oldrow!=y)&(new!=y))]:
                part=case[(case.name==name)&(case.role==role)&(case.kind==kind)];ids=np.flatnonzero(mask&ok)
                assert np.array_equal(part.row_position.to_numpy(),ids),(name,role,kind)
                assert np.array_equal(part.label_index,y[ids]) and np.array_equal(part.new,new[ids]) and np.array_equal(part.old,oldrow[ids])
                assert np.array_equal(part.route,r.iloc[ids].route) and np.array_equal(part.component,r.iloc[ids].component)
                expected.append(len(ids))
    assert sum(expected)==len(case)
    saved=torch.load(folder/'C_epoch060.pt',map_location='cpu',weights_only=True)
    weights={k:t.cuda().detach().requires_grad_(True) for k,t in saved.items()};params=list(weights.values())
    xx=tensor_csr(x[used]);zt=torch.tensor(np.load(folder/'teacher_scores.npy',mmap_mode='r')[used],device='cuda',dtype=torch.float32)
    ct=torch.tensor(cc[used],dtype=torch.float32,device='cuda');label=torch.tensor(old[used].astype(np.int64),device='cuda')
    mass=ct.gather(1,label[:,None]).squeeze(1)
    def gelu(z):return .5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z*z*z)))
    def residual(xpart):
        first=gelu(torch.sparse.mm(xpart,weights['first.weight'].T))
        second=gelu(first@weights['second.weight'].T+weights['second.bias'])
        return second@weights['last.weight'].T+weights['last.bias']
    def logits(xpart,teacher):return teacher+residual(xpart)
    def loss(z,counts):return (torch.nn.functional.softplus(z).sum(1)@counts.sum(1)-(z*counts).sum())/counts.sum()
    def margins(z,lab):return z.gather(1,lab[:,None]).squeeze(1)-z.masked_fill(torch.nn.functional.one_hot(lab,3).bool(),-torch.inf).max(1).values
    scores=logits(xx,zt);main=loss(scores,ct);grad=torch.autograd.grad(main,params);norm=torch.sqrt(sum(g.square().sum() for g in grad))
    teacher64=torch.tensor(np.load(folder/'teacher_scores.npy',mmap_mode='r')[used],dtype=torch.float64,device='cuda')
    def stable_loss():return loss(teacher64+residual(xx).double(),ct.double())
    with torch.no_grad():stable_base=float(stable_loss())
    direction=[-g/norm for g in grad];original=[p.detach().clone() for p in params];epsilon=torch.minimum(torch.full_like(mass,.001),margins(zt,label)/2)
    assert abs(float(norm)-d['local_data_gradient_norm'])<1e-6
    actual=[]
    def verify_step(direction,step,claim):
        with torch.no_grad():
            for p,a,g in zip(params,original,direction):p.copy_(a+step*g)
            z=logits(xx,zt);m=margins(z,label);bad=(epsilon-m)>1e-7;active=mass>0
            values={'violating_input_groups':int((bad&active).sum()),'violating_original_rows':int(mass[bad&active].sum()),
                    'protected_negative_flips':int(mass[(z.argmax(1)!=label)&active].sum())}
            for key,v in values.items():assert v==claim['protection'][key],(key,v,claim['protection'][key])
            difference=float(stable_loss())-stable_base;claimed=claim.get('loss_change',claim.get('data_loss_change'))
            assert abs(difference-claimed)<1e-8,(difference,claimed)
            actual.append({'step':step,'loss_change':difference,**values})
            for p,a in zip(params,original):p.copy_(a)
    for q in d['train_only_gradient_ray_checks']:verify_step(direction,q['normalized_data_loss_gradient_step'],q)
    proj=d['single_constraint_tangent_projection'];j=int(np.searchsorted(used,proj['blocker_fid']))
    boundary=margins(logits(tensor_csr(x[used[j:j+1]]),zt[j:j+1]),label[j:j+1])[0]
    gh=torch.autograd.grad(boundary,params);hdot=sum((g*v).sum() for g,v in zip(gh,direction));hn2=sum(g.square().sum() for g in gh)
    tangent=[v-hdot/hn2*g for g,v in zip(gh,direction)]
    dot=sum((g*v).sum() for g,v in zip(gh,tangent))
    assert abs(float(hdot)-proj['original_margin_directional_derivative'])<1e-5
    assert abs(float(dot))<1e-5
    for q in d['projected_direction_checks']:verify_step(tangent,q['step'],q)
    assert all(torch.equal(saved[k],torch.load(folder/'C_epoch040.pt',map_location='cpu',weights_only=True)[k]) for k in saved)
    assert d['original_native_flow_M_preserved']==int(((r.route=='native_flow')&(y==1)).sum())==32596
    out={'status':'passed','source_sha256':sha(__file__),'problem_ledger_sha256':sha(DEST/'problem_ledger.json'),
       'casebook_sha256':sha(DEST/'problem_casebook.parquet'),'casebook_rows_verified':len(case),'empirical_floors':[floor,lockfloor],
       'local_directions_independently_recomputed':actual,'registered_training_source_unchanged':sha(ROOT/'training/v85_protection.py')==read(DEST/'registration.json')['source_sha256'],
       'new_classifier_fits':0,'scope':'No-fit saved-weight explicit formula and original-row case cross-check; local projection is diagnostic, not a trained replacement or transfer improvement.'}
    save(DEST/'problem_verification.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
