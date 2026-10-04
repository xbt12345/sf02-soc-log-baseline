"""No-fit audit of frozen v87 inputs, losses, gradients, and selection attribution."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v79_execute import rows
from v82_capacity import LAST
from v85_protection import Residual, csr_tensor, supervised_loss, ALPHA
from v87r2_execute import hidden, contrastive_loss, PAIR_WEIGHT
from v75_views import view

DEST=ROOT/'artifacts/v88_root_review_20260927'
PRIOR=ROOT/'artifacts/v87_solver_supervision_r2_20260927'
TEACHER=ROOT/'artifacts/v85_protection_20260927/fold1'


def dot(a,b):
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))


def geometry(a,b):
    aa=dot(a,a);bb=dot(b,b)
    return {'a_norm':aa**.5,'b_norm':bb**.5,'cosine':dot(a,b)/(aa*bb)**.5 if aa*bb else None}


def main():
    DEST.mkdir(exist_ok=True)
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    selected=np.load(PRIOR/'fold1/selected_rows.npy');fit=~r.fold.isin([0,1,2]).to_numpy()
    old=np.load(TEACHER/'teacher_prediction.npy');z0=np.load(TEACHER/'teacher_scores.npy')
    full=np.bincount(fid[fit]*3+y[fit],minlength=x.shape[0]*3).reshape(-1,3)
    counts=np.bincount(fid[selected]*3+y[selected],minlength=x.shape[0]*3).reshape(-1,3)
    pmass=full[np.arange(len(old)),old];wrong=fit&(old[fid]!=y)
    blocked=wrong&(pmass[fid]>0);free=wrong&~blocked
    facts=pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts
    raw=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['message_sanitized','label_binary','src_port'])
    assert np.array_equal(raw.label_binary.map({'benign':0,'malicious':1,'suspicious':2}),y)
    conflict_groups=[]
    for code in np.unique(fid[blocked]):
        ix=np.flatnonzero(fit&(fid==code));texts=[view(raw.message_sanitized.iloc[i])[0] for i in ix]
        keys=pd.DataFrame({'text':texts,'port':raw.src_port.iloc[ix].astype(str).to_numpy(),'label':y[ix]})
        mixed=keys.groupby(['text','port'],dropna=False).label.nunique().gt(1)
        conflict_groups.append({'fid':int(code),'rows':len(ix),'class_counts':np.bincount(y[ix],minlength=3).tolist(),
           'wrong_rows':int((old[code]!=y[ix]).sum()),'raw_message_distinct':int(raw.message_sanitized.iloc[ix].nunique(dropna=False)),
           'view_distinct':len(set(texts)),'facts_distinct':len(set(facts.iloc[r.projection_id.iloc[ix]].tolist())),
           'mixed_view_and_independent_port_keys':int(mixed.sum()),'components':int(r.component.iloc[ix].nunique()),
           'independent_port_serialization_by_class':{
               str(c):{repr(k):int(v) for k,v in raw.src_port.iloc[ix[y[ix]==c]].value_counts(dropna=False).items()}
               for c in np.unique(y[ix])},
           'port_serialization_warning':'The raw None versus empty-string distinction is not an observed port. Do not treat a serialization artifact as recovered behavior.',
           'row_positions':ix.tolist(),'one_view_example':texts[0]})
    coverage=pd.read_parquet(PRIOR/'fold1/pair_coverage.parquet').set_index('row_position')
    hard_m=[]
    for code in np.unique(fid[wrong&(y==1)]):
        ix=np.flatnonzero(wrong&(y==1)&(fid==code));i=int(ix[0]);text,ledger=view(raw.message_sanitized.iloc[i])
        hard_m.append({'fid':int(code),'rows':len(ix),'route':str(r.route.iloc[i]),'row_positions':ix.tolist(),
           'pair_reason':str(coverage.loc[i,'reason']) if i in coverage.index else 'outside_ASA_pair_policy',
           'model_view':text,'facts':json.loads(facts.iloc[int(r.projection_id.iloc[i])]),
           'independent_src_port':str(raw.src_port.iloc[i]),'removed_span_kinds':sorted(set(t[2] for t in ledger['spans'] if t[2]!='behavior_or_unresolved'))})
    # Every format/class in the frozen decisions is retained, not just ASA.
    attribution={}
    for name in ['A0_epoch035','A1_epoch060','A2_epoch020','A2_epoch060','A3_epoch060']:
        table=pd.read_csv(PRIOR/'fold1'/(name+'_classwise.csv'))
        attribution[name]=table.loc[(table.repairs>0)|(table.regressions>0)].to_dict('records')
    bins=pd.read_parquet(PRIOR/'fold1/inner_support_bins.parquet');ix=bins.row_position.to_numpy()
    bins['old_wrong']=old[fid[ix]]!=y[ix]
    bins['new_wrong']=np.load(PRIOR/'fold1/A3_epoch060_all_prediction.npy')[fid[ix]]!=y[ix]
    support=bins.groupby(['route','class','support_bin']).agg(rows=('row_position','size'),old_errors=('old_wrong','sum'),new_errors=('new_wrong','sum')).reset_index()
    support.to_csv(DEST/'support_and_errors.csv',index=False)
    loss_gradient=[];used=np.flatnonzero(counts.sum(1));xt=csr_tensor(x[used]);ct=torch.as_tensor(counts[used],dtype=torch.float32,device='cuda')
    teacher=torch.as_tensor(z0[used],dtype=torch.float32,device='cuda')
    correct_counts=np.zeros_like(counts[used]);correct_counts[np.arange(len(used)),old[used]]=counts[used,old[used]]
    err_counts=counts[used]-correct_counts
    pct=torch.as_tensor(correct_counts,dtype=torch.float32,device='cuda');ect=ct-pct
    pairs=pd.read_parquet(PRIOR/'fold1/pairs.parquet')
    pi=[torch.as_tensor(np.searchsorted(used,pairs[k]),device='cuda') for k in ['anchor_fid','positive_fid','negative_fid']]
    pl=torch.as_tensor(pairs.anchor_class.to_numpy(),device='cuda')
    for name in ['A3_epoch000','A3_epoch020','A3_epoch060']:
        path=PRIOR/'fold1'/(name+'.pt');before=sha(path)
        model=Residual().cuda();state=torch.load(path,map_location='cpu',weights_only=True)['model'];model.load_state_dict(state)
        h=hidden(model,xt);z=teacher+model.last(h);den=ct.sum()
        # Both contributions use the SAME global denominator, not unrelated means.
        lp=supervised_loss(z,pct)*pct.sum()/den;le=supervised_loss(z,ect)*ect.sum()/den
        pair=PAIR_WEIGHT*contrastive_loss(h,pi,pl)
        em=ect.clone();em[:,[0,2]]=0
        lm=supervised_loss(z,em)*em.sum()/den
        objectives={'P':lp,'E':le,'main':lp+le,'weighted_pair':pair,'E_M':lm}
        params=list(model.parameters());grads={}
        for key,value in objectives.items():
            gg=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
            grads[key]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gg)]
        groups={'encoder':slice(0,3),'head':slice(3,5)}
        stat={'name':name,'loss_P_global_mass':float(lp),'loss_E_global_mass':float(le),'main_loss':float(lp+le),
              'weighted_pair_loss':float(pair),'error_fraction_of_classification_loss':float(le/(lp+le)),
              'P_E_geometry':geometry(grads['P'],grads['E']),
              'main_pair_geometry':geometry(grads['main'],grads['weighted_pair']),
              'layer_groups':{key:geometry(grads['main'][sl],grads['weighted_pair'][sl]) for key,sl in groups.items()}}
        gmain=grads['main'];gpair=grads['weighted_pair'];gjoint=[a+b for a,b in zip(gmain,gpair)]
        stat['gradient_step_first_order_E_loss_change']={'classification_only':-dot(grads['E'],gmain),'classification_plus_pair':-dot(grads['E'],gjoint)}
        stat['gradient_step_first_order_hard_M_loss_change']={'classification_only':-dot(grads['E_M'],gmain),'classification_plus_pair':-dot(grads['E_M'],gjoint)}
        # Independent float64 analytic BCE head gradient; no optimizer and no fit.
        zn=z.detach().cpu().numpy().astype(np.float64);hn=h.detach().cpu().numpy().astype(np.float64)
        cc=counts[used].astype(np.float64);res=(expit(zn)*cc.sum(1)[:,None]-cc)/cc.sum()
        gw=res.T@hn;gb=res.sum(0)
        stat['independent_head_gradient_max_abs_error']=max(
            float(np.abs(gw-grads['main'][3].cpu().numpy()).max()),
            float(np.abs(gb-grads['main'][4].cpu().numpy()).max()))
        stat['independent_main_loss']=float((np.logaddexp(0,zn).sum(1)*cc.sum(1)-(zn*cc).sum(1)).sum()/cc.sum())
        assert stat['independent_head_gradient_max_abs_error']<1e-7
        assert abs(stat['independent_main_loss']-stat['main_loss'])<1e-6
        stat['scope']='Gradient geometry at frozen weights only; excludes Adam preconditioning, L2 and feasibility projection; not an executed update or causal attribution of actual steps.'
        assert abs(float(lp+le)-float(supervised_loss(z,ct)))<1e-6
        assert before==sha(path)
        loss_gradient.append(stat)
        print(json.dumps(stat),flush=True)
        del model,state,h,z,grads,gmain,gpair,gjoint,params,objectives;torch.cuda.empty_cache()
    out={'status':'no_fit_diagnosis_complete','actual_new_classifier_fits':0,'actual_new_calibration_fits':0,
        'source_sha256':sha(__file__),'prior_delivery_sha256':sha(ROOT/'evidence/2026-09-27/v87_solver_supervision/delivery.json'),
        'same_input_conflict_groups':conflict_groups,'hard_malicious_inputs':hard_m,
        'selected_candidate_attribution':attribution,'loss_and_gradient':loss_gradient,
        'main_loss_rows':int(ct.sum()),'main_loss_correct_rows':int(pct.sum()),'main_loss_error_rows':int(ect.sum()),
        'full_protected_correct_rows':int((fit&~wrong).sum()),
        'original_errors':int(wrong.sum()),'blocked_errors':int(blocked.sum()),'unblocked_errors':int(free.sum()),
        'scope':'Frozen model gradients, original official-row input checks, and existing source-isolated predictions. No optimizer step, head fit, calibration, changed labels or data.'}
    save(DEST/'diagnosis.json',out)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
