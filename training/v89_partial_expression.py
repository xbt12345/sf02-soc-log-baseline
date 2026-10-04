"""One registered, row-preserving partial-fact residual; no validation tuning."""
import argparse,json,time
import numpy as np,pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from v89_common import *
from v89_solver import Problem
from v81_training_contract import compare

NAME='F00'
WIDTH=256


def tokens(observation):
    a=json.loads(observation);f=a['facts'];s=a['states'];out={}
    # Unknown/redacted/invalid are the same missing observation, never a label fingerprint.
    k={n:v for n,v in f.items() if s.get(n)=='known'}
    if k.get('action')=='deny':
        for n,v in sorted(k.items()):out['asa:'+n+'='+str(v)]=1.
        names=['action','outcome','transport_protocol','src_role','dst_role']
        if all(n in k for n in names):
            ordered='|'.join(n+'='+str(k[n]) for n in names)
            out['asa:ordered:'+ordered]=1.
            for n in ['src_port_fixed','dst_port_fixed']:
                if n in k:out['asa:ordered:'+ordered+'|'+n+'='+str(k[n])]=1.
    else:
        for n,v in sorted(k.items()):out['windows:'+n+'='+str(v)]=1.
        if 'event_code_observed' in k:
            for n in sorted(k):
                if n.startswith('privilege_'):out['windows:event='+str(k['event_code_observed'])+'|'+n]=1.
    return out


def dictionary_features(observations):
    # Compact local parameter banks; complete literal facts and original R0 remain separately retained.
    h=FeatureHasher(n_features=128,input_type='dict',alternate_sign=False)
    from scipy import sparse
    a=[tokens(s) for s in observations]
    x=h.transform([{k:v for k,v in z.items() if k.startswith('asa:')} for z in a])
    w=h.transform([{k:v for k,v in z.items() if k.startswith('windows:')} for z in a])
    return normalize(sparse.hstack([x,w],format='csr'),norm='l2')


def register():
    check();assert read(DEST/'selection.json')['primary_selected'] is None
    assert read(DEST/'row_facts_receipt.json')['status']=='complete'
    assert not (DEST/'partial_expression_registration.json').exists()
    save(DEST/'partial_expression_registration.json',{
        'source_sha256':sha(__file__),'name':NAME,'fits':1,'width':WIDTH,
        'gate':'All four matched readouts failed; known literal extraction gaps verified before any fit.',
        'fixed_design':'Frozen R0 teacher plus zero-initialized local residual of partial literal facts; no hidden checkpoint/teacher-coefficient search or global intercept. Separate ASA/Windows128 banks, atomic facts plus named ordered behavior and event/privilege interactions.',
        'representation':'Generic signed-disabled feature hashing, L2 normalized per observation, invertible training-frequency RMS conditioning. Hash collisions possible; original R0 and full observation dictionary preserved, not an assertion of lossless model encoding.',
        'unknowns':'No redacted vs invalid, NULL vs empty, exact identity, product, absolute time or guessed source port features.',
        'grouping':'Unique (old R0 input id, original-row observation id), independent of label; no projection consensus.',
        'objectives':'Same protected readout solver/positive margin/slack/L1/risk and budgets as main registration. Known22 input conflicts still excluded from hard E; all main-loss rows and full P retained.',
        'selection':'Same inner ASA and exact class/nonregression guards before C/H; one fixed branch only.',
        'input_bindings':{p.name:sha(p) for p in [DEST/'row_fact_code.npy',DEST/'row_fact_dictionary.parquet',DEST/'row_facts_receipt.json',DEST/'selection.json']},
        'no_new_port_coding_search':True,'external_data':False,'development_not_blind':True})
    emit(stage='partial_expression_registered',name=NAME)


def fit_branch():
    check();reg=read(DEST/'partial_expression_registration.json');assert reg['source_sha256']==sha(__file__)
    for p,h in reg['input_bindings'].items():assert sha(DEST/p)==h
    assert not (DEST/(NAME+'_fit.json')).exists();li=ledger_start('classifier_fit',NAME);start=time.monotonic()
    r,y,fid,z0,old,sel,fit=data();obs=np.load(DEST/'row_fact_code.npy')
    pairs,gid=np.unique(np.c_[fid,obs],axis=0,return_inverse=True);gid=gid.astype(np.int32);np.save(DEST/'F00_row_group.npy',gid);np.save(DEST/'F00_group_keys.npy',pairs)
    dz=dictionary_features(pd.read_parquet(DEST/'row_fact_dictionary.parquet').observation_json)
    n=len(pairs);z=np.asarray(z0[pairs[:,0]]);op=old[pairs[:,0]]
    full=raw_counts(gid,y,fit,n);mask=np.zeros(len(r),bool);mask[sel]=True;cc=raw_counts(gid,y,mask,n);used=np.flatnonzero(cc.sum(1))
    pmass=full[np.arange(n),op];P=np.flatnonzero(pmass)
    # Keep the same 293 original hard-target rows as the four primary diagnostic arms.
    oldcc=raw_counts(fid,y,fit,len(old));free=fit&(old[fid]!=y)&(oldcc[np.arange(len(old)),old][fid]==0)
    ec=raw_counts(gid,y,free,n);E=np.flatnonzero(ec.sum(1));truth=ec.argmax(1)[E]
    assert int(ec.sum())==293 and (ec[E]>0).sum(1).max()==1
    weights=np.zeros(len(E))
    for cls in [1,2]:
        m=truth==cls;weights[m]=ec[E[m],cls]/ec[E[m],cls].sum()/2
    trainobs=np.bincount(pairs[used,1],weights=cc[used].sum(1),minlength=dz.shape[0])
    scale=np.sqrt(np.asarray(dz.power(2).T@trainobs).ravel()/cc.sum());scale=np.where(scale>1e-12,scale,1.)
    V=np.lib.format.open_memmap(DEST/'F00_basis.npy',mode='w+',dtype=np.float64,shape=(n,WIDTH))
    for beg in range(0,n,8192):
        ids=pairs[beg:beg+8192,1];V[beg:beg+len(ids)]=dz[ids].toarray()/scale
    V.flush();np.save(DEST/'F00_scale.npy',scale)
    margin=z[np.arange(n),op]-np.max(np.where(np.eye(3,dtype=bool)[op],-np.inf,z),axis=1)
    epsilon=np.minimum(.001,margin/2)+np.minimum(1e-5,margin/4)
    problem=Problem(V,z,P,epsilon,E,truth,weights,NAME)
    no,fn=problem.linear(False);u,fp=problem.linear(True)
    report={'name':NAME,'representation':'partial_observed_facts','new_classifier_fits':1,'source_sha256':sha(__file__),
        'P_original_rows':int(pmass.sum()),'P_input_groups':len(P),'E_original_rows':int(ec.sum()),'E_input_groups':len(E),
        'noP_feasibility':fn,'P_feasibility':fp,'all_training_M_S_used':True}
    if u is None:
        u,slack=problem.linear(True,True);report['slack_optimization']=slack
        if u is not None:problem.slack_cap=float(weights@u[problem.n:])+1e-8
    if u is None:
        report.update(status='aborted_no_verified_start',seconds=time.monotonic()-start);save(DEST/'F00_fit.json',report);ledger_end(li,'aborted',reason=report['status']);return
    u,tr=problem.train(u,used,cc[used]);report['training']=tr
    if u is None:
        report.update(status='aborted_readout_solver',seconds=time.monotonic()-start);save(DEST/'F00_fit.json',report);ledger_end(li,'aborted',reason=report['status']);return
    np.savez_compressed(DEST/'F00_model.npz',theta=u,scale=scale,cuts=np.asarray(problem.cuts),E=E,truth=truth,weights=weights,
        slack_cap=np.array(np.nan if problem.slack_cap is None else problem.slack_cap),uses_slack=np.array(problem.slack))
    pred=problem.scores(u).argmax(1).astype(np.int8);np.save(DEST/'F00_all_prediction.npy',pred)
    roles,cells=evaluate(r,y,gid,op,pred,fit,masks={'fit_full':fit,'inner':r.fold.eq(1).to_numpy()})
    cells.to_csv(DEST/'F00_selection_classwise.csv',index=False);asa=cells[(cells.role=='inner')&(cells.route=='asa')]
    guard=compare(cm_from_counts(raw_counts(gid,y,r.fold.eq(1).to_numpy(),n),op),np.array(roles['inner']['cm']),True)
    eligible=roles['fit_full']['negative_flips']==0 and roles['fit_full']['positive_flips']>0 and roles['inner']['negative_flips']==0 and guard['eligible'] and asa.repairs.sum()>0
    ck=problem.scan(u,True);assert ck['max_full_violation']<=TOL and ck['protected_changed_inputs']==0
    report.update(status='completed',roles=roles,inner_ASA={'repairs':int(asa.repairs.sum()),'regressions':int(asa.regressions.sum()),'errors':int((asa.support-asa.new_correct).sum()),'baseline_errors':int((asa.support-asa.old_correct).sum())},
        primary_eligible=bool(eligible),inner_metric_guard=guard,risk_converged=bool(tr.get('converged',False)),full_constraint_check=ck,seconds=time.monotonic()-start,model_sha256=sha(DEST/'F00_model.npz'))
    save(DEST/'F00_fit.json',report);save(DEST/'partial_expression_selection.json',{'selected':NAME if eligible else None,'selected_before_C_H':True})
    roles,cells=evaluate(r,y,gid,op,pred,fit);save(DEST/'F00_full_diagnosis.json',roles);cells.to_csv(DEST/'F00_classwise.csv',index=False)
    ledger_end(li,'completed',eligible=bool(eligible));emit(stage='partial_expression_complete',name=NAME,eligible=bool(eligible),fit_errors=report['roles']['fit_full']['errors'],inner_ASA=report['inner_ASA'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','fit']);a=ap.parse_args()
    with threadpool_limits(limits=4):register() if a.stage=='register' else fit_branch()
