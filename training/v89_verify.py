"""Independent original-row decision, constraint, support and numeric replay; no fits."""
import json,time
import numpy as np,pandas as pd,joblib,torch
from threadpoolctl import threadpool_limits
from scipy.special import expit
from run_v75 import load_sparse
from v89_common import ROOT,OUT,DEST,LAST,OLD,PRIOR,read,save,sha


def counts(gid,y,mask,n):
    return np.bincount(gid[mask]*3+y[mask],minlength=n*3).reshape(n,3)


def main():
    start=time.monotonic();reg=read(DEST/'registration.json')
    for p,h in {**reg['source_bindings'],**reg['input_bindings']}.items():assert sha(ROOT/p)==h,p
    r=pd.read_parquet(OUT/'rows.parquet');y=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.array_equal(y,r.label_index) and np.array_equal(r.row_position,np.arange(len(y)))
    fid=np.load(LAST/'row_feature_id.npy');z=np.load(OLD/'teacher_scores.npy',mmap_mode='r');old=z.argmax(1);fit=~r.fold.isin([0,1,2]).to_numpy()
    selected=np.load(PRIOR/'fold1/selected_rows.npy');mask=np.zeros(len(y),bool);mask[selected]=True
    assert np.array_equal(mask&(y>0),fit&(y>0))
    cc=counts(fid,y,fit,len(old));pm=cc[np.arange(len(old)),old];P=np.flatnonzero(pm);E=np.flatnonzero((cc.sum(1)>0)&(pm==0))
    assert pm.sum()==922009 and cc[E].sum()==293
    X=load_sparse(LAST/'X');rng=np.random.default_rng(89017);sample=np.unique(np.r_[rng.choice(len(old),1024,replace=False),E])
    # Independently evaluate both layers on CPU float64, without helper model.forward/hidden.
    hiddencheck={}
    for arm in ['A0','A3']:
        a=torch.load(PRIOR/f'fold1/{arm}_epoch020.pt',map_location='cpu',weights_only=True)['model']
        w1=a['first.weight'].double().numpy();w2=a['second.weight'].double().numpy();b2=a['second.bias'].double().numpy()
        h1=np.asarray(X[sample]@w1.T)
        gelu=lambda t:.5*t*(1+np.tanh(np.sqrt(2/np.pi)*(t+.044715*t**3)))
        hh=gelu(gelu(h1)@w2.T+b2)
        stored=np.load(DEST/(arm+'_hidden.npy'),mmap_mode='r')[sample]
        err=float(np.abs(hh-stored).max());assert err<1e-12,(arm,err)
        hiddencheck[arm]={'cpu_manual_replay_inputs':len(sample),'max_absolute_difference':err}
    result={};role_masks={'fit_full':fit,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    names=list(reg['arms'])+(['F00'] if (DEST/'F00_fit.json').exists() else [])
    for name in names:
        rec=read(DEST/(name+'_fit.json'))
        if rec['status']!='completed':result[name]={'status':rec['status'],'model_replay':False};continue
        m=np.load(DEST/(name+'_model.npz'));theta=m['theta'];V=np.load(DEST/(name+'_basis.npy'),mmap_mode='r');d=V.shape[1]
        gid=np.load(DEST/'F00_row_group.npy') if name=='F00' else fid
        orig=np.load(DEST/'F00_group_keys.npy')[:,0] if name=='F00' else np.arange(len(old))
        z0=np.asarray(z[orig]);baseline=old[orig];full=counts(gid,y,fit,len(orig));pidx=np.flatnonzero(full[np.arange(len(orig)),baseline]);main=counts(gid,y,mask,len(orig));used=np.flatnonzero(main.sum(1))
        scale=m['scale'];assert np.all(scale>0)
        if name!='F00':
            arm,free=reg['arms'][name];hh=np.load(DEST/(arm+'_hidden.npy'),mmap_mode='r')
            block=np.c_[z[used],hh[used]] if free else hh[used]
            sc=np.sqrt(np.square(block).T@main[used].sum(1)/main.sum());sc=np.where(sc>1e-12,sc,1.)
            assert np.allclose(sc,scale,rtol=1e-13,atol=1e-14)
            sampleblock=np.c_[z[sample],hh[sample]] if free else hh[sample]
            assert np.allclose(V[sample,:-1],sampleblock/scale,rtol=1e-13,atol=1e-14) and np.all(V[sample,-1]==1)
        else:
            from v89_partial_expression import dictionary_features
            dz=dictionary_features(pd.read_parquet(DEST/'row_fact_dictionary.parquet').observation_json)
            keys=np.load(DEST/'F00_group_keys.npy');obsweights=np.bincount(keys[used,1],weights=main[used].sum(1),minlength=dz.shape[0]);sc=np.sqrt(np.asarray(dz.power(2).T@obsweights).ravel()/main.sum());sc=np.where(sc>1e-12,sc,1.)
            assert np.allclose(sc,scale,rtol=1e-13,atol=1e-14)
            fcheck=rng.choice(len(orig),1024,replace=False)
            assert np.allclose(V[fcheck],dz[keys[fcheck,1]].toarray()/scale,rtol=1e-13,atol=1e-14)
        scores=z0+V@theta[:3*d].reshape(3,d).T;pred=scores.argmax(1)
        assert np.array_equal(pred,np.load(DEST/(name+'_all_prediction.npy')))
        margin=z0[np.arange(len(orig)),baseline]-np.max(np.where(np.eye(3,dtype=bool)[baseline],-np.inf,z0),axis=1)
        eps=np.minimum(.001,margin/2)+np.minimum(1e-5,margin/4)
        pz=scores[pidx];base=baseline[pidx];diff=eps[pidx,None]-(pz[np.arange(len(pidx)),base,None]-pz);diff[np.arange(len(pidx)),base]=-np.inf
        violation=max(0.,float(diff.max()));assert violation<=1e-8 and not (pz.argmax(1)!=base).any()
        ei=m['E'];truth=m['truth'];ez=scores[ei];ed=.01-(ez[np.arange(len(ei)),truth,None]-ez);ed[np.arange(len(ei)),truth]=-np.inf
        if bool(m['uses_slack']):
            sl=theta[3*d:];ed-=sl[:,None];assert sl.min()>=-1e-8 and m['weights']@sl<=float(m['slack_cap'])+1e-8
        assert ed.max()<=1e-8
        rr={};newrows=pred[gid];oldrows=baseline[gid]
        diagnosis=read(DEST/(name+'_full_diagnosis.json'))
        for role,take in role_masks.items():
            cm=np.bincount(y[take]*3+newrows[take],minlength=9).reshape(3,3)
            repairs=int((take&(oldrows!=y)&(newrows==y)).sum());nf=int((take&(oldrows==y)&(newrows!=y)).sum())
            assert cm.tolist()==diagnosis[role]['cm'] and repairs==diagnosis[role]['positive_flips'] and nf==diagnosis[role]['negative_flips']
            rr[role]={'cm':cm.tolist(),'repairs':repairs,'regressions':nf}
        result[name]={'status':'replayed','original_rows_checked':len(y),'full_P_rows_checked':int(full[np.arange(len(orig)),baseline].sum()),'max_P_violation':violation,'max_E_violation':max(0.,float(ed.max())),'roles':rr,'risk_converged':rec['risk_converged']}
        del scores,V,z0
    certs={}
    for p in sorted(DEST.glob('*_certificate_constraints.npz')):
        key=p.name.replace('_certificate_constraints.npz','');a=np.load(p);lam=np.load(DEST/(key+'_farkas_multiplier.npy'))
        eq=float(np.abs(a['A'].T@lam).max());rhs=float(a['b']@lam)
        assert eq<1e-9 and rhs>1e-7 and lam.min()>=-1e-10 and abs(lam.sum()-1)<1e-8
        certs[key]={'equation_residual':eq,'positive_rhs':rhs,'scope':'Floating point Farkas check at registered tolerances, not arbitrary-representation impossibility.'}
    support={};er=np.load(DEST/'support_evaluation_rows.npy');target=np.load(DEST/'support_target_rows.npy')
    for name in reg['support_controls']:
        a=read(DEST/('support_'+name+'_fit.json'));model=joblib.load(DEST/('support_'+name+'.joblib'));tr=np.load(DEST/('support_'+name+'_rows.npy'))
        assert not set(r.component.iloc[tr]).intersection(set(r.component.iloc[er]))
        nc=counts(fid,y,np.isin(np.arange(len(y)),tr),len(old));used=np.flatnonzero(nc.sum(1));xx=X[used];zfit=np.asarray(xx@model['coef'])+model['intercept'];total=nc[used].sum(1);den=total.sum()
        res=(expit(zfit)*total[:,None]-nc[used])/den;gradient=np.vstack([np.asarray(xx.T@res)+1e-6*model['coef'],res.sum(0)])
        grad=float(np.abs(gradient).max());assert abs(grad-a['gradient_inf'])<1e-12
        saved=np.load(DEST/('support_'+name+'_predictions.npz'));score=np.asarray(X[saved['unique_input_ids']]@model['coef'])+model['intercept']
        assert np.allclose(score,saved['scores'],atol=1e-12,rtol=1e-12)
        p=score.argmax(1)[np.searchsorted(saved['unique_input_ids'],fid[target])];assert np.array_equal(p,saved['target_prediction'])
        cm=np.bincount(y[target]*3+p,minlength=9).reshape(3,3);assert cm.tolist()==a['target_evaluation']['cm']
        support[name]={'target_cm':cm.tolist(),'gradient_inf':grad,'converged':a['converged'],'training_rows':len(tr),'training_class_counts':np.bincount(y[tr],minlength=3).tolist()}
    for cls in [1,2]:
        a=np.load(DEST/f'support_withdraw_{cls}_rows.npy');b=np.load(DEST/f'support_control_{cls}_rows.npy')
        assert np.bincount(y[a],minlength=3).tolist()==np.bincount(y[b],minlength=3).tolist()
        assert r.fold.iloc[a].value_counts().sort_index().equals(r.fold.iloc[b].value_counts().sort_index())
    prior={};paths=['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json']
    for ver,key in [(81,'diagnosis'),(82,'capacity'),(83,'root_review'),(84,'preservation'),(85,'protection'),(86,'boundary_review'),(87,'solver_supervision'),(88,'root_review')]:paths.append(f'evidence/2026-09-27/v{ver}_{key}/delivery.json')
    for p in paths:prior.update(read(ROOT/p)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    ledger=read(DEST/'execution_ledger.json');assert all(a['status']!='started' for a in ledger)
    rowfacts=read(DEST/'row_facts_receipt.json');assert rowfacts['row_mapping_sha256']==sha(DEST/'row_fact_code.npy') and rowfacts['dictionary_sha256']==sha(DEST/'row_fact_dictionary.parquet')
    hard=read(DEST/'hard_m_facts.json');obs=np.load(DEST/'row_fact_code.npy');dictionary=pd.read_parquet(DEST/'row_fact_dictionary.parquet').observation_json
    for a in hard:assert json.loads(dictionary.iloc[obs[a['row_position']]])=={k:a['extracted'][k] for k in ['facts','states']}
    receipt={'status':'passed','source_sha256':sha(__file__),'new_fits':0,'original_labels_checked':len(y),'all_train_M_S_in_main_loss':True,'manual_cpu_hidden_checks':hiddencheck,'readouts':result,'infeasibility_certificates':certs,'support_fits':support,'original_row_fact_mapping_preserved':True,'hard_M_literal_facts_replayed':len(hard),'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'actual_classifier_fits':sum(a['kind']=='classifier_fit' for a in ledger),'solver_calls':{k:sum(a['kind']==k for a in ledger) for k in ['linear_solver_call','quadratic_solver_call']},'unfinished_ledger_records':0,'registration_sha256':sha(DEST/'registration.json'),'output_bindings':{p.name:sha(p) for p in DEST.glob('*_fit.json')},'seconds':time.monotonic()-start,'scope':'Original labels and all original-row decisions/guards replayed; CPU hidden forward independently checked on sampled and all hard-error inputs, not all hidden coordinates. Five support gradients/models replayed; no external, blind or full raw-task replay.'}
    save(DEST/'verification.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
