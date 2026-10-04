"""Independent witness and clustered-change check of final frozen artifacts; no fits."""
import json
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST,OLD,read,save,sha


def main():
    v=read(DEST/'verification.json');assert v['status']=='passed'
    stress=read(DEST/'stress_model_replay.json');assert stress['status']=='complete' and stress['source_sha256']==sha(ROOT/'training/v89_stress_verify.py')
    branch=read(DEST/'partial_expression_registration.json');assert branch['source_sha256']==sha(ROOT/'training/v89_partial_expression.py')
    for p,h in branch['input_bindings'].items():assert sha(DEST/p)==h
    witness={}
    z=np.load(OLD/'teacher_scores.npy',mmap_mode='r')
    r=pd.read_parquet(OUT/'rows.parquet');fid=np.load(ROOT/'artifacts/v79_execution_20260927/row_feature_id.npy');old=z.argmax(1)[fid];y=r.label_index.to_numpy();inner=r.fold.eq(1).to_numpy()
    rng=np.random.default_rng(89091);clustered={};counts=[]
    for name in ['R00','R01','R10','R11','F00']:
        rec=read(DEST/(name+'_fit.json'));assert rec['model_sha256']==sha(DEST/(name+'_model.npz')) and rec['source_sha256']==sha(ROOT/('training/v89_partial_expression.py' if name=='F00' else 'training/v89_execute.py'))
        a=np.load(DEST/(name+'_dual_derived_witness.npz'));D=a['D'];bb=a['b'];V=np.load(DEST/(name+'_basis.npy'),mmap_mode='r');cut=np.load(DEST/(name+'_slack_linear_solution.npz'))['cuts']
        orig=np.load(DEST/'F00_group_keys.npy')[:,0] if name=='F00' else np.arange(len(z));z0=np.asarray(z[orig]);base=z0.argmax(1);margin=z0[np.arange(len(orig)),base]-np.max(np.where(np.eye(3,dtype=bool)[base],-np.inf,z0),axis=1);eps=np.minimum(.001,margin/2)+np.minimum(1e-5,margin/4)
        for i,(kind,row,cls,other,j) in enumerate(cut):
            target=np.zeros(3*V.shape[1]);target[cls*V.shape[1]:(cls+1)*V.shape[1]]=V[row];target[other*V.shape[1]:(other+1)*V.shape[1]]=-V[row]
            assert np.array_equal(target,D[i]) and abs(bb[i]-((.01 if kind else eps[row])-(z0[row,cls]-z0[row,other])))<1e-14
        lam=a['multiplier'];eq=float(np.abs(D.T@lam).max());rhs=float(bb@lam);assert eq<1e-9 and rhs>1e-7 and lam.min()>=-1e-10 and abs(lam.sum()-1)<1e-8
        witness[name]={'independent_equation_residual':eq,'independent_positive_rhs':rhs,'all_witness_constraints_rebuilt':len(cut),'scope':'Approximate fixed-basis Farkas certificate; does not show arbitrary raw inputs cannot distinguish classes.'}
        gid=np.load(DEST/'F00_row_group.npy') if name=='F00' else fid;pred=np.load(DEST/(name+'_all_prediction.npy'))[gid]
        asa=inner&r.route.eq('asa').to_numpy();data=pd.DataFrame({'component':r.component[asa].to_numpy(),'repairs':((old[asa]!=y[asa])&(pred[asa]==y[asa])).astype(int),'regressions':((old[asa]==y[asa])&(pred[asa]!=y[asa])).astype(int)})
        comp=data.groupby('component')[['repairs','regressions']].sum();net=(comp.regressions-comp.repairs).to_numpy();boots=[]
        for j in range(20):
            ix=rng.integers(0,len(net),size=(100,len(net)));boots.extend(net[ix].sum(1).tolist())
        ci=np.quantile(boots,[.025,.975]).tolist();clustered[name]={'ASA_inner_error_difference':int(net.sum()),'repairs':int(comp.repairs.sum()),'regressions':int(comp.regressions.sum()),'repair_components':int((comp.repairs>0).sum()),'regression_components':int((comp.regressions>0).sum()),'evaluation_components':len(comp),'component_bootstrap_error_difference_interval_95':ci,'bootstrap_replicates':2000,'scope':'Exploratory resampling of inspected development components; errors counted in original rows. Neither row-independent nor a new unseen-domain confidence certificate.'}
        for cls in [0,1,2]:
            take=inner&(y==cls);counts.append({'model':name,'class':cls,'support':int(take.sum()),'old_correct':int((old[take]==cls).sum()),'new_correct':int((pred[take]==cls).sum()),'repairs':int(((old[take]!=cls)&(pred[take]==cls)).sum()),'regressions':int(((old[take]==cls)&(pred[take]!=cls)).sum()),'recall':float((pred[take]==cls).mean())})
    pd.DataFrame(counts).to_csv(DEST/'inner_class_counts.csv',index=False)
    bindings={p.name:sha(p) for p in [DEST/'verification.json',DEST/'stress_model_replay.json',DEST/'dual_derived_certificates.json',DEST/'support_train_and_holdout_diagnosis.json',DEST/'inner_class_counts.csv']}
    for p in DEST.glob('*_dual_derived_witness.npz'):bindings[p.name]=sha(p)
    save(DEST/'postcheck.json',{'status':'passed','source_sha256':sha(__file__),'new_fits':0,'new_solver_calls':0,'witnesses':witness,'clustered_ASA_changes':clustered,'output_bindings':bindings,'no_candidate_promoted':read(DEST/'selection.json')['primary_selected'] is None and read(DEST/'partial_expression_selection.json')['selected'] is None})
    print(json.dumps({'status':'passed','certificates_checked':len(witness),'clustered_ASA_changes':clustered},ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
