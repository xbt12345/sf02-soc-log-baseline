"""Independent fixed-epoch rotation replay and canonical-fold/teacher checks."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST,DEST as V82
from v85_verify import gpu_manual,cpu_manual,direct,compare_fields
from v87r2_verify import verify_pairs

DEST=ROOT/'artifacts/v87_solver_supervision_r2_20260927'


def main():
    start=time.monotonic();spec=read(DEST/'rotation/registration.json');reg=read(DEST/'registration.json')
    assert spec['source_sha256']==sha(ROOT/'training/v87r2_rotation.py')
    assert spec['primary_registration_sha256']==sha(DEST/'registration.json')
    assert spec['primary_selection_sha256']==sha(DEST/'continuation.json')
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    selected_all=np.zeros(len(r),bool);selected_all[np.load(V82/'selected_rows.npy')]=True
    fits=0;count=0;full_count=0;maxcpu=0.;details=[]
    for h in spec['folds']:
        folder=DEST/'rotation'/f'fold{h}';fit=~r.fold.isin([0,2,h]).to_numpy();inner=r.fold.eq(h).to_numpy();sel=selected_all&fit
        selected=np.flatnonzero(sel);assert np.array_equal(selected,np.load(folder/'selected_rows.npy'))
        assert np.array_equal(sel&(y>0),fit&(y>0))
        mf=pd.read_parquet(folder/'manifest.parquet');assert np.array_equal(mf.canonical_fold,r.fold) and np.array_equal(mf.selected,sel)
        for field in ['component','body_group','source_symbol']:
            take=r[field]>=0
            assert pd.DataFrame({'id':r.loc[take,field],'role':mf.role[take]}).groupby('id').role.nunique().max()==1
        teacher=joblib.load(folder/'teacher.joblib');z=np.asarray(x@teacher['coef'])+teacher['intercept'];old=z.argmax(1).astype(np.int8)
        np.testing.assert_allclose(z,np.load(folder/'teacher_scores.npy',mmap_mode='r'),atol=1e-10,rtol=0)
        assert np.array_equal(old,np.load(folder/'teacher_prediction.npy'))
        tf=read(folder/'teacher_fit.json');assert tf['source_sha256']==spec['source_sha256'] and tf['converged'] and tf['gradient_inf']<=1e-5
        for path,digest in tf['input_and_preparation_sha256'].items():assert sha(folder/path)==digest,path
        full=np.bincount(fid[fit]*3+y[fit],minlength=x.shape[0]*3).reshape(-1,3);mass=full[np.arange(x.shape[0]),old];pid=np.flatnonzero(mass)
        assert np.array_equal(pid,np.load(folder/'protection_fids.npy')) and np.array_equal(mass[pid],np.load(folder/'protection_original_mass.npy'))
        oldm=z[pid,old[pid]]-np.max(np.where(np.eye(3,dtype=bool)[old[pid]],-np.inf,z[pid]),axis=1);eps=np.minimum(.001,oldm/2)
        assert (oldm>0).all();np.testing.assert_array_equal(eps,np.load(folder/'protection_epsilon.npy'))
        paircheck=verify_pairs(r,y,fid,full,selected,folder)
        fits+=1;roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
        for arm in spec['arms']:
            info=read(folder/(arm+'_fit.json'));assert info['source_sha256']==spec['source_sha256'];fits+=1
            assert info['fixed_primary_epoch']==spec['fixed_epoch'] and not info['new_checkpoint_selection']
            trace=read(folder/(arm+'_selection_trace.json'));progress=read(folder/(arm+'_progress.json'))
            assert info['attempts']==len(progress)<=spec['fixed_epoch']
            for p in progress:assert p['protection']['violating_input_groups']==p['protection']['protected_negative_flips']==0
            for claim in trace:
                name=claim['name'];bundle=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True)
                assert bundle['source_sha256']==spec['source_sha256'] and bundle['canonical_inner_fold']==h
                for v in bundle['optimizer']['state'].values():assert int(v['step'])==sum(i['accepted'] for i in progress[:claim['epoch']])
                p=np.load(folder/(name+'_predictions.npz'));ids=np.unique(np.concatenate([pid,p['fit_feature_ids'],p['inner_feature_ids']]))
                delta=gpu_manual(x,ids,bundle['model']);pred=(z[ids]+delta).argmax(1).astype(np.int8)
                for key in ['fit','inner']:assert np.array_equal(pred[np.searchsorted(ids,p[key+'_feature_ids'])],p[key+'_prediction']),name
                replay=np.full(x.shape[0],-1,np.int8);replay[ids]=pred
                compare_fields(direct(y[sel],old[fid[sel]],replay[fid[sel]]),claim['selected_fit']);compare_fields(direct(y[inner],old[fid[inner]],replay[fid[inner]]),claim['inner'])
                zp=z[pid]+delta[np.searchsorted(ids,pid)];newm=zp[np.arange(len(pid)),old[pid]]-np.max(np.where(np.eye(3,dtype=bool)[old[pid]],-np.inf,zp),axis=1)
                assert ((eps-newm)>reg['nonlinear_margin_tolerance']).sum()==0
                assert mass[pid][zp.argmax(1)!=old[pid]].sum()==0
                probe=np.linspace(0,len(ids)-1,min(512,len(ids))).astype(int)
                cp=cpu_manual(x[ids[probe]],bundle['model']);difference=float(np.max(abs(cp-delta[probe])))
                maxcpu=max(maxcpu,difference);assert difference<1e-4
                count+=1;del bundle,delta;torch.cuda.empty_cache()
                print(json.dumps({'verified_rotation_state':name,'fold':h}),flush=True)
            name=info['saved_states'][-1];bundle=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True)
            replay=(z+gpu_manual(x,np.arange(x.shape[0]),bundle['model'])).argmax(1).astype(np.int8)
            assert np.array_equal(replay,np.load(folder/(name+'_all_prediction.npy')))
            dx=next(d for d in read(folder/'diagnosis.json') if d['name']==name)
            for claim in dx['roles']:
                mask=roles[claim['role']];compare_fields(direct(y[mask],old[fid[mask]],replay[fid[mask]]),claim)
            full_count+=1;del bundle;torch.cuda.empty_cache()
        details.append({'fold':h,'train_population_rows':int(fit.sum()),'train_selected_rows':int(sel.sum()),
                       'protected_original_rows':int(mass.sum()),'pair_verification':paircheck})
    cont=read(DEST/'rotation/continuation.json');assert fits==cont['actual_classifier_fits']
    out={'status':'passed','source_sha256':sha(__file__),'registered_rotation_source_sha256':spec['source_sha256'],
         'actual_classifier_fits':fits,'actual_calibration_fits':0,'new_teacher_fits':len(spec['folds']),
         'all_saved_states_replayed':count,'all_input_final_models_replayed':full_count,'CPU_max_difference':maxcpu,
         'folds':details,'seconds':time.monotonic()-start,
         'scope':'Independent original-row, true-label, canonical-role and saved-weight replay of the fixed primary epoch on rotation folds. No re-selection; unsaved epoch full-P live checks remain logs. Not unknown-input acceptance.'}
    save(DEST/'rotation/verification.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
