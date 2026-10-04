"""Independent saved-weight/row-level replay for the registered preservation trial."""
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST,DEST as V82

DEST=ROOT/'artifacts/v85_protection_20260927'


def tensor_csr(x):
    return torch.sparse_csr_tensor(torch.as_tensor(x.indptr,dtype=torch.int64,device='cuda'),
       torch.as_tensor(x.indices,dtype=torch.int64,device='cuda'),torch.as_tensor(x.data,device='cuda'),size=x.shape)


def gelu_numpy(z):return .5*z*(1+np.tanh(math.sqrt(2/math.pi)*(z+.044715*z*z*z)))


def cpu_manual(x,s):
    w={name:t.numpy() for name,t in s.items()}
    a=gelu_numpy(np.asarray(x@w['first.weight'].T))
    b=gelu_numpy(a@w['second.weight'].T+w['second.bias'])
    return b@w['last.weight'].T+w['last.bias']


def gpu_manual(x,ids,state):
    w={name:t.cuda() for name,t in state.items()};result=np.empty((len(ids),3),np.float32)
    def gelu(z):return .5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z*z*z)))
    with torch.no_grad():
        for j in range(0,len(ids),4096):
            a=gelu(torch.sparse.mm(tensor_csr(x[ids[j:j+4096]]),w['first.weight'].T))
            b=gelu(a@w['second.weight'].T+w['second.bias'])
            result[j:j+4096]=(b@w['last.weight'].T+w['last.bias']).cpu().numpy()
    return result


def direct(y,old,new):
    cm=np.bincount(y*3+new,minlength=9).reshape(3,3)
    p=(old!=y)&(new==y);n=(old==y)&(new!=y)
    return {'cm':cm.tolist(),'errors':int((new!=y).sum()),'positive_flips':int(p.sum()),'negative_flips':int(n.sum()),
        'positive_flips_by_class':np.bincount(y[p],minlength=3).tolist(),
        'negative_flips_by_class':np.bincount(y[n],minlength=3).tolist(),
        'wrong_to_different_wrong':int(((old!=y)&(new!=y)&(old!=new)).sum()),'changed_decisions':int((old!=new).sum())}


def compare_fields(actual,claimed):
    for k,value in actual.items():assert value==claimed[k],(k,value,claimed[k])


def main():
    if (DEST/'execution_receipt.json').exists():raise FileExistsError('Published training frozen')
    start=time.monotonic();reg=read(DEST/'registration.json')
    assert reg['source_sha256']==sha(ROOT/'training/v85_protection.py')
    for relative,digest in reg['input_sha256'].items():assert sha(ROOT/relative)==digest,relative
    assert sha(DEST/'manifest.parquet')==reg['manifest_sha256']
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    manifest=pd.read_parquet(DEST/'manifest.parquet')
    assert np.array_equal(manifest.row_position,r.row_position) and np.array_equal(manifest.fold,r.fold)
    original_selection=np.zeros(len(r),bool);original_selection[np.load(V82/'selected_rows.npy')]=True
    assert np.array_equal(original_selection,manifest.selected)
    result=[];fits=0;state_count=0;max_cpu_difference=0.;init_hashes=[]
    for folder in sorted(DEST.glob('fold*')):
        h=int(folder.name[4:]);mask=~r.fold.isin([0,2,h]).to_numpy();inner=r.fold.eq(h).to_numpy();sel=original_selection&mask
        actual_selected=np.load(folder/'selected_rows.npy');assert np.array_equal(actual_selected,np.flatnonzero(sel))
        assert np.array_equal(sel&(y>0),mask&(y>0))
        for field in ['component','body_group','source_symbol']:
            role=np.where(~r.fold.isin([0,2]).to_numpy(),np.where(inner,'inner','train'),'locked')
            valid=r[field].ge(0);assert pd.DataFrame({'id':r.loc[valid,field],'role':role[valid]}).groupby('id').role.nunique().max()==1
        teacher=joblib.load(folder/'teacher.joblib');z=np.asarray(x@teacher['coef'])+teacher['intercept']
        saved=np.load(folder/'teacher_scores.npy',mmap_mode='r');assert np.max(abs(z-saved))<1e-10
        old=z.argmax(1).astype(np.int8);assert np.array_equal(old,np.load(folder/'teacher_prediction.npy'))
        tf=read(folder/'teacher_fit.json');assert tf['converged'] and tf['model_sha256']==sha(folder/'teacher.joblib');fits+=1
        for m,name in [(sel,'train'),(inner,'inner')]:
            base=direct(y[m],old[fid[m]],old[fid[m]])
            assert base['cm']==tf[name]['cm'] and base['errors']==tf[name]['errors']
        unique=np.unique(fid[sel]);raw=np.bincount(fid[sel]*3+y[sel],minlength=x.shape[0]*3).reshape(-1,3)
        pcount=raw[np.arange(len(raw)),old];pactive=pcount>0
        oldm=z[np.arange(len(z)),old]-np.max(np.where(np.eye(3,dtype=bool)[old],-np.inf,z),axis=1)
        epsilon=np.minimum(.001,oldm/2)
        assert np.min(oldm[pactive])>0
        fitinfo={'fold':h,'actual_train_rows':int(sel.sum()),'P_original_rows':int(pcount.sum()),'E_original_rows':int(sel.sum()-pcount.sum()),
          'mixed_protected_input_groups':int(((raw.sum(1)>pcount)&pactive).sum()),
          'unavoidable_errors_under_fixed_protected_input_decisions':int((raw.sum(1)-pcount)[pactive].sum()),'snapshots':[]}
        for path in sorted(folder.glob('[ABC]_fit.json')):
            arm=path.name[0];info=read(path);assert info['source_sha256']==reg['source_sha256'];fits+=1;init_hashes.append(info['initial_state_hash'])
            trace=read(folder/(arm+'_selection_trace.json'));assert len(trace)==5
            for stage in trace:
                name=stage['name'];state=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True)
                savedp=np.load(folder/(name+'_predictions.npz'));ids=np.unique(np.concatenate([savedp['fit_feature_ids'],savedp['inner_feature_ids']]))
                delta=gpu_manual(x,ids,state);pred=(z[ids]+delta).argmax(1).astype(np.int8)
                assert np.array_equal(pred[np.searchsorted(ids,savedp['fit_feature_ids'])],savedp['fit_prediction']),name
                assert np.array_equal(pred[np.searchsorted(ids,savedp['inner_feature_ids'])],savedp['inner_prediction']),name
                replay=np.full(x.shape[0],-1,np.int8);replay[ids]=pred
                for m,key in [(sel,'selected_fit'),(inner,'inner')]:compare_fields(direct(y[m],old[fid[m]],replay[fid[m]]),stage[key])
                sample=np.arange(min(256,len(ids)));manual=cpu_manual(x[ids[sample]],state);diff=float(np.max(abs(manual-delta[sample])))
                max_cpu_difference=max(max_cpu_difference,diff);assert diff<1e-4,(name,diff)
                trainids=savedp['fit_feature_ids'];trainz=z[trainids]+delta[np.searchsorted(ids,trainids)]
                newmargin=trainz[np.arange(len(trainids)),old[trainids]]-np.max(np.where(np.eye(3,dtype=bool)[old[trainids]],-np.inf,trainz),axis=1)
                pp=pcount[trainids]>0;viol=(epsilon[trainids]-newmargin)>1e-7
                exact={'violating_input_groups':int((viol&pp).sum()),'protected_negative_flips':int(pcount[trainids][pp&(trainz.argmax(1)!=old[trainids])].sum()),
                    'minimum_protected_margin_float64_teacher':float(newmargin[pp].min())}
                if arm=='C':assert exact['violating_input_groups']==exact['protected_negative_flips']==0
                if stage['eligible']:assert exact['violating_input_groups']==0 and stage['inner']['negative_flips']==0 and stage['inner']['positive_flips']>0
                if stage['epoch']==0:assert np.count_nonzero(delta)==0
                fitinfo['snapshots'].append({'name':name,'raw_original_row_statistics_match':True,'all_saved_decisions_recomputed':True,
                     'manual_cpu_max_difference':diff,'protection_replay':exact,'eligible':stage['eligible']});state_count+=1
            if arm=='C':
                progress=read(folder/(arm+'_progress.json'));assert len(progress)==60
                assert all(p['protection_after_step']['violating_input_groups']==0 and p['protection_after_step']['protected_negative_flips']==0 for p in progress)
            for named in [p.stem.replace('_all_prediction','') for p in folder.glob(arm+'_epoch*_all_prediction.npy')]:
                state=torch.load(folder/(named+'.pt'),map_location='cpu',weights_only=True)
                rp=(z+gpu_manual(x,np.arange(x.shape[0]),state)).argmax(1).astype(np.int8)
                assert np.array_equal(rp,np.load(folder/(named+'_all_prediction.npy'))),named
                dx=next(d for d in read(folder/'diagnosis.json') if d['name']==named)
                for claim in dx['roles']:
                    m=mask if claim['role']=='fit_full' else inner if claim['role']=='inner' else r.fold.eq(2 if claim['role']=='C' else 0).to_numpy()
                    compare_fields(direct(y[m],old[fid[m]],rp[fid[m]]),claim)
                print(json.dumps({'verified_full_snapshot':named,'fold':h}),flush=True)
        result.append(fitinfo)
    assert len(set(init_hashes))==1
    prior={}
    for p in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
          'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
          'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json']:
        prior.update(read(ROOT/p)['artifact_sha256'])
    bad=[p for p,digest in prior.items() if sha(ROOT/p)!=digest];assert not bad,bad
    out={'status':'passed','source_sha256':sha(__file__),'registered_source_sha256':reg['source_sha256'],
      'actual_classifier_fits':fits,'actual_calibration_fits':0,'saved_training_snapshots_recomputed':state_count,
      'manual_cpu_max_difference':max_cpu_difference,'old_bound_files_rehashed':len(prior),'old_bound_files_changed':bad,
      'registered_inputs_rehashed':len(reg['input_sha256']),'raw_original_unchanged':True,'fold_checks':result,
      'seconds':time.monotonic()-start,'scope':'All saved fit/inner predictions and diagnosed full snapshots replayed with saved weights and an explicit independent GELU implementation; all statistics recomputed on original rows. CPU sparse/dense formula cross-check on 256 inputs per snapshot. Unsaved epoch feasibility relies on recorded checks. Not unseen transfer acceptance.'}
    save(DEST/'verification.json',out);print(json.dumps({k:v for k,v in out.items() if k!='fold_checks'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
