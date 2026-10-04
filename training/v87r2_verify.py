"""Independent original-row, saved-weight, exposure and pair verification."""
import json
import re
import time
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,OUT,read,save,sha,load_sparse
from v79_execute import rows
from v82_capacity import LAST,DEST as V82
from v85_verify import gpu_manual,cpu_manual,direct,compare_fields

DEST=ROOT/'artifacts/v87_solver_supervision_r2_20260927'
OLD=ROOT/'artifacts/v85_protection_20260927'
ARMS=['A0','A1','A2','A3']


def verify_pairs(r,y,fid,full,selected,folder=None):
    folder=DEST/'fold1' if folder is None else folder
    pairs=pd.read_parquet(folder/'pairs.parquet')
    facts=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet',columns=['facts']).facts]
    raw=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['message_sanitized','label_binary'])
    truth=raw.label_binary.map({'benign':0,'malicious':1,'suspicious':2})
    assert truth.notna().all() and np.array_equal(truth.to_numpy(),y)
    selected_set=set(selected.tolist());pure=(full>0).sum(1)==1
    fields=['action','outcome','transport_protocol','src_role','dst_role']
    pattern=re.compile(r'\bDeny\s+(tcp|udp)\s+src\s+(\w[\w-]*):((?:\d{1,3}\.){3}\d{1,3})/(\d+)\s+dst\s+(\w[\w-]*):((?:\d{1,3}\.){3}\d{1,3})/(\d+)\s+by\s+.+?\[0x0,\s*0x0\]\s*$',re.I)
    checked_raw=set()
    for p in pairs.itertuples(index=False):
        a,b,c=[int(getattr(p,n)) for n in ['anchor_row','positive_row','negative_row']]
        assert all(t in selected_set for t in [a,b,c])
        assert y[a]==y[b] and y[a]!=y[c] and y[a] in [1,2]
        assert fid[a]!=fid[b] and fid[a]!=fid[c]
        assert r.component.iloc[a]!=r.component.iloc[b] and r.component.iloc[a]!=r.component.iloc[c]
        ff=[facts[int(r.projection_id.iloc[t])] for t in [a,b,c]]
        keys=fields+(['dst_port_fixed'] if p.grade=='fine' else [])
        assert p.grade in ['fine','coarse']
        assert all(all(k in f for k in keys) for f in ff)
        assert all(tuple(str(f[k]) for k in keys)==tuple(str(ff[0][k]) for k in keys) for f in ff)
        for t,f in zip([a,b,c],ff):
            assert pure[fid[t]] and r.route.iloc[t]=='asa'
            assert int(fid[t])==int(getattr(p,['anchor_fid','positive_fid','negative_fid'][[a,b,c].index(t)]))
            if t not in checked_raw:
                m=pattern.search(raw.message_sanitized.iloc[t]);assert m is not None,t
                assert f['action']=='deny' and f['outcome']=='blocked' and m[1].lower()==f['transport_protocol']
                assert int(m[4])==int(f['src_port_fixed']) and int(m[7])==int(f['dst_port_fixed'])
                checked_raw.add(t)
    return {'triples_checked':len(pairs),'unique_raw_rows_checked':len(checked_raw),
            'original_train_labels_checked':len(raw),'all_pairs_train_only':True,
            'all_pairs_crosscomponent_and_input_conflict_filtered':True,
            'scope':'Independent raw syntax, exact ports/protocol, true class, role and behavior-key agreement; not incident/intent equivalence.'}


def main():
    start=time.monotonic();reg=read(DEST/'registration.json')
    assert not (DEST/'execution_receipt.json').exists()
    for file,digest in reg['source_bindings'].items():assert sha(ROOT/file)==digest,file
    assert reg['source_sha256']==sha(ROOT/'training/v87r2_execute.py')
    r=rows();y=r.label_index.to_numpy();fid=np.load(LAST/'row_feature_id.npy');x=load_sparse(LAST/'X')
    fit=~r.fold.isin([0,1,2]).to_numpy();inner=r.fold.eq(1).to_numpy()
    original=np.zeros(len(r),bool);original[np.load(V82/'selected_rows.npy')]=True;sel=original&fit
    selected=np.flatnonzero(sel);folder=DEST/'fold1'
    assert np.array_equal(np.load(folder/'selected_rows.npy'),selected)
    assert np.array_equal(sel&(y>0),fit&(y>0))
    mf=pd.read_parquet(DEST/'manifest.parquet')
    assert np.array_equal(mf.selected,sel) and np.array_equal(mf.fold,r.fold)
    for field in ['component','body_group','source_symbol']:
        take=r[field]>=0
        assert pd.DataFrame({'id':r.loc[take,field],'role':mf.role[take]}).groupby('id').role.nunique().max()==1
    teacher=joblib.load(OLD/'fold1/teacher.joblib');z=np.asarray(x@teacher['coef'])+teacher['intercept']
    np.testing.assert_allclose(z,np.load(OLD/'fold1/teacher_scores.npy',mmap_mode='r'),atol=1e-10,rtol=0)
    old=z.argmax(1).astype(np.int8);assert np.array_equal(old,np.load(OLD/'fold1/teacher_prediction.npy'))
    full=np.bincount(fid[fit]*3+y[fit],minlength=x.shape[0]*3).reshape(-1,3)
    pmass=full[np.arange(x.shape[0]),old];pid=np.flatnonzero(pmass)
    assert np.array_equal(pid,np.load(folder/'protection_fids.npy'))
    assert np.array_equal(pmass[pid],np.load(folder/'protection_original_mass.npy'))
    oldm=z[pid,old[pid]]-np.max(np.where(np.eye(3,dtype=bool)[old[pid]],-np.inf,z[pid]),axis=1)
    eps=np.minimum(.001,oldm/2);np.testing.assert_array_equal(eps,np.load(folder/'protection_epsilon.npy'))
    paircheck=verify_pairs(r,y,fid,full,selected)
    roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    count=0;fits=0;max_cpu=0.;states=[];full_replays=0
    for arm in ARMS:
        info=read(folder/(arm+'_fit.json'));assert info['source_sha256']==reg['source_sha256'];fits+=info['actual_classifier_fits']
        progress=read(folder/(arm+'_progress.json'));assert len(progress)==info['attempts']<=reg['max_attempts']
        for item in progress:
            assert item['protection']['protected_negative_flips']==item['protection']['violating_input_groups']==0
            for qp in item['QP_records']:
                if qp.get('kkt_passed'):assert qp['actual_scaled_primal_violation']<=reg['QP_tolerance']
        trace=read(folder/(arm+'_selection_trace.json'))
        for claim in trace:
            name=claim['name'];bundle=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True);state=bundle['model']
            assert bundle['source_sha256']==reg['source_sha256'] and bundle['epoch']==claim['epoch']
            accepted=sum(p['accepted'] for p in progress[:claim['epoch']])
            for values in bundle['optimizer']['state'].values():assert int(values['step'])==accepted
            saved=np.load(folder/(name+'_predictions.npz'))
            ids=np.unique(np.concatenate([saved['fit_feature_ids'],saved['inner_feature_ids'],pid]))
            delta=gpu_manual(x,ids,state);pred=(z[ids]+delta).argmax(1).astype(np.int8)
            for key in ['fit','inner']:
                assert np.array_equal(pred[np.searchsorted(ids,saved[key+'_feature_ids'])],saved[key+'_prediction']),name
            replay=np.full(x.shape[0],-1,np.int8);replay[ids]=pred
            compare_fields(direct(y[sel],old[fid[sel]],replay[fid[sel]]),claim['selected_fit'])
            compare_fields(direct(y[inner],old[fid[inner]],replay[fid[inner]]),claim['inner'])
            probe=np.linspace(0,len(ids)-1,min(512,len(ids))).astype(int)
            cpu=cpu_manual(x[ids[probe]],state);difference=float(np.max(abs(cpu-delta[probe])))
            max_cpu=max(max_cpu,difference);assert difference<1e-4,(name,difference)
            zp=z[pid]+delta[np.searchsorted(ids,pid)]
            newm=zp[np.arange(len(pid)),old[pid]]-np.max(np.where(np.eye(3,dtype=bool)[old[pid]],-np.inf,zp),axis=1)
            assert int(pmass[pid][zp.argmax(1)!=old[pid]].sum())==0,name
            assert ((eps-newm)>reg['nonlinear_margin_tolerance']).sum()==0,name
            if claim['epoch']==0:assert np.count_nonzero(delta)==0
            states.append({'name':name,'original_row_statistics_match':True,'full_train_protection_verified':True,
                           'optimizer_steps_verified':accepted,'CPU_probe_max_difference':difference});count+=1
            print(json.dumps({'verified_state':name,'CPU_probe_difference':difference}),flush=True)
            del bundle,state,delta;torch.cuda.empty_cache()
        for path in folder.glob(arm+'_epoch*_all_prediction.npy'):
            name=path.stem.replace('_all_prediction','');bundle=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=True)
            replay=(z+gpu_manual(x,np.arange(x.shape[0]),bundle['model'])).argmax(1).astype(np.int8)
            assert np.array_equal(replay,np.load(path)),name
            dx=next(d for d in read(folder/'diagnosis.json') if d['name']==name)
            for claim in dx['roles']:
                m=roles[claim['role']];compare_fields(direct(y[m],old[fid[m]],replay[fid[m]]),claim)
            full_replays+=1;del bundle;torch.cuda.empty_cache()
    assert fits==4
    wrong_positions=np.flatnonzero(fit&(old[fid]!=y))
    conflicts=np.isin(fid[wrong_positions],pid)
    feasibility={'old_train_errors':len(wrong_positions),
        'same_encoded_input_has_protected_correct_rows':int(conflicts.sum()),
        'no_same_input_protected_correct_rows':int((~conflicts).sum()),
        'blocked_per_class':np.bincount(y[wrong_positions[conflicts]],minlength=3).tolist(),
        'unblocked_per_class':np.bincount(y[wrong_positions[~conflicts]],minlength=3).tolist(),
        'unblocked_unique_inputs':int(np.unique(fid[wrong_positions[~conflicts]]).size)}
    prior={}
    for file in ['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json',
       'evidence/2026-09-27/v81_diagnosis/delivery.json','evidence/2026-09-27/v82_capacity/delivery.json',
       'evidence/2026-09-27/v83_root_review/delivery.json','evidence/2026-09-27/v84_preservation/delivery.json',
       'evidence/2026-09-27/v85_protection/delivery.json','evidence/2026-09-27/v86_boundary_review/delivery.json']:
        prior.update(read(ROOT/file)['artifact_sha256'])
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    out={'status':'passed','source_sha256':sha(__file__),'registered_source_sha256':reg['source_sha256'],
         'actual_classifier_fits':fits,'actual_calibration_fits':0,'new_teacher_fits':0,
         'all_saved_states_replayed':count,'all_input_final_models_replayed':full_replays,
         'CPU_probe_inputs_per_state':512,'CPU_max_difference':max_cpu,'states':states,
         'same_input_feasibility':feasibility,'pair_verification':paircheck,'protected_original_rows':int(pmass.sum()),
         'registered_files_rehashed':len(reg['source_bindings']),'old_bound_files_rehashed':len(prior),'old_bound_files_changed':changed,
         'seconds':time.monotonic()-start,
         'scope':'All saved fit/inner decisions, all full-P boundaries and diagnosed final full-input decisions replayed using independent saved-weight formulas; original rows and true labels checked. CPU probes per state. Unsaved epoch constraints rely on live checks recorded in progress, not retrospective every-epoch weight replay. Not unseen transfer acceptance.'}
    save(DEST/'verification.json',out);print(json.dumps({k:v for k,v in out.items() if k!='states'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=4):main()
