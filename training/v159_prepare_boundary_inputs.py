"""Saved-array identities/floors only; no head construction or model calls."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from v159_boundary_runtime import ROOT,PREP,BANK,INPUT,review,save,sha,read
from experiment_review import check_bindings

def visible_keys(x,p,ids):
    result={}
    for i in ids:
        a,b=x.indptr[i:i+2]
        content=x.indices[a:b].astype('<i8').tobytes()+x.data[a:b].astype('<f4').tobytes()+p[i][np.lexsort((p[i,:,2],p[i,:,1],p[i,:,0]))].astype('<f8').tobytes()
        result[int(i)]=hashlib.sha256(content).hexdigest()
    return result

def floors(frame,p,key,x_key,scope):
    r=frame[['row_position','local','truth','root','fold','canonical_key','inner_fold']].copy()
    r['visible_input_key']=r.local.map(key);r['current_input_key']=r.local.map(x_key);r['initial_pred']=p[r.local].mean(1).argmax(1);r['initial_correct']=r.truth.eq(r.initial_pred)
    grouped=r.groupby('visible_input_key')
    raw=forced=0
    for _,g in grouped:
        counts=np.bincount(g.truth,minlength=3);raw+=len(g)-int(counts.max())
        protected=g.loc[g.initial_correct,'initial_pred'].unique();assert len(protected)<=1
        forced+=len(g)-int(counts[int(protected[0])]) if len(protected) else len(g)-int(counts.max())
    pure=grouped.truth.nunique().eq(1);r['pure_visible_input']=r.visible_input_key.map(pure)
    xgroups=r.groupby('current_input_key');xpure=xgroups.truth.nunique().eq(1);r['pure_current_input']=r.current_input_key.map(xpure)
    xfloor=sum(len(g)-int(np.bincount(g.truth,minlength=3).max()) for _,g in xgroups)
    r['protected_correct']=r.initial_correct if scope=='deployment' else r.initial_correct&r.pure_current_input
    return r,dict(unconstrained_minimum_original_errors=raw,all_initial_correct_guard_constrained_minimum_errors=forced,
        initial_correct_guard_constrained_minimum_errors=forced if scope=='deployment' else raw,
        guard_and_unconstrained_floor_compatible=(raw==forced if scope=='deployment' else True),selected_retention_and_registered_current_input_target_compatible=(forced<=xfloor if scope=='deployment' else raw<=xfloor),current_input_minimum_errors=xfloor,protected_initial_correct_original_rows=int(r.protected_correct.sum()),initial_correct_original_rows=int(r.initial_correct.sum()),
        initial_errors=int((~r.initial_correct).sum()),pure_original_rows=int(r.pure_visible_input.sum()),mixed_original_rows=int((~r.pure_visible_input).sum()),visible_groups=len(pure))

def main():
    plan=review();assert not PREP.exists();PREP.mkdir()
    paths=[Path(__file__).resolve(),INPUT,ROOT/'training/review_policy/v159_boundary_execution_contract.json']
    for f in range(3):paths.extend(BANK/f'fold{f}'/n for n in ['legal_FIT_reference.parquet','OOF_probabilities.npy','deployment_probabilities.npy'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(PREP/'pre_array_bindings.json',dict(status='before_array_decode_no_classifier_or_feature',source_sha256=bindings))
    old=read(ROOT/'artifacts/v158_fusion_trial_20261001/run_seal.json')['source_sha256']
    for p in paths:
        name=p.relative_to(ROOT).as_posix()
        if name in old:assert bindings[name]==old[name]
    x=load_npz(INPUT).tocsr();x.sort_indices();xkeys={i:hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest() for i in range(22546)};assert x.shape==(22546,66287) and x.dtype==np.float32 and np.isfinite(x.data).all()
    roles=[]
    for f in range(3):
        folder=PREP/f'fold{f}';folder.mkdir();frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet');ids=np.sort(frame.local.unique())
        assert (frame.fold!=f).all() and frame.truth.isin([1,2]).all() and frame.groupby('root').inner_fold.nunique().eq(1).all()
        counts=np.bincount(frame.local.to_numpy(np.int64)*3+frame.truth.to_numpy(np.int64),minlength=22546*3).reshape(22546,3)
        assert counts.sum()==len(frame) and not counts[:,0].any()
        info={}
        for scope in ['OOF','deployment']:
            p=np.load(BANK/f'fold{f}/{scope}_probabilities.npy',mmap_mode='r')[:,:16]
            check_ids=ids if scope=='OOF' else np.arange(22546)
            assert p.shape==(22546,16,3) and np.isfinite(p[check_ids]).all() and (p[check_ids]>=0).all() and np.abs(p[check_ids].sum(-1)-1).max()<=1e-12
            keys=visible_keys(x,p,ids);rows,info[scope]=floors(frame,p,keys,xkeys,scope)
            rows['training_role']=f;rows.to_parquet(folder/f'{scope}_visible_input_rows.parquet',index=False)
            raw=p[check_ids].mean(1);fixed=np.maximum(raw,1e-12);fixed/=fixed.sum(-1,keepdims=True)
            assert np.abs(raw-fixed).max()<=3e-12 and np.array_equal(raw.argmax(1),fixed.argmax(1))
            info[scope]['saved_array_origin_probability_gap']=float(np.abs(raw-fixed).max())
        assert info['deployment']['current_input_minimum_errors']==[22,6,28][f]
        roles.append(dict(role=f,original_rows=len(frame),original_class_mass=counts.sum(0).tolist(),locals=len(ids),scopes=info))
    check_bindings(bindings)
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),PREP/'pre_array_bindings.json']}
    for f in range(3):sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in (PREP/f'fold{f}').glob('*')})
    save(PREP/'qualification.json',dict(status='complete_candidate_visible_input_and_guard_floors_from_saved_arrays',roles=roles,
        input_qualification_passed=all(r['scopes'][s]['selected_retention_and_registered_current_input_target_compatible'] for r in roles for s in ['OOF','deployment']),
        official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,quality_acceptance=False,
        source_sha256=sources,prospective_array_sources=bindings))
    print(json.dumps(dict(status='input_prepared_without_official_model_function',roles=roles),ensure_ascii=False))

if __name__=='__main__':main()
