"""Audit saved V158 OOF capacity and observable input identities; zero fitting.

No project feature functions or classifiers are called. Sparse row hashing is
identity evidence only and is not supplied to a trained classifier.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
RUN = ROOT/'artifacts/v158_fusion_trial_20261001'
OUT = ROOT/'artifacts/v159_independent_OOF_capacity_and_input_review_20261001'


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def identity(x):
    x=x.astype(np.float32).tocsr(copy=True)
    x.sum_duplicates();x.sort_indices();x.eliminate_zeros()
    return [hashlib.sha256(x.indices[x.indptr[i]:x.indptr[i+1]].astype('<i8').tobytes()+
                x.data[x.indptr[i]:x.indptr[i+1]].astype('<f4').tobytes()).hexdigest()
            for i in range(x.shape[0])]


def empirical_floor(frame,keys):
    count=frame.assign(observation_key=keys).groupby(['observation_key','truth']).size().unstack(fill_value=0)
    return dict(groups=len(count),mixed_groups=int((count.gt(0).sum(1)>1).sum()),
                minimum_original_errors=int((count.sum(1)-count.max(1)).sum()))


def main():
    assert not OUT.exists()
    assert (RUN/'final_delivery.json').is_file()
    current_path=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    legacy_path=ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
    key_path=ROOT/'artifacts/v130_learning_review_20260930_r2/training_role_error_ledger.parquet'
    trace_path=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    gold_path=ROOT/'data/official/train.parquet'
    seal=read(RUN/'run_seal.json')
    for p in [current_path,legacy_path,key_path,trace_path,gold_path]:
        rel=p.relative_to(ROOT).as_posix();assert rel in seal['source_sha256'], rel
        assert sha(p)==seal['source_sha256'][rel], rel
    # The physical legacy input has its own original content; equal dimensionality
    # is not accepted as evidence of numerical/input-policy equivalence.
    x=sparse.load_npz(current_path)
    old=sparse.load_npz(legacy_path)
    assert x.shape==old.shape==(22546,66287)
    current_keys=identity(x);legacy_keys=identity(old)
    key_frame=pd.read_parquet(key_path,columns=['local','canonical_key']).drop_duplicates()
    assert key_frame.local.is_unique
    lookup=key_frame.set_index('local').canonical_key
    assert all(current_keys[i]==lookup.loc[i] for i in range(22546))
    trace=pd.read_parquet(trace_path,columns=['row_position','local','root','fold','truth'])
    gold=pd.read_parquet(gold_path,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
    assert len(gold)==2056871 and len(trace)==112807
    assert np.array_equal(gold[trace.row_position],trace.truth)
    union_keys=[hashlib.sha256((a+'|'+b).encode()).hexdigest() for a,b in zip(current_keys,legacy_keys)]
    input_diff=(x.astype(np.float64)-old.astype(np.float64)).tocsr();input_diff.eliminate_zeros()
    differs=np.diff(input_diff.indptr)>0
    key_meta=pd.DataFrame(dict(local=np.arange(22546),current_key=current_keys,legacy_key=legacy_keys,union_key=union_keys,physical_input_differs=differs))
    results=[];ledgers=[];sources=[]
    for f in range(3):
        frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet')
        canonical=frame.local.map(dict(enumerate(current_keys)))
        assert np.array_equal(canonical,frame.canonical_key)
        assert frame.fold.ne(f).all() and not set(frame.root)&set(trace.loc[trace.fold.eq(f),'root'])
        inner=frame.root.map(lambda r:int.from_bytes(hashlib.sha256(f'V128|split=12801|outer={f}|root={r}'.encode()).digest()[:8],'big')%3)
        assert np.array_equal(inner,frame.inner_fold)
        bank_path=BANK/f'fold{f}/OOF_probabilities.npy'
        assert sha(bank_path)==seal['source_sha256'][bank_path.relative_to(ROOT).as_posix()]
        bank=np.load(bank_path);p=bank[frame.local]
        assert p.shape==(len(frame),17,3) and np.isfinite(p).all()
        yy=frame.truth.to_numpy();true=np.take_along_axis(p,np.broadcast_to(yy[:,None,None],(len(frame),17,1)),axis=2)[...,0]
        other=np.take_along_axis(p,np.broadcast_to((3-yy)[:,None,None],(len(frame),17,1)),axis=2)[...,0]
        max_margin=(true-other).max(1);blocked=max_margin<0
        any_correct=(p.argmax(2)==yy[:,None]).any(1)
        plain=p[:,:16].mean(1).argmax(1)
        augmented=[hashlib.sha256(frame.canonical_key.iloc[i].encode()+p[i,-1].astype('<f8').tobytes()).hexdigest() for i in range(len(frame))]
        floors=dict(current_complete_input=empirical_floor(frame,canonical),
            legacy_complete_input=empirical_floor(frame,frame.local.map(dict(enumerate(legacy_keys)))),
            physical_union_input=empirical_floor(frame,frame.local.map(dict(enumerate(union_keys)))),
            current_input_plus_new_OOF_legacy_scores=empirical_floor(frame,augmented))
        class_stats={}
        for cl in [1,2]:
            mask=yy==cl
            class_stats[str(cl)]=dict(original_rows=int(mask.sum()),
                all17_wrong_negative_margin_rows=int((mask&blocked).sum()),
                all17_without_correct_expert_rows=int((mask&~any_correct).sum()),
                current16_mean_wrong=int((mask&(plain!=yy)).sum()))
            for arm in 'AB':
                aa=pd.read_parquet(RUN/f'fold{f}_{arm}/endpoint_original_OOF_rows.parquet')
                assert np.array_equal(aa.row_position,frame.row_position) and np.array_equal(aa.truth,yy)
                bad=aa.pred.to_numpy()!=yy
                class_stats[str(cl)][arm+'_wrong']=int((mask&bad).sum())
                class_stats[str(cl)][arm+'_wrong_proven_convex_blocked']=int((mask&bad&blocked).sum())
                class_stats[str(cl)][arm+'_wrong_with_correct_expert']=int((mask&bad&any_correct).sum())
        part=frame.copy();part['some_expert_correct']=any_correct;part['strict_convex_wrong_margin']=blocked
        part['best_true_vs_other_margin']=max_margin;part['physical_union_key']=frame.local.map(dict(enumerate(union_keys)))
        part['current16_mean_pred']=plain
        for arm in 'AB':part['pred_'+arm]=pd.read_parquet(RUN/f'fold{f}_{arm}/endpoint_original_OOF_rows.parquet').pred.to_numpy()
        ledgers.append(part)
        group=part.groupby(['root','truth']).agg(original_rows=('truth','size'),blocked_rows=('strict_convex_wrong_margin','sum'))
        sources.append(group.reset_index().assign(training_role=f))
        results.append(dict(training_role=f,original_rows=len(frame),original_class_mass=np.bincount(yy,minlength=3).tolist(),
            input_floors=floors,class_capacity=class_stats,query_root_and_outer_exclusion_rebuilt=True))
    result=dict(status='saved_OOF_capacity_and_complete_sparse_input_identities_verified',latest_actual_training='V158',
        new_classifier_or_feature_functions=0,new_fits=0,new_gradients=0,new_updates=0,
        rows_physical_current_legacy_input_differ=int(differs.sum()),roles=results,
        scope=['Exact observed numerical equality audits, not proof of causal security evidence or external Bayes risk.',
            'OOF scores depend on different source-excluded fitted experts; their collision floor does not describe raw-data identifiability.',
            'Legacy/full union includes historical header differences; it is not approved as a future classifier input.',
            'No features or thresholds are selected from the observed outer errors.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),current_path,legacy_path,key_path,trace_path,gold_path,RUN/'run_seal.json']})
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    key_meta.to_parquet(OUT/'all_sparse_input_identities.parquet',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'all_original_OOF_capacity_rows.parquet',index=False)
    pd.concat(sources,ignore_index=True).to_parquet(OUT/'all_source_class_capacity_rows.parquet',index=False)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
