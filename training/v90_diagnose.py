"""Frozen v89 diagnostics: representation aliases, unseen tokens, bounded output path.

No fitting, threshold selection, checkpoint selection or model promotion.
"""
import collections,hashlib,json,time
import numpy as np,pandas as pd
from sklearn.feature_extraction import FeatureHasher
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST as V89,OLD,LAST,read,save,sha
from v89_partial_expression import tokens,dictionary_features

DEST=ROOT/'artifacts/v90_root_review_20260928'
SCALES=[0.,.01,.05,.1,.25,.5,.75,1.]


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'diagnosis.json').exists();start=time.monotonic()
    receipt=read(ROOT/'evidence/2026-09-27/v89_readout_support/delivery.json')
    for p in ['training/v89_partial_expression.py','artifacts/v89_readout_support_20260927/F00_model.npz','artifacts/v89_readout_support_20260927/F00_all_prediction.npy']:
        assert sha(ROOT/p)==receipt['artifact_sha256'][p]
    r=pd.read_parquet(OUT/'rows.parquet');y=r.label_index.to_numpy();fit=~r.fold.isin([0,1,2]).to_numpy();inner=r.fold.eq(1).to_numpy()
    obs=np.load(V89/'row_fact_code.npy');observations=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    tokenlists=[sorted(tokens(s)) for s in observations];dz=dictionary_features(observations)
    gid=np.load(V89/'F00_row_group.npy');pairs=np.load(V89/'F00_group_keys.npy');z=np.load(OLD/'teacher_scores.npy',mmap_mode='r');basez=np.asarray(z[pairs[:,0]]);op=basez.argmax(1);old=op[gid]
    m=np.load(V89/'F00_model.npz');coef=m['theta'][:768].reshape(3,256).T/m['scale'][:,None];delta=np.asarray(dz@coef);new=(basez+delta[pairs[:,1]]).argmax(1)[gid]
    assert np.array_equal(new,np.load(V89/'F00_all_prediction.npy')[gid])
    regressed=inner&(y==1)&(old==y)&(new!=y);repaired=inner&(y==2)&(old!=y)&(new==y)
    assert regressed.sum()==668 and repaired.sum()==38
    canonical=[json.dumps(t,separators=(',',':')) for t in tokenlists];tok_id,unique_tokens=pd.factorize(canonical,sort=True)
    signatures=[]
    for i in range(len(observations)):
        a,b=dz.indptr[i:i+2];signatures.append(hashlib.sha256(dz.indices[a:b].astype('<i4').tobytes()+dz.data[a:b].astype('<f8').tobytes()).hexdigest())
    hash_id,unique_hash=pd.factorize(signatures,sort=True)
    aliases=collections.defaultdict(set)
    for t,h in zip(tok_id,hash_id):aliases[int(h)].add(int(t))
    hashed_alias=np.array([len(aliases[int(h)])>1 for h in hash_id])
    def mass(key,mask,n):return np.bincount(key[obs[mask]]*3+y[mask],minlength=n*3).reshape(n,3)
    exact_counts=mass(tok_id,fit,len(unique_tokens));hash_counts=mass(hash_id,fit,len(unique_hash))
    train_observations=np.unique(obs[fit]);vocabulary=set(t for o in train_observations for t in tokenlists[o]);alltokens=sorted(set(t for a in tokenlists for t in a))
    hasher=FeatureHasher(n_features=128,input_type='dict',alternate_sign=False);unit=hasher.transform([{t:1.} for t in alltokens]);bucket={t:int(unit.indices[unit.indptr[i]])+(0 if t.startswith('asa:') else 128) for i,t in enumerate(alltokens)}
    unseen=np.zeros(len(observations),int);unseen_contribution=np.zeros((len(observations),3));reconstructed=np.zeros_like(delta)
    for o,tks in enumerate(tokenlists):
        if not tks:continue
        bins=np.array([bucket[t] for t in tks]);norm=np.sqrt(np.square(np.bincount(bins,minlength=256)).sum());reconstructed[o]=coef[bins].sum(0)/norm
        missing=[bucket[t] for t in tks if t not in vocabulary];unseen[o]=len(missing)
        if missing:unseen_contribution[o]=coef[missing].sum(0)/norm
    assert np.allclose(reconstructed,delta,atol=1e-11,rtol=1e-12)
    assert not unseen[obs[fit]].any()
    roles={'fit_full':fit,'inner':inner,'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()};curves=[]
    def evaluate(label,pred):
        for role,take in roles.items():
            cm=np.bincount(y[take]*3+pred[take],minlength=9).reshape(3,3)
            rows=[]
            for cls in [0,1,2]:
                takeclass=take&(y==cls);rows.append({'class':cls,'support':int(takeclass.sum()),'correct':int((pred[takeclass]==cls).sum()),'repairs':int((takeclass&(old!=y)&(pred==y)).sum()),'regressions':int((takeclass&(old==y)&(pred!=y)).sum())})
            curves.append({'state':label,'role':role,'errors':int((take&(pred!=y)).sum()),'cm':cm.tolist(),'classwise':rows})
    for alpha in SCALES:evaluate('fixed_scale_'+str(alpha),(basez+alpha*delta[pairs[:,1]]).argmax(1)[gid])
    neutral=(basez+(delta-unseen_contribution)[pairs[:,1]]).argmax(1)[gid];evaluate('unseen_token_contribution_zero_diagnostic',neutral)
    assert np.array_equal(neutral[fit],new[fit])
    details=[]
    baseline_row_z=z[pairs[gid,0]]
    for label,mask in [('M_regression',regressed),('S_repair',repaired)]:
        for o in np.unique(obs[mask]):
            ii=np.flatnonzero(mask&(obs==o));marg=baseline_row_z[ii,1]-baseline_row_z[ii,2]
            details.append({'kind':label,'observation':int(o),'original_rows':len(ii),'input_groups':len(np.unique(gid[ii])),'components':int(r.component.iloc[ii].nunique()),
                'known_facts':json.loads(observations.iloc[o])['facts'],'canonical_train_class_counts':exact_counts[tok_id[o]].tolist(),'hash_train_class_counts':hash_counts[hash_id[o]].tolist(),
                'distinct_token_sets_sharing_full_hash_vector':len(aliases[int(hash_id[o])]),'unseen_token_count':int(unseen[o]),'tokens':tokenlists[o],
                'old_M_minus_S_min':float(marg.min()),'old_M_minus_S_median':float(np.median(marg)),'old_M_minus_S_max':float(marg.max()),
                'added_S_minus_M':float(delta[o,2]-delta[o,1]),'unseen_added_S_minus_M':float(unseen_contribution[o,2]-unseen_contribution[o,1]),
                'unseen_zero_restored_M':int((neutral[ii]==1).sum()) if label=='M_regression' else None,'representative_row':int(ii[0])})
    counts={}
    for label,mask in [('M_regression',regressed),('S_repair',repaired)]:
        os=obs[mask];ec=exact_counts[tok_id[os]];hc=hash_counts[hash_id[os]]
        counts[label]={'rows':int(mask.sum()),'observation_groups':len(np.unique(os)),'components':int(r.component[mask].nunique()),'whole_vector_alias_rows':int(hashed_alias[os].sum()),
            'unseen_token_rows':int((unseen[os]>0).sum()),'same_exact_tokens_M_support_rows':int((ec[:,1]>0).sum()),'same_exact_tokens_S_support_rows':int((ec[:,2]>0).sum()),
            'same_exact_tokens_no_training_support_rows':int((ec.sum(1)==0).sum()),'false_hash_only_S_support_rows':int(((ec[:,2]==0)&(hc[:,2]>0)).sum())}
    # Decompose the observed score shifts by buckets, including unrelated token sharing.
    vocab_per_bucket=collections.defaultdict(set)
    for t in vocabulary:vocab_per_bucket[bucket[t]].add(t)
    top=[]
    for b in np.argsort(-(coef[:,2]-coef[:,1]))[:15]:
        top.append({'bucket':int(b),'S_minus_M_coefficient_before_input_normalization':float(coef[b,2]-coef[b,1]),'training_literal_tokens_sharing_bucket':len(vocab_per_bucket[int(b)]),
            'training_token_examples':sorted(vocab_per_bucket[int(b)])[:12]})
    save(DEST/'fixed_model_counterfactuals.json',{'new_fits':0,'predefined_scales':SCALES,'states':curves,'selected_state':None,'scope':'Exploratory frozen coefficient interpolation and known-training-vocabulary counterfactual; inspected development roles, no calibration fit or candidate selection.'})
    save(DEST/'error_observation_groups.json',details)
    result={'status':'complete','source_sha256':sha(__file__),'actual_new_classifier_fits':0,'actual_calibration_fits':0,'scope':'Frozen v89 no-fit root-cause diagnostics; no unseen validation claim, no selected coefficients or new promoted model.',
        'base_F00_original_row_predictions_reproduced':len(y),'observation_sets':len(observations),'canonical_token_sets':len(unique_tokens),'distinct_full_hashed_vectors':len(unique_hash),
        'hash_alias_groups':sum(len(a)>1 for a in aliases.values()),'all_literal_tokens':len(alltokens),'training_literal_tokens':len(vocabulary),'train_hash_buckets_used':len(vocab_per_bucket),
        'target_counts':counts,'largest_S_shifting_buckets':top,'seconds':time.monotonic()-start,
        'input_bindings':{p:receipt['artifact_sha256'][p] for p in ['training/v89_partial_expression.py','artifacts/v89_readout_support_20260927/F00_model.npz','artifacts/v89_readout_support_20260927/F00_all_prediction.npy']}}
    save(DEST/'diagnosis.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
