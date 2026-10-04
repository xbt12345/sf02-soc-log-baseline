"""Fixed F00 score decomposition and objective audit. No optimizer or tuning."""
import collections,json,time
import numpy as np,pandas as pd
from sklearn.feature_extraction import FeatureHasher
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST as V89,OLD,PRIOR,save,sha
from v89_partial_expression import tokens,dictionary_features
DEST=ROOT/'artifacts/v90_root_review_20260928'

def main():
    assert not (DEST/'score_decomposition.json').exists();start=time.monotonic()
    r=pd.read_parquet(OUT/'rows.parquet');y=r.label_index.to_numpy();fit=~r.fold.isin([0,1,2]).to_numpy();inner=r.fold.eq(1).to_numpy()
    obs=np.load(V89/'row_fact_code.npy');s=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    gid=np.load(V89/'F00_row_group.npy');keys=np.load(V89/'F00_group_keys.npy');z=np.load(OLD/'teacher_scores.npy',mmap_mode='r')[keys[:,0]]
    m=np.load(V89/'F00_model.npz');theta=m['theta'];coef=theta[:768].reshape(3,256).T/m['scale'][:,None]
    dz=dictionary_features(s);delta=np.asarray(dz@coef);old=z.argmax(1)[gid];new=(z+delta[keys[:,1]]).argmax(1)[gid]
    contributions=np.zeros((len(s),4,3));coarse=[];terms=[]
    h=FeatureHasher(n_features=128,input_type='dict',alternate_sign=False)
    for o,ss in enumerate(s):
        a=json.loads(ss);known={k:v for k,v in a['facts'].items() if a['states'].get(k)=='known'}
        coarse.append(json.dumps({k:v for k,v in known.items() if k!='src_port_fixed'},sort_keys=True))
        tt=sorted(tokens(ss));terms.append(tt)
        if not tt:continue
        unit=h.transform([{t:1.} for t in tt]);bins=np.array([int(unit.indices[unit.indptr[j]])+(0 if t.startswith('asa:') else 128) for j,t in enumerate(tt)])
        norm=np.sqrt(np.square(np.bincount(bins,minlength=256)).sum())
        for t,b in zip(tt,bins):
            c=0 if 'src_port_fixed=' in t else 1 if 'dst_port_fixed=' in t else 2 if t.startswith('asa:') else 3
            contributions[o,c]+=coef[b]/norm
    assert np.allclose(contributions.sum(1),delta,atol=1e-11,rtol=1e-12)
    ccode,ctext=pd.factorize(np.asarray(coarse),sort=True)
    ccounts=np.bincount(ccode[obs[fit]]*3+y[fit],minlength=len(ctext)*3).reshape(-1,3)
    no_src=(z+(delta-contributions[:,0])[keys[:,1]]).argmax(1)[gid]
    nf=inner&(y==1)&(old==y)&(new!=y);prot=fit&(y==1)&(old==y)
    summaries=[]
    for c in np.unique(ccode[obs[nf]]):
        err=nf&(ccode[obs]==c);train=prot&(ccode[obs]==c);ii=np.flatnonzero(err);shift=contributions[obs[ii],:,2]-contributions[obs[ii],:,1]
        summaries.append({'coarse_behavior':json.loads(ctext[c]),'M_regression_rows':int(err.sum()),'training_class_counts':ccounts[c].tolist(),
            'fit_protected_M_rows':int(train.sum()),'fit_protected_M_that_need_source_terms_to_stay_M':int((train&(new==1)&(no_src!=1)).sum()),
            'M_regressions_remaining_without_source_terms':int((err&(no_src!=1)).sum()),
            'mean_S_minus_M_shift_by_part':dict(zip(['source_port','destination_port','common_ASA','windows'],shift.mean(0).tolist()))})
    selected=np.load(PRIOR/'fold1/selected_rows.npy');loss=[]
    # Original-frequency one-versus-rest BCE used by v89, plus direct softmax CE for diagnosis only.
    for name,d in [('teacher',0.*delta),('F00',delta)]:
        score=z[gid[selected]]+d[obs[selected]];truth=y[selected]
        bce=np.logaddexp(0,score).sum(1)-score[np.arange(len(truth)),truth]
        maximum=score.max(1);ce=np.log(np.exp(score-maximum[:,None]).sum(1))+maximum-score[np.arange(len(truth)),truth]
        loss.append({'model':name,'main_loss_rows':len(selected),'OVR_BCE':float(bce.mean()),'softmax_CE':float(ce.mean()),
            'classwise_BCE':{str(k):float(bce[truth==k].mean()) for k in range(3)},'L2_term_at_alpha_1e_6':float(1e-6/2*(theta[:768]@theta[:768])) if name=='F00' else 0.})
    ei=m['E'];et=m['truth'];ee=z[ei]
    # Exclude the true-class column from competitor margin.
    gaps=.01+ee-ee[np.arange(len(ei)),et,None];gaps[np.arange(len(ei)),et]=-np.inf;slack0=np.maximum(0,gaps.max(1))
    result={'source_sha256':sha(__file__),'actual_new_fits':0,'original_rows':len(y),
        'decomposition_identity_max_error':float(np.abs(contributions.sum(1)-delta).max()),
        'inner_M_regressions':int(nf.sum()),'inner_M_regressions_remaining_without_source_terms':int((nf&(no_src!=1)).sum()),
        'all_fit_protected_M_rows':int(prot.sum()),'all_fit_protected_M_that_need_source_terms_to_stay_M':int((prot&(new==1)&(no_src!=1)).sum()),
        'coarse_groups':sorted(summaries,key=lambda a:-a['M_regression_rows']),'main_risk':loss,
        'zero_correction_weighted_minimum_E_slack':float(m['weights']@slack0),'locked_weighted_E_slack_cap':float(m['slack_cap']),
        'scope':'Exact fixed-score algebra using original normalization. Removing source terms is a diagnostic, not a new model or recommended feature deletion. Bucket-shared coefficients do not have exclusive semantic attribution.',
        'seconds':time.monotonic()-start}
    save(DEST/'score_decomposition.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
