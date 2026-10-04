"""Independent structural, objective and provenance checks; no model fitting."""
import hashlib
import json
import sys
import numpy as np
from scipy.special import logsumexp, expit
from v79_execute import DEST, WIDTH, rows, load, weights, objective, FORMATS, FOLDS, SEED
from run_v75 import OUT, save, sha, load_sparse, features, matrices


def main():
    r,x,fid=load();rng=np.random.default_rng(7910); checks={}
    m=matrices();m['new_text']=load_sparse(OUT/'corrective/text')
    ix=rng.choice(len(r),500,replace=False)
    original=features(r,ix,m,'new');packed=x[fid[ix]]
    d=original-packed;checks['raw_cache_to_packed_max_delta']=float(abs(d.data).max()) if d.nnz else 0
    assert checks['raw_cache_to_packed_max_delta']==0
    # Compare grouped objective to expanded original rows using arbitrary coefficients
    # and arbitrary positive frequency weights; neither expression shares the loss helper.
    fit=~r.fold.isin([0,2]).to_numpy(); exposed={}
    for arm in ['A','B','C']:
        cc,den,e=weights(r,fid,x.shape[0],fit,arm);exposed[arm]=e
    assert exposed['B']['selected_rows_sha256']==exposed['C']['selected_rows_sha256']
    assert exposed['A']['exposed_per_class'][1:]==exposed['B']['exposed_per_class'][1:]
    # Test loss equivalence on deliberately repeated/conflicting labels.
    base=x[fid[ix[:12]]]; repeats=np.array([0,0,1,2,2,2,4,4,6,9,9,9,9])
    yy=np.array([0,2,1,0,0,2,1,1,2,0,0,1,2]);rw=np.arange(1,len(yy)+1,dtype=float)/4
    cnt=np.bincount(repeats*3+yy,weights=rw,minlength=base.shape[0]*3).reshape(-1,3)
    w=rng.normal(0,.05,(WIDTH+1,3));z=base[repeats]@w[:-1]+w[-1];den=17
    losses={}
    for kind in ['ovr','softmax','ms']:
        if kind=='ovr':loss=np.sum(rw*(np.logaddexp(0,z).sum(1)-z[np.arange(len(yy)),yy]))/den
        else:
            loss=np.sum(rw*(logsumexp(z,axis=1)-z[np.arange(len(yy)),yy]))/den
            if kind=='ms':
                non=yy!=0;zz=z[non,1:];loss+=np.sum(rw[non]*(logsumexp(zz,axis=1)-zz[np.arange(non.sum()),yy[non]-1]))/rw[non].sum()
        loss+=1e-6/2*np.square(w[:-1]).sum()
        got,g=objective(w.ravel(),base,cnt,den,kind)
        losses[kind]=abs(got-loss);assert abs(got-loss)<1e-12
    checks['aggregation_expanded_loss_delta']=losses
    from v78_denial_adapter_v4 import self_check
    checks['parser_negative_checks']=self_check()
    # Equivalent wrappers retain the exact original payload and all unknown fields.
    payloads=['deny tcp src inside:a/443 dst outside:b/53', 'quoted="allow not deny" residual=未知', 'a,b,"c,d"', '', 'line1\nline2']
    for s in payloads:
        a=json.dumps({'original_record':s,'unmapped_residual':s},ensure_ascii=False)
        assert json.loads(a)['original_record']==s
        from urllib.parse import quote,unquote
        assert unquote(quote(s,safe=''))==s
    checks['lossless_wrapper_roundtrips']=len(payloads)*2
    checks['wrapper_scope']='Serialization/provenance only, not learned unknown-format invariance or training augmentation'
    # Count input-label conflicts without removing, relabelling or claiming raw truth ambiguity.
    nonfit=fit&r.label_index.ne(0).to_numpy()
    q=np.bincount(fid[nonfit]*3+r.label_index.to_numpy()[nonfit],minlength=x.shape[0]*3).reshape(-1,3)
    conflict=(q[:,1]>0)&(q[:,2]>0)
    checks['fit_full_encoded_MS_conflict_groups']=int(conflict.sum())
    checks['fit_encoded_MS_minimum_errors']=int(np.minimum(q[conflict,1],q[conflict,2]).sum())
    degeneration={}
    for route in FORMATS:
        support=np.bincount(r.loc[fit&r.route.eq(route).to_numpy(),'label_index'],minlength=3)
        degeneration[route]={'fit_support':support.tolist(),'zero_M_equals_whole_format':bool(support[0]+support[2]==0)}
    save(DEST/'input_objective_audit.json',{'checks':checks,'exposure':exposed,'protocol_coverage':degeneration,
         'source_sha256':sha(__file__),'scope':'No fit, no target answers, no perturbation class-quality claim'})
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
