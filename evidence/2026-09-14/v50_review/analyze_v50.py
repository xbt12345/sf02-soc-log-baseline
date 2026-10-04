"""Post-fit diagnostics only. Never used for fit, selection or relabeling."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

def main(root,out):
    sys.path.insert(0,str(out/'runtime'))
    from run_v48 import save,sha
    from run_v49 import data,metrics
    import v50_views
    rows,ids,texts,facts=data(root);fit=rows.inner_role.to_numpy()!=2;ev=~fit
    old=joblib.load(root/'artifacts/v48_information_repair_r2_20260914/pressure/model.joblib')
    model=joblib.load(out/'pressure/model.joblib');x=sparse.hstack([old['text_encoder'].transform(texts),old['fact_encoder'].transform(facts)],format='csr')
    x.eliminate_zeros();x.sort_indices()
    keys=[]
    for k in range(x.shape[0]):
        a,b=x.indptr[k:k+2];keys.append(hashlib.sha256(x.indices[a:b].astype('<i8').tobytes()+x.data[a:b].astype('<f8').tobytes()).hexdigest())
    keys=np.asarray(keys);d=rows.copy();d['encoded_key']=keys[ids]
    train=d[fit & (d.route=='asa')]
    counts=train.groupby(['encoded_key','label_index']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
    evaluation=pd.read_parquet(out/'pressure/evaluation.parquet');assert np.array_equal(evaluation.row_position,rows.loc[ev,'row_position'])
    index=np.flatnonzero(ev & (d.route=='asa'));asa=d.iloc[index].copy();probs=model['model'].predict_proba(x)[ids[index]];asa['prediction']=probs.argmax(1)
    c=counts.reindex(asa.encoded_key,fill_value=0).to_numpy();labels=asa.label_index.to_numpy()
    bucket=np.where(c.sum(1)==0,'unseen_encoded_input',
        np.where(c[np.arange(len(c)),labels]==0,'seen_but_true_class_absent_in_fit',
        np.where((c>0).sum(1)>1,'seen_mixed_fit_labels','seen_pure_same_label')))
    asa['bucket']=bucket;asa['error']=asa.prediction!=asa.label_index
    for i,name in enumerate(['fit_B','fit_M','fit_S']):asa[name]=c[:,i]
    stats=[]
    for k,g in asa.groupby('bucket'):
        stats.append({'bucket':k,'rows':len(g),'errors':int(g.error.sum()),'body_groups':int(g.body_group.nunique()),'encoded_inputs':int(g.encoded_key.nunique())})
    finite=asa.groupby(['encoded_key','label_index']).size().unstack(fill_value=0)
    floor=int((finite.sum(axis=1)-finite.max(axis=1)).sum())
    dominant=[]
    for key,n in asa[asa.error].encoded_key.value_counts().head(2).items():
        g=asa[asa.encoded_key==key];pid=int(g.iloc[0].projection_id)
        ff=json.loads(pd.read_parquet(root/'artifacts/v48_information_repair_r2_20260914/projections.parquet').iloc[pid].facts)
        relevant=[]
        for field in ['icmp_code','icmp_unreachable']:
            value=ff.get(field);matches=np.asarray([f.get(field)==value for f in facts])[ids]&fit
            relevant.append({'field':field,'value':value,'all_route_fit_rows':int(matches.sum())})
        dominant.append({'rows':len(g),'errors':int(n),'body_groups':int(g.body_group.nunique()),'facts':ff,
            'labels_B_M_S':[int((g.label_index==i).sum()) for i in range(3)],'individual_feature_support':relevant})
    auth=[]
    for k in np.unique(ids[rows.route.to_numpy()=='authentication']):
        ps=model['model'].predict_proba(x[k])[0];pb=old['model'].predict_proba(x[k])[0];mask=(ids==k)&ev
        auth.append({'text':texts[k],'facts':facts[k],'rows':int(mask.sum()),'before':pb.tolist(),'after':ps.tolist()})
    common=[i for i in np.flatnonzero(fit) if facts[ids[i]].get('credential_check')=='invalid' and facts[ids[i]].get('auth_result')=='failure']
    partial=[v50_views.partial_auth(facts[ids[i]]) for i in common]
    xp=sparse.hstack([old['text_encoder'].transform(['']*len(partial)),old['fact_encoder'].transform(partial)],format='csr')
    newp=model['model'].predict_proba(xp);oldp=old['model'].predict_proba(xp)
    negative=rows.route.to_numpy()=='asa_acl';q=model['model'].predict_proba(x)[ids[negative]];b=old['model'].predict_proba(x)[ids[negative]]
    yy=rows.loc[negative,'label_index'].to_numpy()
    evidence=root/'evidence/2026-09-14/v50_review';evidence.mkdir(parents=True,exist_ok=True)
    asa.to_parquet(evidence/'asa_fit_support_rows.parquet',index=False)
    group=evaluation.copy();group['fix']=(group[['b_benign','b_malicious','b_suspicious']].to_numpy().argmax(1)!=group.label_index)&(group[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1)==group.label_index)
    fixes=group[group.fix].groupby(['route','body_group']).size()
    save(evidence/'summary.json',{'pressure':metrics(evaluation),'authentication':auth,
        'authentication_fit_partial_probe':{'rows':len(common),'before_errors':int((oldp.argmax(1)!=rows.iloc[common].label_index).sum()),'after_errors':int((newp.argmax(1)!=rows.iloc[common].label_index).sum())},
        'gain_body_groups':[dict(route=r,body_group=int(g),fixed_rows=int(n)) for (r,g),n in fixes.items()],
        'ASA_fit_support':stats,'ASA_finite_evaluation_error_floor':floor,'dominant_ASA_inputs':dominant,
        'asa_acl_normal_counterexamples':{'rows':int(negative.sum()),'before_errors':int((b.argmax(1)!=yy).sum()),'after_errors':int((q.argmax(1)!=yy).sum()),'fit_rows':int((negative&fit).sum())},
        'interpretation':'Inspected development diagnostic. Fit support is not new information or label truth. Same encoded-input bounds are empirical and representation-specific.',
        'analysis_source_sha256':sha(Path(__file__))})
    # Regression attribution: same input vectors, different shared coefficients.
    if (out/'execution.json').exists():
        regressions=[];comparisons=[];paired=[];public=[]
        for fold in range(3):
            candidate=joblib.load(out/('fold_'+str(fold))/'model.joblib');control=joblib.load(out/('control_fold_'+str(fold))/'model.joblib')
            te=control['text_encoder'];fe=control['fact_encoder'];names=np.r_[te.names(),fe.names()]
            assert np.array_equal(te.names(),candidate['text_encoder'].names())
            assert np.array_equal(te.idf,candidate['text_encoder'].idf)
            assert np.array_equal(fe.transform(facts).data,candidate['fact_encoder'].transform(facts).data)
            e=pd.read_parquet(out/('fold_'+str(fold))/'evaluation.parquet');paired.append(e)
            older=pd.read_parquet(root/('artifacts/v39_local_r2_20260913/primary/fold_%s/SEMANTIC'%fold)/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
            public_e=e.copy()
            for name in ['benign','malicious','suspicious']:public_e['b_'+name]=older['p_'+name].to_numpy()
            public.append(public_e)
            pp=e[['p_benign','p_malicious','p_suspicious']].to_numpy();bp=e[['b_benign','b_malicious','b_suspicious']].to_numpy();yy=e.label_index.to_numpy()
            bad=(pp.argmax(1)!=yy)&(bp.argmax(1)==yy)
            byposition=rows.set_index('row_position');active=np.unique(rows.projection_id)
            delta=(candidate['model'].coef_[2]-candidate['model'].coef_[1])-(control['model'].coef_[2]-control['model'].coef_[1])
            intercept=float((candidate['model'].intercept_[2]-candidate['model'].intercept_[1])-(control['model'].intercept_[2]-control['model'].intercept_[1]))
            for j in np.flatnonzero(bad):
                pos=int(e.iloc[j].row_position);k=int(np.searchsorted(active,byposition.loc[pos,'projection_id']))
                xx=sparse.hstack([te.transform([texts[k]]),fe.transform([facts[k]])],format='csr')
                values=xx.toarray().ravel()*delta
                actual=float(np.log(pp[j,2]/pp[j,1])-np.log(bp[j,2]/bp[j,1]));error=abs(actual-values.sum()-intercept);assert error<1e-10
                top=np.argsort(-abs(values))[:6]
                regressions.append({'fold':fold,'row_position':pos,'body_group':int(e.iloc[j].body_group),'route':str(e.iloc[j].route),
                    'official_label':int(yy[j]),'facts':facts[k],'before':bp[j].tolist(),'after':pp[j].tolist(),
                    'S_minus_M_margin_change':actual,'intercept_delta':intercept,'decomposition_error':error,
                    'largest_feature_changes':[{'feature':str(names[t]),'contribution':float(values[t])} for t in top if values[t]]})
            comparisons.append({'fold':fold,**metrics(e)})
        save(evidence/'primary_analysis.json',{'folds':comparisons,'paired_v48':metrics(pd.concat(paired,ignore_index=True)),
            'versus_public_B':metrics(pd.concat(public,ignore_index=True)),'regressions':regressions,
            'source': 'Input encoders unchanged within each pair; all regression margins decomposed exactly.'})
    shutil_source=Path(__file__);(evidence/'analyze_v50.py').write_bytes(shutil_source.read_bytes())
    print(json.dumps({'ASA':stats,'ASA_floor':floor,'auth_partial_before_errors':int((oldp.argmax(1)!=rows.iloc[common].label_index).sum()),'auth_partial_after_errors':int((newp.argmax(1)!=rows.iloc[common].label_index).sum())}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.out.resolve())
