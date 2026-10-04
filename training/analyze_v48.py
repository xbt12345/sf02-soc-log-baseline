"""Post-fit paired group sensitivity and regression attribution; zero fits."""
import argparse
import collections
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

def main(root,out):
    sys.path.insert(0,str(out/'runtime'))
    import v48_input as v
    from run_v48 import raw_rows,save,sha,read
    if (out/'paired_analysis.json').exists():raise FileExistsError('Existing analysis')
    d=pd.read_parquet(out/'pressure/evaluation.parquet');rows=pd.read_parquet(out/'rows.parquet')
    y=d.label_index.to_numpy();a=d[['p_benign','p_malicious','p_suspicious']].to_numpy();b=d[['b_benign','b_malicious','b_suspicious']].to_numpy()
    old=b.argmax(1);new=a.argmax(1);d['old_wrong']=old!=y;d['new_wrong']=new!=y
    d['fix']=d.old_wrong&~d.new_wrong;d['regression']=~d.old_wrong&d.new_wrong
    groups=d.groupby(['route','body_group']).agg(rows=('row_position','size'),before_errors=('old_wrong','sum'),after_errors=('new_wrong','sum'),fixes=('fix','sum'),regressions=('regression','sum')).reset_index()
    groups['net']=groups.before_errors-groups.after_errors
    groups.to_parquet(out/'paired_body_groups.parquet',index=False)
    best=groups.sort_values('net',ascending=False).iloc[0]
    sensitive=d[~((d.route==best.route)&(d.body_group==best.body_group))]
    def burden(z):
        return {'rows':len(z),'before_errors':int(z.old_wrong.sum()),'after_errors':int(z.new_wrong.sum()),'net_reduction':int(z.old_wrong.sum()-z.new_wrong.sum())}
    negative={}
    for name in ['asa_acl','authentication','native_flow']:
        z=rows[rows.route==name]
        negative[name]={'allowed_rows':len(z),'labels_B_M_S':[int((z.label_index==k).sum()) for k in range(3)],
                        'fit_rows':int((z.inner_role!=2).sum()),'pressure_rows':int((z.inner_role==2).sum())}
    # The real normal-denial ASA ACL rows are not in this pressure evaluation.
    # Score the current model on them without claiming holdout validation.
    acl=rows[rows.route=='asa_acl'];aclraw=list(raw_rows(root/'data/official/train.parquet',acl.row_position))
    candidate=joblib.load(out/'pressure/model.joblib');prior=joblib.load(root/'artifacts/v39_local_r2_20260913/old_protocol_stress/model.joblib')
    if aclraw:
        q,_=v.classify_records(candidate,[{'message_sanitized':raw} for pos,raw in aclraw])
        labels=acl.set_index('row_position').loc[[pos for pos,raw in aclraw],'label_index'].to_numpy()
        negative['asa_acl']['current_fit_side_predictions_B_M_S']=[int((q.argmax(1)==k).sum()) for k in range(3)]
        negative['asa_acl']['current_fit_side_errors']=int((q.argmax(1)!=labels).sum())
        negative['asa_acl']['is_heldout_quality_evidence']=False
    reg=d[d.regression].copy();regression_details=[]
    tc=candidate['text_encoder'];tb=prior['text_encoder'];fc=candidate['fact_encoder'];fb=prior['fact_encoder']
    same_names=np.array_equal(tc.names(),tb.names()) and np.array_equal(fc.names(),fb.names())
    assert same_names
    assert np.array_equal(tc.idf,tb.idf)
    assert np.array_equal(fc.base.scale,fb.base.scale)
    names=np.concatenate([tc.names(),fc.names()])
    for pos,raw in raw_rows(root/'data/official/train.parquet',reg.row_position):
        r=v.prepare_message(raw);previous=v.old.prepare_message(raw)
        assert (r['text'],r['facts'])==(previous['text'],previous['facts'])
        x=sparse.hstack([tc.transform([r['text']]),fc.transform([r['facts']])],format='csr')
        z=sparse.hstack([tb.transform([previous['text']]),fb.transform([previous['facts']])],format='csr')
        assert (x!=z).nnz==0
        j=d.index[d.row_position==pos][0]
        # Exact additive decomposition of the suspicious-versus-benign margin.
        delta=(candidate['model'].coef_[2]-candidate['model'].coef_[0])-(prior['model'].coef_[2]-prior['model'].coef_[0])
        values=x.toarray().ravel()*delta
        intercept=float((candidate['model'].intercept_[2]-candidate['model'].intercept_[0])-(prior['model'].intercept_[2]-prior['model'].intercept_[0]))
        margin_before=float(np.log(b[j,2]/b[j,0]));margin_after=float(np.log(a[j,2]/a[j,0]))
        assert abs(float(values.sum())+intercept-(margin_after-margin_before))<1e-10
        top=np.argsort(-np.abs(values))[:8]
        regression_details.append({'row_position':pos,'body_group':int(d.loc[j,'body_group']),'route':str(d.loc[j,'route']),
            'label_index':int(y[j]),'raw':raw,'model_text':r['text'],'facts':r['facts'],
            'before_B_M_S':b[j].tolist(),'after_B_M_S':a[j].tolist(),'input_and_encoded_vector_identical':True,
            'fit_same_route_rows':int(((rows.route==r['route'])&(rows.inner_role!=2)).sum()),
            'S_versus_B_margin_before':margin_before,'S_versus_B_margin_after':margin_after,
            'intercept_delta':intercept,'sum_feature_deltas':float(values.sum()),
            'top_margin_changes':[{'feature':str(names[k]),'delta':float(values[k])} for k in top if values[k]]})
    save(out/'regression_details.json',regression_details)
    # Unchanged facts do not establish irreducible label noise. These are finite
    # same-input collision counts on inspected evaluation, not population limits.
    p=pd.read_parquet(out/'projections.parquet');active=np.unique(d.projection_id)
    keys={int(i):v.old.canonical([r.text,json.loads(r.facts)]) for i,r in p.loc[active].iterrows()}
    cells=collections.defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for row in d.itertuples():cells[(row.route,keys[int(row.projection_id)])][row.label_index]+=1
    lower=collections.Counter()
    for (route,key),counts in cells.items():lower[route]+=int(counts.sum()-counts.max())
    save(out/'paired_analysis.json',{'pressure':burden(d),'body_groups_improved':int((groups.net>0).sum()),
        'body_groups_worsened':int((groups.net<0).sum()),'largest_net_improvement_group':best.to_dict(),
        'excluding_largest_improvement_group':burden(sensitive),'negative_and_fit_support':negative,
        'regression_rows':len(regression_details),'text_vocabulary_idf_and_fact_scales_unchanged':True,
        'finite_evaluation_same_input_error_floor_by_route':dict(lower),
        'scope':'Seen-development diagnostic; same-input collision lower bound is representation-specific, not irreducible ground truth.',
        'analysis_code_sha256':sha(Path(__file__))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())
