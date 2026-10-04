"""No-fit audit of representation ambiguity, component support and split limits."""
import collections,json,time
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST as V89,OLD,LAST,read,save,sha
from run_v75 import load_sparse
from v89_partial_expression import tokens
DEST=ROOT/'artifacts/v91_first_principles_20260928'

def summary(g,y,mask):
    cc=np.bincount(g[mask]*3+y[mask],minlength=(int(g.max())+1)*3).reshape(-1,3)
    used=cc.sum(1)>0;mixed=(cc>0).sum(1)>1
    return {'rows':int(mask.sum()),'observed_groups':int(used.sum()),'mixed_label_groups':int(mixed.sum()),
       'rows_in_mixed_groups':int(cc[mixed].sum()),'minimum_errors_of_any_deterministic_lookup_on_this_representation':int((cc.sum(1)-cc.max(1)).sum())},cc

def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'audit.json').exists();start=time.monotonic()
    r=pd.read_parquet(OUT/'rows.parquet');y=r.label_index.to_numpy();fit=~r.fold.isin([0,1,2]).to_numpy();inner=r.fold.eq(1).to_numpy();fid=np.load(LAST/'row_feature_id.npy')
    obs=np.load(V89/'row_fact_code.npy');ss=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json
    literal=np.asarray([json.dumps(sorted(tokens(s)),separators=(',',':')) for s in ss]);obskey,_=pd.factorize(literal,sort=True)
    joint=np.load(V89/'F00_row_group.npy');old=np.load(OLD/'teacher_scores.npy',mmap_mode='r').argmax(1)[fid];new=np.load(V89/'F00_all_prediction.npy')[joint]
    matrices={};floors=[]
    for name,g in [('R0_complete_current_encoding',fid),('literal_partial_facts_no_hash',obskey[obs]),('R0_plus_partial_facts',joint)]:
        for role,mask in [('fit',fit),('inner',inner),('fit_and_inner',fit|inner)]:
            out,cc=summary(g,y,mask);floors.append({'representation':name,'role':role,**out})
            if role=='fit':matrices[name]=cc
    # Exact observed semantic groups, excluding source port, using only known facts.
    coarse=[]
    for text in ss:
        a=json.loads(text);coarse.append(json.dumps({k:v for k,v in a['facts'].items() if a['states'].get(k)=='known' and k!='src_port_fixed'},sort_keys=True))
    coarseid,coarsenames=pd.factorize(np.asarray(coarse),sort=True);cg=coarseid[obs]
    t=pd.DataFrame({'coarse':cg[fit],'class':y[fit],'component':r.component.to_numpy()[fit]})
    groups=t.groupby(['coarse','class']).agg(rows=('component','size'),components=('component','nunique'))
    support={}
    for name,mask in [('inner_old_S_error',inner&(y==2)&(old!=y)),('inner_F00_M_regression',inner&(y==1)&(old==y)&(new!=y))]:
        records=[]
        for c in np.unique(cg[mask]):
            cc=[];nc=[]
            for cls in range(3):
                a=groups.loc[(c,cls)] if (c,cls) in groups.index else None
                cc.append(0 if a is None else int(a['rows']));nc.append(0 if a is None else int(a['components']))
            records.append({'known_behavior':json.loads(coarsenames[c]),'target_rows':int((mask&(cg==c)).sum()),'fit_class_rows':cc,'fit_class_components':nc})
        support[name]=records
    # Re-evaluate only the predefined eighty training S examples of the fresh support model.
    projections=pd.read_parquet(OUT/'projections.parquet',columns=['facts']);fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']
    fs=[json.loads(s) for s in projections.facts];keys=[tuple(str(f[k]) for k in fields) if all(k in f for k in fields) else None for f in fs]
    chosen={tuple(v) for v in read(V89/'support_design.json')['chosen_keys']};tr=np.load(V89/'support_full_rows.npy')
    target=np.array([i for i in tr if y[i]==2 and keys[int(r.projection_id.iloc[i])] in chosen])
    x=load_sparse(LAST/'X');m=joblib.load(V89/'support_full.joblib');p=np.asarray(x[fid[target]]@m['coef'])+m['intercept'];err=target[p.argmax(1)!=2]
    assert len(target)==80 and len(err)==52
    mask=np.zeros(len(y),bool);mask[tr]=True;error_views={}
    for name,g in [('R0_complete_current_encoding',fid),('literal_partial_facts_no_hash',obskey[obs]),('R0_plus_partial_facts',joint)]:
        _,cc=summary(g,y,mask);ec=cc[g[err]]
        error_views[name]={'S_error_rows':len(err),'distinct_inputs':len(np.unique(g[err])),
            'errors_with_other_class_same_input':int((ec[:,:2].sum(1)>0).sum()),'errors_where_S_is_strict_minority_against_M':int((ec[:,2]<ec[:,1]).sum())}
    np.save(DEST/'support_S_error_rows.npy',err)
    # Exact independent-component counts, rather than repeated row counts, limit role allocation.
    cell=[]
    for key in sorted(chosen):
        ii=target[np.array([keys[int(r.projection_id.iloc[i])]==key for i in target])]
        cell.append({'key':list(key),'S_rows':len(ii),'S_components':int(r.component.iloc[ii].nunique()),'S_complete_R0_inputs':len(np.unique(fid[ii])),
            'S_error_rows':int(np.isin(ii,err).sum()),'can_populate_three_disjoint_component_roles_with_this_class':bool(r.component.iloc[ii].nunique()>=3)})
    result={'source_sha256':sha(__file__),'new_classifier_fits':0,'new_calibration_fits':0,'rows':len(y),'representation_floors':floors,
        'support_diagnostics':support,'support_S_error_views':error_views,'fine_S_independent_support':cell,
        'scope':'Empirical deterministic lookup error lower bounds for specified representations, not Bayes error, raw semantic impossibility or deployable oracle. Component is a grouping proxy, not independently verified real entity. No predictions trained or tuned.',
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'rows.parquet',V89/'row_fact_code.npy',V89/'row_fact_dictionary.parquet',V89/'support_full.joblib',V89/'support_full_rows.npy',V89/'support_design.json']},'seconds':time.monotonic()-start}
    save(DEST/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['support_diagnostics','input_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
