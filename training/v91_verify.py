"""Original-label and independent grouping verification of the no-fit evidence audit."""
import json,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from v89_common import ROOT,OUT,DEST as V89,LAST,read,save,sha
from v89_partial_expression import tokens
from v75_corrective import stable
from v75_views import byte_matrix,BYTE_FEATURES
from run_v75 import load_sparse
DEST=ROOT/'artifacts/v91_first_principles_20260928'

def main():
    start=time.monotonic();assert not (DEST/'verification.json').exists();a=read(DEST/'audit.json');context=read(DEST/'current_input_context.json')
    assert a['source_sha256']==sha(ROOT/'training/v91_evidence_audit.py')
    for p,h in a['input_sha256'].items():assert sha(ROOT/p)==h,p
    r=pd.read_parquet(OUT/'rows.parquet');y=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.array_equal(y,r.label_index) and np.array_equal(r.row_position,np.arange(len(y)))
    fit=~r.fold.isin([0,1,2]).to_numpy();inner=r.fold.eq(1).to_numpy();fid=np.load(LAST/'row_feature_id.npy');joint=np.load(V89/'F00_row_group.npy');obs=np.load(V89/'row_fact_code.npy')
    ss=pd.read_parquet(V89/'row_fact_dictionary.parquet').observation_json;literal=np.asarray([json.dumps(sorted(tokens(s)),separators=(',',':')) for s in ss]);codes,_=pd.factorize(literal,sort=True)
    gg={'R0_complete_current_encoding':fid,'literal_partial_facts_no_hash':codes[obs],'R0_plus_partial_facts':joint};roles={'fit':fit,'inner':inner,'fit_and_inner':fit|inner}
    for record in a['representation_floors']:
        mask=roles[record['role']];frame=pd.DataFrame({'input':gg[record['representation']][mask],'y':y[mask]})
        counts=frame.groupby(['input','y']).size().unstack(fill_value=0);mass=counts.sum(axis=1);mixed=(counts>0).sum(axis=1)>1
        assert len(counts)==record['observed_groups'] and int(mixed.sum())==record['mixed_label_groups']
        assert int(mass[mixed].sum())==record['rows_in_mixed_groups'] and int((mass-counts.max(axis=1)).sum())==record['minimum_errors_of_any_deterministic_lookup_on_this_representation']
    coarse=[]
    for s in ss:
        t=json.loads(s);coarse.append(json.dumps({k:v for k,v in t['facts'].items() if t['states'].get(k)=='known' and k!='src_port_fixed'},sort_keys=True))
    coarse=np.asarray(coarse)[obs];rowscomp=r.component.to_numpy()
    for target,records in a['support_diagnostics'].items():
        for rec in records:
            cmask=coarse==json.dumps(rec['known_behavior'],sort_keys=True)
            for cls in range(3):
                ix=np.flatnonzero(fit&cmask&(y==cls));assert len(ix)==rec['fit_class_rows'][cls] and len(set(rowscomp[ix]))==rec['fit_class_components'][cls]
    tr=np.load(V89/'support_full_rows.npy');err=np.load(DEST/'support_S_error_rows.npy');assert len(err)==52 and np.all(y[err]==2) and np.isin(err,tr).all()
    x=load_sparse(LAST/'X');model=__import__('joblib').load(V89/'support_full.joblib')
    assert (np.asarray(x[fid[err]]@model['coef'])+model['intercept']).argmax(1).tolist()==[1]*52
    for name,g in gg.items():
        table=pd.DataFrame({'input':g[tr],'y':y[tr]}).groupby(['input','y']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
        counts=table.loc[g[err]].to_numpy();rec=a['support_S_error_views'][name]
        assert len(set(g[err]))==rec['distinct_inputs'] and int((counts[:,:2].sum(1)>0).sum())==rec['errors_with_other_class_same_input']
        assert int((counts[:,2]<counts[:,1]).sum())==rec['errors_where_S_is_strict_minority_against_M']
    projections=[json.loads(s) for s in pd.read_parquet(OUT/'projections.parquet').facts];fields=['action','outcome','transport_protocol','src_role','dst_role','dst_port_fixed']
    keys=[tuple(str(t[k]) for k in fields) if all(k in t for k in fields) else None for t in projections]
    for rec in a['fine_S_independent_support']:
        ix=np.array([i for i in tr if y[i]==2 and keys[int(r.projection_id.iloc[i])]==tuple(rec['key'])]);nc=len(set(rowscomp[ix]))
        assert len(ix)==rec['S_rows'] and nc==rec['S_components'] and (nc>=3)==rec['can_populate_three_disjoint_component_roles_with_this_class']
        assert len(set(fid[ix]))==rec['S_complete_R0_inputs'] and int(np.isin(ix,err).sum())==rec['S_error_rows']
    for name,g in [('R0',fid),('partial_facts_no_hash',codes[obs]),('R0_plus_facts',joint)]:
        mask=fit&r.route.eq('asa').to_numpy();table=pd.DataFrame({'input':g[mask],'y':y[mask]}).groupby(['input','y']).size().unstack(fill_value=0)
        rec=context['ASA_fit_representation_bounds'][name];assert int((table.sum(axis=1)-table.max(axis=1)).sum())==rec['minimum_deterministic_lookup_errors']
        assert int(((table>0).sum(axis=1)>1).sum())==rec['mixed_groups'] and int(mask.sum())==rec['rows']
    texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    for rec in context['examples']:
        i,j=rec['S_row'],rec['M_row'];assert i in err and j in tr and y[i]==2 and y[j]==1 and codes[obs[i]]==codes[obs[j]]
        assert fid[i]==rec['S_R0_id'] and fid[j]==rec['M_R0_id'] and fid[i]!=fid[j]
        for prefix,row in [('S',i),('M',j)]:
            text=texts.loc[r.new_text_id.iloc[row]];assert text==rec[prefix+'_residual_text']
            assert stable(text)==rec[prefix+'_current_R0_residual_text']
            diff=x[fid[row],:BYTE_FEATURES]-byte_matrix([stable(text)])
            assert diff.nnz==0 or np.abs(diff.data).max()<1e-7
    # Preserve all historical bound training models, sources and results.
    prior={};paths=['evidence/2026-09-27/v79_execution/delivery.json','artifacts/v80_attribution_20260927/review_receipt.json']
    for ver,key in [(81,'diagnosis'),(82,'capacity'),(83,'root_review'),(84,'preservation'),(85,'protection'),(86,'boundary_review'),(87,'solver_supervision'),(88,'root_review'),(89,'readout_support')]:paths.append(f'evidence/2026-09-27/v{ver}_{key}/delivery.json')
    paths.append('evidence/2026-09-28/v90_root_review/delivery.json')
    for p in paths:
        for path,h in read(ROOT/p)['artifact_sha256'].items():
            assert path not in prior or prior[path]==h,path
            prior[path]=h
    changed=[p for p,h in prior.items() if sha(ROOT/p)!=h];assert not changed,changed
    result={'status':'passed','source_sha256':sha(__file__),'new_fits':0,'original_labels_checked':len(y),'grouping_bounds_independently_recomputed':len(a['representation_floors']),
        'ASA_bounds_recomputed':3,'support_error_rows_verified':len(err),'representative_context_pairs_and_actual_R0_text_matrices_verified':len(context['examples']),
        'prior_bound_files_rehashed':len(prior),'prior_bound_files_changed':changed,'receipt_sha256':{p:sha(ROOT/p) for p in paths},
        'outputs_sha256':{p.name:sha(p) for p in DEST.iterdir() if p.is_file()},'seconds':time.monotonic()-start,
        'scope':'Independent pandas label grouping and support counts;52 saved error rows replayed;two representative context pairs checked against actual corrective byte matrices. Reuses v89 literal extractor verified in v90; not semantic sufficiency or new generalization.'}
    save(DEST/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['receipt_sha256','outputs_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
