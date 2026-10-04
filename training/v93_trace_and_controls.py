"""Adaptive follow-up: rule out new-input corruption and cheap acceptance shortcuts."""
import difflib,hashlib,json,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from v89_common import ROOT,OUT,read,save,sha,data,raw_counts
from v93_regression_audit import DEST,PREV
from v75_corrective import stable
from v75_views import byte_matrix
from v89_partial_facts import parse

def main():
    assert not (DEST/'trace_controls.json').exists()
    r,y,fid,z,old,_,fit=data();t=pd.read_parquet(DEST/'inner_500M_58S_trace.parquet');wanted=set(t.row_position);raws={};offset=0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(batch_size=32768,columns=['message_sanitized','label_binary'],use_threads=False):
        ii=np.array(sorted(i-offset for i in wanted if offset<=i<offset+batch.num_rows),int)
        if len(ii):
            sub=batch.take(ii).to_pandas()
            for j,text,label in zip(ii,sub.message_sanitized,sub.label_binary):
                i=int(offset+j);raws[i]=text;assert y[i]=={'benign':0,'malicious':1,'suspicious':2}[label]
        offset+=batch.num_rows
    assert len(raws)==558
    x=sparse.load_npz(PREV/'ASA_R0.npz');ids=np.load(PREV/'ASA_input_ids.npy');texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    facts=pd.read_parquet(OUT/'projections.parquet').facts
    total=raw_counts(fid,y,np.ones(len(r),bool),len(old));groups=[]
    for f,b in t.groupby('R0_id'):
        i=int(b.row_position.iloc[0]);current=stable(texts.loc[r.new_text_id.iloc[i]]);diff=x[np.searchsorted(ids,f),:65792]-byte_matrix([current]);assert diff.nnz==0 or abs(diff.data).max()<1e-7
        known=json.loads(b.known.iloc[0]);missing=[]
        for j in b.row_position:
            p=parse(raws[int(j)],'asa');k={a:v for a,v in p['facts'].items() if p['states'].get(a)=='known'};assert k==known
            oldfacts=json.loads(facts.iloc[r.projection_id.iloc[j]])
            if any(oldfacts.get(a)!=v for a,v in known.items()):missing.append(int(j))
        groups.append({'R0_id':int(f),'rows':b.row_position.tolist(),'kind':b.kind.iloc[0],'current_text':current,'known_facts':known,
            'missing_known_facts_rows':missing,'all_official_class_counts':total[f].tolist(),'same_X_other_label':bool((total[f]>0).sum()>1)})
    save(DEST/'all_94_input_reconstruction.json',groups)
    t['raw_sha256']=t.row_position.map(lambda i:hashlib.sha256(raws[int(i)].encode()).hexdigest())
    t[['row_position','R0_id','kind','label_index','raw_sha256']].to_parquet(DEST/'558_raw_identity_trace.parquet',index=False)
    gates={
        'new_uncalibrated_confidence_ge_99':t.new_softmax_max_uncalibrated>=.99,
        'nearest_raw_R0_prefers_S':t.nearest_R0_2>t.nearest_R0_1,
        'at_least_two_coarse_S_training_components':t.fit_coarse_2_components>=2,
        'at_least_one_exact_known_S_training_component':t.fit_known_2_components>=1,
        'coarse_S_training_present':t.fit_coarse_2_rows>0,
    }
    controls={name:{'retained_S_repairs':int((accept&t.kind.eq('repair')).sum()),'retained_M_regressions':int((accept&t.kind.eq('regression')).sum())} for name,accept in gates.items()}
    summary={'status':'completed','raw_original_labels_verified':len(t),'actual_R0_text_groups_verified':len(groups),
        'known_fact_omission_rows':sum(len(g['missing_known_facts_rows']) for g in groups),
        'all_official_exact_input_conflict_rows_in_558':sum(len(g['rows']) for g in groups if g['same_X_other_label']),
        'fixed_acceptance_control_on_558_changes':controls,
        'scope':'Adaptive diagnostic controls on inspected inner changes only; not full evaluation, trained gate, generalization proof or promotion.',
        'new_classifier_fits':0,'new_calibration_fits':0,'source_sha256':sha(__file__)}
    save(DEST/'trace_controls.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
