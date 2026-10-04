"""Versioned correction: V151 v1 used old partial header handling, kept intact."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from experiment_review import read,sha,check_bindings
from v75_views import BYTE_FEATURES,byte_matrix,view
from v124_header import transform,old_text

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v151_context_coherence_20261001'
OUT=ROOT/'artifacts/v151_current_context_view_20261001'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def counts(d,key):
    t=d.groupby([key,'truth']).size().unstack(fill_value=0).reindex(columns=[1,2],fill_value=0)
    mixed=t[(t>0).sum(1)>1]
    return dict(original_rows=len(d),unique_keys=len(t),mixed_keys=len(mixed),mixed_original_rows=int(mixed.sum().sum()),
        deterministic_observed_projection_floor=int((t.sum(1)-t.max(1)).sum()))

def main():
    assert not OUT.exists();old=read(BASE/'audit.json');check_bindings(old['source_sha256'])
    tracepath=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    trace=pd.read_parquet(tracepath,columns=['row_position','local','root','fold','truth','raw_message'])
    rows=pd.read_parquet(BASE/'all_ASA_context_ledger.parquet');assert np.array_equal(trace.row_position,rows.row_position)
    cache={};texts=[];gaps=[];head=[];v1=[];prev=[]
    for raw in trace.raw_message:
        if raw not in cache:
            text,span,gap=transform(raw)
            assert raw[:span[1]]+raw[span[1]:]==raw
            body=raw[span[1]:]
            # This exact date/clock perturbation was accepted by V124's input contract.
            assert text==transform('<164>Jan 01 2000 01:02:03: '+body)[0]==transform('<164>Dec 31 2001 USER-CRED-45: '+body)[0]
            cache[raw]=(text,span[1],gap,digest(view(raw)[0]),digest(old_text(raw)))
        text,end,gap,a,b=cache[raw];texts.append(text);head.append(end);gaps.append(gap);v1.append(a);prev.append(b)
    assert sum(gaps)==682 and np.array_equal(rows.ordered_identity_clock_removed_key,np.array(v1))
    corrected=trace[['row_position','local','root','fold','truth']].copy()
    corrected['v1_old_v75_partial_clock_view_key']=v1
    corrected['old_normalized_partial_clock_view_key']=prev
    corrected['current_V124_normalized_ordered_view_key']=[digest(t) for t in texts]
    corrected['current_facts_plus_ordered_view_key']=[digest(a+b) for a,b in zip(rows.facts_key,corrected.current_V124_normalized_ordered_view_key)]
    corrected['old_header_gap']=gaps;corrected['current_header_end']=head
    keys=['v1_old_v75_partial_clock_view_key','old_normalized_partial_clock_view_key','current_V124_normalized_ordered_view_key','current_facts_plus_ordered_view_key']
    projections=[dict(scope='whole_population',key=k,**counts(corrected,k)) for k in keys]
    for f in range(3):
        legal=corrected[corrected.fold.ne(f)]
        for k in keys:projections.append(dict(scope='legal_TRAIN',fold=f,key=k,**counts(legal,k)))
    # Reconstruct current text grams, every current local representative; fact
    # and record-port coordinates remain the actual unmodified cache.
    representatives=trace[['local']].assign(text=texts).drop_duplicates('local').sort_values('local')
    assert np.array_equal(representatives.local,np.arange(22546))
    actual=sparse.load_npz(ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz').tocsr()
    rebuilt=byte_matrix(representatives.text.tolist());delta=rebuilt-actual[:,:BYTE_FEATURES]
    gap=0. if not delta.nnz else float(np.max(np.abs(delta.data)))
    assert gap<1e-7
    text_counts=trace[['local']].assign(key=corrected.current_V124_normalized_ordered_view_key).groupby('local').key.nunique()
    OUT.mkdir();corrected.to_parquet(OUT/'all_ASA_current_ordered_view_ledger.parquet',index=False)
    result=dict(status='versioned_current_header_view_correction_no_model_or_fit',latest_actual_classifier='V146',
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,original_rows=len(trace),v1_preserved=True,
        v1_name_correction='V151 v1 ordered_identity_clock_removed_key actually is old_v75_partial_clock_view, not all-clocks-removed; old audit and keys preserved.',
        old_header_gap_original_rows=int(sum(gaps)),current_header_date_clock_invariance_all_rows=True,
        current_text_byte_input_replayed_local_entries=len(representatives),current_text_byte_input_max_abs_gap=gap,
        same_local_multiple_current_ordered_texts=int(text_counts.gt(1).sum()),
        current_ordered_text_or_semantic_equality_not_full_input_identity=True,projections=projections,
        quality_acceptance=False,issue_solved=False,next_classifier_training_registered=False,
        limits=['Current ordered residual matches the actual input text path, but unresolved ACL/interface naming/style is not a trusted threat teacher.',
            'Date-invariance fixtures verify this registered grammar only; no labels are transferred across changed ports or roles.',
            'Header removal includes facility/severity, so separate logging-level projection is a new contextual hypothesis, not existing hidden information.',
            'No historical input, source, model or V151-v1 artifact modified; no model/gradient/fitting executed.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'training/v124_header.py',ROOT/'training/v75_views.py',
            ROOT/'training/v75_corrective.py',ROOT/'training/v99_normalization_feasibility.py',tracepath,BASE/'audit.json',BASE/'all_ASA_context_ledger.parquet',
            ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz']})
    save(OUT/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','limits']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
