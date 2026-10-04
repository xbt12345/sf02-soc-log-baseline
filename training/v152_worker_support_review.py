"""Independently reconstruct conditional support and original-mass concentration.

No fitted rules, classifier calls, gradients, pseudo-labels or class decisions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import read,sha,check_bindings

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/v152_independent_leave_root_conditions_20261001'
OUT=ROOT/'artifacts/v152_worker_support_review_20261001'
TRACE=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
FIELDS=('action','outcome','transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed',
        'src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable')
UNKNOWN={'src_port_fixed':65536,'dst_port_fixed':65536,'icmp_type':256,'icmp_code':256}

def key(f,omit):
    states=[]
    for field in FIELDS:
        value=f.get(field)
        known=value is not None and not(field in UNKNOWN and value==UNKNOWN[field])
        states.append([field,known,value if known and not(omit and field=='src_port_fixed') else None])
    return json.dumps(states,ensure_ascii=False,separators=(',',':'))

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists(),'Preserve executed evidence'
    parent=read(PARENT/'audit.json');check_bindings(parent['source_sha256'])
    trace=pd.read_parquet(TRACE).sort_values('row_position').reset_index(drop=True)
    q=pd.read_parquet(PARENT/'all_legal_train_leave_root_support.parquet')
    assert len(trace)==112807 and len(q)==451228
    facts=trace.facts_json.map(json.loads)
    scopes=['complete_facts_mask_key','omit_source_port_key']
    for scope in scopes:trace[scope]=facts.map(lambda f:key(f,scope==scopes[1]))
    t=trace.set_index('row_position')
    assert np.array_equal(q.truth,t.loc[q.row_position,'truth'])
    assert np.array_equal(q.root,t.loc[q.row_position,'root'])
    assert np.array_equal(q.fold,t.loc[q.row_position,'fold'])
    assert q.fold.ne(q.outer_training_role).all()
    assert q.groupby(['key_scope','row_position']).size().eq(2).all()
    outputs=[];metric=[]
    for role in range(3):
        legal=trace[trace.fold.ne(role)]
        assert not set(legal.root)&set(trace[trace.fold.eq(role)].root)
        for scope in scopes:
            z=q[q.outer_training_role.eq(role)&q.key_scope.eq(scope)].copy()
            assert set(z.row_position)==set(legal.row_position)
            assert np.array_equal(z.condition_key,t.loc[z.row_position,scope])
            for cls in [1,2]:
                # Whole original-mass root incidence, independent of parent's
                # global-count subtraction and explicit per-query checks.
                cells=legal[legal.truth.eq(cls)].groupby([scope,'root']).size().rename('mass').reset_index()
                groups=cells.groupby(scope).mass.agg(total='sum',roots='size')
                groups['squared_mass']=cells.assign(squared=cells.mass.astype(float)**2).groupby(scope).squared.sum()
                ordered=cells.sort_values([scope,'mass','root'],ascending=[True,False,True],kind='stable')
                ranks=ordered.groupby(scope).cumcount();top=ordered[ranks.eq(0)].set_index(scope)
                second=ordered[ranks.eq(1)].set_index(scope)
                idx=pd.MultiIndex.from_arrays([z.condition_key,z.root],names=[scope,'root'])
                own=cells.set_index([scope,'root']).mass.reindex(idx,fill_value=0).to_numpy(np.int64)
                g=groups.reindex(z.condition_key).fillna(0)
                remaining=g.total.to_numpy(np.int64)-own
                roots=g.roots.to_numpy(np.int64)-(own>0)
                assert np.array_equal(remaining,z[f'class{cls}_other_root_rows'])
                assert np.array_equal(roots,z[f'class{cls}_other_root_count'])
                maxima=top.mass.reindex(z.condition_key,fill_value=0).to_numpy(np.int64)
                firstroot=top.root.reindex(z.condition_key,fill_value=-1).to_numpy(np.int64)
                alternate=second.mass.reindex(z.condition_key,fill_value=0).to_numpy(np.int64)
                maxima=np.where(firstroot==z.root.to_numpy(),alternate,maxima)
                sq=g.squared_mass.to_numpy(float)-own.astype(float)**2
                share=np.divide(maxima,remaining,out=np.full(len(z),np.nan),where=remaining>0)
                hhi=np.divide(sq,remaining.astype(float)**2,out=np.full(len(z),np.nan),where=remaining>0)
                z[f'class{cls}_max_other_root_mass_share']=share
                z[f'class{cls}_other_root_original_mass_HHI']=hhi
                z[f'class{cls}_effective_other_roots']=np.divide(1,hhi,out=np.full(len(z),np.nan),where=hhi>0)
                assert np.all((remaining==0)==np.isnan(hhi))
                assert np.all(hhi[remaining>0]>0) and np.all(hhi[remaining>0]<=1)
            same=z.truth.eq(1).to_numpy()
            for suffix in ['max_other_root_mass_share','other_root_original_mass_HHI','effective_other_roots']:
                z['same_class_'+suffix]=np.where(same,z['class1_'+suffix],z['class2_'+suffix])
            for (cls,state),g in z.groupby(['truth','support_state']):
                s=g.same_class_max_other_root_mass_share.dropna()
                e=g.same_class_effective_other_roots.dropna()
                metric.append(dict(role=role,key_scope=scope,truth=int(cls),support_state=state,original_role_rows=len(g),
                    supported_role_rows=len(s),max_root_share_min=None if s.empty else float(s.min()),
                    max_root_share_median=None if s.empty else float(s.median()),max_root_share_max=None if s.empty else float(s.max()),
                    effective_root_min=None if e.empty else float(e.min()),effective_root_median=None if e.empty else float(e.median()),
                    effective_root_max=None if e.empty else float(e.max())))
            outputs.append(z)
    allq=pd.concat(outputs,ignore_index=True)
    u=allq.groupby(['key_scope','row_position']).agg(roles=('row_position','size'),truth=('truth','first'),
        min_same=('same_class_other_root_rows','min'),max_same=('same_class_other_root_rows','max'),
        min_roots=('same_class_other_root_count','min'),max_opposite=('opposite_class_other_root_rows','max'),
        historic_outer_correct=('actual_V146_B_correct','first'),known_578=('known_578_cohort','first')).reset_index()
    assert u.roles.eq(2).all() and len(u)==225614
    u['multi_same_no_opposite_both_roles']=u.min_roots.ge(2)&u.max_opposite.eq(0)
    u['no_same_support_either_role']=u.max_same.eq(0)
    summaries=[]
    for (scope,cls),g in u.groupby(['key_scope','truth']):
        absent=g[g.no_same_support_either_role]
        summaries.append(dict(key_scope=scope,truth=int(cls),unique_original_rows=len(g),
            multi_same_no_opposite_both_roles=int(g.multi_same_no_opposite_both_roles.sum()),
            no_same_support_either_role=len(absent),historic_outer_correct_without_same_support=int(absent.historic_outer_correct.sum()),
            historic_prediction_scope='Historical outer model only, not the corresponding legal TRAIN role prediction.'))
    context=pd.read_parquet(ROOT/'artifacts/v151_context_coherence_20261001/all_ASA_context_ledger.parquet').set_index('row_position')
    current=pd.read_parquet(ROOT/'artifacts/v151_current_context_view_20261001/all_ASA_current_ordered_view_ledger.parquet').set_index('row_position')
    examples=[]
    for case in read(PARENT/'legal_train_original_counterexamples.json'):
        m,s=case['examples'];a=t.loc[m['row_position']];b=t.loc[s['row_position']]
        assert a.root!=b.root and a.fold!=case['outer_training_role'] and b.fold!=case['outer_training_role']
        assert a.truth==1 and b.truth==2
        assert a.complete_facts_mask_key==b.complete_facts_mask_key==case['condition_key']
        for row in [m,s]:assert row['raw_message']==t.loc[row['row_position'],'raw_message']
        examples.append(dict(outer_training_role=case['outer_training_role'],M_row=int(a.row_position),S_row=int(b.row_position),
            parsed_facts_and_observed_states_equal=True,raw_strings_equal=a.raw_message==b.raw_message,
            current_ordered_text_equal=current.loc[a.row_position,'current_V124_normalized_ordered_view_key']==current.loc[b.row_position,'current_V124_normalized_ordered_view_key'],
            M_product_state=context.loc[a.row_position,'product_name_state'],S_product_state=context.loc[b.row_position,'product_name_state'],
            interpretation='Legal TRAIN examples show equal parsed facts but different raw/context. This is not full input/threat identity, label noise proof, or a context rule.'))
    OUT.mkdir();allq.to_parquet(OUT/'all_query_root_mass_concentration.parquet',index=False)
    u.to_parquet(OUT/'two_role_unique_original_support.parquet',index=False)
    pd.DataFrame(metric).to_parquet(OUT/'all_class_state_concentration_metrics.parquet',index=False)
    save(OUT/'original_counterexample_scope_review.json',examples)
    evidence=[Path(__file__),PARENT/'audit.json',PARENT/'all_legal_train_leave_root_support.parquet',
        PARENT/'legal_train_original_counterexamples.json',TRACE,
        ROOT/'artifacts/v151_context_coherence_20261001/all_ASA_context_ledger.parquet',
        ROOT/'artifacts/v151_current_context_view_20261001/all_ASA_current_ordered_view_ledger.parquet']
    result=dict(status='independent_support_identity_and_original_mass_concentration_executed',latest_actual_classifier='V146',
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,issue_solved=False,
        parent_query_rows_verified=451228,unique_original_ASA_rows=112807,two_legal_roles_not_independent_samples=True,
        both_class_counts_and_roots_match_parent=True,no_same_support_not_error_cause=True,original_support_summaries=summaries,
        legal_counterexamples_verified=len(examples),concentration_metrics=metric,
        limits=['No fitted predictor, class lookup, pseudo-label, weighting or model selection is generated.',
            'Class-specific original-mass concentration is descriptive, not an equal-root sampler or threat criterion.',
            'Full-key support absence does not exclude compositional generalization; root is an isolation proxy, not verified entity.',
            'Historical outer correctness is only a descriptive control and cannot certify TRAIN-role classifier retention.',
            'Retained facts/masks are not whole input; all unknown/ICMP/mixed/correct rows remain.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence})
    save(OUT/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['concentration_metrics','source_sha256','limits']},ensure_ascii=False))

if __name__=='__main__':main()
