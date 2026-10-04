"""CPU replay of saved complete logits, all facts and original roles; no model calls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from experiment_review import ROOT,read,sha,check_bindings
from v157_functional_logit_audit_v2 import OUT,require,save

def main():
    require();assert not (OUT/'v2_verification.json').exists()
    audit=read(OUT/'audit.json');check_bindings(audit['source_sha256'])
    d=pd.read_parquet(OUT/'original_reference.parquet')
    rows=pd.read_parquet(OUT/'all_original_role_function_summary.parquet')
    prior_path=ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet'
    prior=pd.read_parquet(prior_path,columns=['row_position','known_578_cohort','same_family_and_outer_fold_control_S','both_ports_observed','facts_json'])
    assert np.array_equal(prior.row_position,d.row_position)
    geometry_path=ROOT/'artifacts/v156_conditional_representation_neighborhood_20261001/all_original_role_neighbors.parquet'
    geometry=pd.read_parquet(geometry_path,columns=['training_role','space','row_position','pred_baseline','pred_V155_B'])
    partition=read(OUT/'field_partition.json');verdicts=[];summaries=[];field_summaries=[];witnesses=[];binding=[Path(__file__),prior_path,geometry_path,OUT/'audit.json',OUT/'all_original_role_function_summary.parquet']
    facts=prior.facts_json.map(json.loads)
    icmp=facts.map(lambda x:str(x.get('transport_protocol','')).upper()=='ICMP').to_numpy()
    assert int(icmp.sum())==2373
    # Also report all unknown/missing ports; ICMP N/A is retained in that population.
    for f in range(3):
        folder=OUT/f'fold{f}';names=['logits.npy','body.npy','facts.npy','bias.npy','probabilities.npy','field_logits.npz','local_function_summary.parquet']
        binding.extend(folder/n for n in names)
        z=np.load(folder/'logits.npy');b=np.load(folder/'body.npy');direct=np.load(folder/'facts.npy');bias=np.load(folder/'bias.npy');q=np.load(folder/'probabilities.npy')
        assert z.shape==b.shape==(22546,16,3) and direct.shape==(22546,3) and bias.shape==(16,3)
        assert np.array_equal(z,b+bias+direct[:,None,:])
        fields=np.load(folder/'field_logits.npz');assert set(fields.files)==set(partition)
        rebuilt=sum(fields[k] for k in partition);fgap=float(np.max(np.abs(rebuilt-direct)));assert fgap<=1e-12
        cpu=softmax(z,axis=-1).mean(1);gap=float(np.max(np.abs(cpu-q)));assert gap<=2e-12 and np.array_equal(cpu.argmax(1),q.argmax(1))
        g=rows[rows.training_role.eq(f)].reset_index(drop=True)
        assert np.array_equal(g.row_position,d.row_position) and np.array_equal(g.truth,d.truth)
        assert np.array_equal(g.pred,q[d.local].argmax(1)) and np.array_equal(g[['p0','p1','p2']].to_numpy(),q[d.local])
        assert np.array_equal(g.query_role,np.where(d.fold.eq(f),'outer_HELD','legal_TRAIN'))
        margin=z[:,:,2]-z[:,:,1]
        checks={'body_S_minus_M_mean':(b[:,:,2]-b[:,:,1]).mean(1),'facts_S_minus_M':direct[:,2]-direct[:,1],
            'bias_S_minus_M_mean':np.full(22546,(bias[:,2]-bias[:,1]).mean()),'total_S_minus_M_mean':margin.mean(1),
            'positive_S_minus_M_members':(margin>0).sum(1),'member_S_predictions':(z.argmax(-1)==2).sum(1),'mean_logit_pred':z.mean(1).argmax(1)}
        for k,v in checks.items():assert np.allclose(g[k],v[d.local],atol=1e-12,rtol=0)
        gg=geometry[geometry.training_role.eq(f)&geometry.space.eq('all16_V146_A_H2')].reset_index(drop=True)
        assert np.array_equal(gg.row_position,d.row_position) and np.array_equal(gg.pred_baseline,g.pred)
        masks={'all_ASA':np.ones(len(d),bool),'hard578':prior.known_578_cohort.to_numpy(),
            'strict51':prior.same_family_and_outer_fold_control_S.to_numpy(),'unknown_or_missing_port':~prior.both_ports_observed.to_numpy(),
            'ICMP':icmp,'mixed_actual_local':d.local.map(d.groupby('local').truth.nunique()).gt(1).to_numpy(),
            'V155_new_M16':g.truth.eq(1).to_numpy()&gg.pred_baseline.eq(g.truth).to_numpy()&gg.pred_V155_B.ne(g.truth).to_numpy()}
        for cohort,mask in masks.items():
            for role in ['legal_TRAIN','outer_HELD']:
                for cl in [1,2]:
                    selected=g[np.asarray(mask)&g.query_role.eq(role)&g.truth.eq(cl)]
                    summaries.append(dict(fold=f,cohort=cohort,role=role,truth=cl,original_rows=len(selected),errors=int(selected.pred.ne(cl).sum()),
                        mean_logit_vs_real_disagreements=int(selected.mean_logit_pred.ne(selected.pred).sum()),
                        unanimous_M_members=int(selected.member_M_predictions.eq(16).sum()),unanimous_S_members=int(selected.member_S_predictions.eq(16).sum()),
                        body_positive_S_margin=int(selected.body_S_minus_M_mean.gt(0).sum()),facts_positive_S_margin=int(selected.facts_S_minus_M.gt(0).sum()),
                        **{k+'_mean':None if selected.empty else float(selected[k].mean()) for k in ['body_S_minus_M_mean','facts_S_minus_M','bias_S_minus_M_mean','total_S_minus_M_mean','p2']}))
                    for k in partition:
                        col='field_'+k+'_S_minus_M'
                        field_summaries.append(dict(fold=f,cohort=cohort,role=role,truth=cl,field=k,original_rows=len(selected),
                            mean=None if selected.empty else float(selected[col].mean()),positive=int(selected[col].gt(0).sum()),negative=int(selected[col].lt(0).sum())))
            if cohort in ['hard578','strict51','V155_new_M16']:
                part=g[np.asarray(mask)&g.query_role.eq('outer_HELD')].drop_duplicates('local').copy();part['cohort']=cohort;witnesses.append(part)
        verdicts.append(dict(fold=f,all_22546_local_complete_logits_recomposed_exactly=True,all_29_fact_groups_verified=True,
            CPU_probability_max_gap=gap,field_sum_max_gap=fgap,original_rows=112807,original_roles_verified=True))
    pd.DataFrame(summaries).to_parquet(OUT/'v2_fixed_cohort_function_summary.parquet',index=False)
    pd.DataFrame(field_summaries).to_parquet(OUT/'v2_all_field_fixed_cohort_summary.parquet',index=False)
    pd.concat(witnesses,ignore_index=True).to_parquet(OUT/'v2_fixed_cohort_actual_local_witnesses.parquet',index=False)
    binding.extend(OUT/n for n in ['v2_fixed_cohort_function_summary.parquet','v2_all_field_fixed_cohort_summary.parquet','v2_fixed_cohort_actual_local_witnesses.parquet'])
    result=dict(status='saved_all_member_three_class_function_and_original_roles_independently_CPU_replayed',verdicts=verdicts,
        original_rows=112807,original_role_rows=338421,CPU_probability_tolerance=2e-12,field_sum_tolerance=1e-12,
        classifier_forward_calls=0,feature_function_calls=0,gradients=0,fits=0,updates=0,
        scope='Saved arrays replay only; not an independent physical model or semantic cause replay.',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in binding})
    save(OUT/'v2_verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))

if __name__=='__main__':main()
