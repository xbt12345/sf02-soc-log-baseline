"""Recompute all original factual readout results; zero model or decoder fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import read,sha,check_bindings
from v150_field_decoder import FIELDS,decode

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v150_field_readout_diagnostic_20261001'
LAYERS=['H1','V146_B_H2']
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def clean(v):return 'unknown' if v is None else str(v)

def main():
    assert not (OUT/'result_audit.json').exists()
    check_bindings(read(OUT/'run_seal.json')['source_sha256'])
    entries=read(OUT/'target_schema.json')['entries']
    names=read(ROOT/'artifacts/v92_evidence_training_20260928/representation_contract.json')['old_fact_coordinate_names']
    # Independent expected values from already audited actual fact coordinates.
    cols=[]
    for e in entries:
        f=e['field']
        cols.extend([names.index(f+':bit'+str(i)) for i in range(e['stop']-e['start'])] if e['values'] is None else [names.index(f+'='+v) for v in e['values']])
    exact=decode((np.load(ROOT/'artifacts/v138_single_issue_round1_20260930/facts.npy')[:,cols]>0).astype(float),entries)
    trace=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['row_position','facts_json'])
    facts=trace.set_index('row_position').facts_json.map(json.loads).to_dict()
    conditional=pd.read_parquet(ROOT/'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet')
    support=conditional[['row_position','known_578_cohort','same_class_roots','opposite_class_rows']].set_index('row_position')
    field_rows=[];value_rows=[];scope_rows=[];bundles={z:[] for z in LAYERS};sources={}
    for fold in range(3):
        for layer in LAYERS:
            folder=OUT/f'fold{fold}_{layer}';receipt=read(folder/'fit.json');r=pd.read_parquet(folder/'all_original_field_ledger.parquet')
            assert receipt['status']=='diagnostic_decoder_fit_executed' and receipt['classifier_state_unchanged']
            assert sha(folder/'all_original_field_ledger.parquet')==receipt['rows_sha256'] and sha(folder/'decoder.pt')==receipt['decoder_sha256']
            assert len(r)==112807 and not r.row_position.duplicated().any()
            loc=r.local.to_numpy();proto=np.array([facts[int(i)].get('transport_protocol') for i in r.row_position])
            fit=r.readout_role.eq('readout_FIT').to_numpy()
            for field in FIELDS:
                expected=exact[field][loc];pred=r[field+'_decoded'].to_numpy();known=r[field+'_observed'].to_numpy()
                assert np.array_equal(pred==expected,r[field+'_exact'])
                values=np.array([clean(v) for v in expected]);seen=set(values[fit&known]);value_seen=np.array([v in seen for v in values])
                # Unknown and not applicable are reported separately; neither is a concrete value match.
                applicable=np.isin(proto,['tcp','udp']) if field.startswith(('src_port','dst_port')) else proto=='icmp' if field.startswith('icmp_') else np.ones(len(r),bool)
                strata={'all_observed':known,'observed_value_seen_decoder_fit':known&value_seen,
                    'observed_value_unseen_decoder_fit':known&~value_seen,'unknown_applicable':applicable&~known,
                    'not_applicable':~applicable,'observed_zero':known&(values=='0')}
                for role in ['readout_FIT','inner_HELD','outer_HELD']:
                    population=r.readout_role.eq(role).to_numpy()
                    for name,mask in strata.items():
                        take=population&mask;n=int(take.sum());errors=int((take&~r[field+'_exact'].to_numpy()).sum())
                        field_rows.append(dict(fold=fold,layer=layer,field=field,role=role,stratum=name,original_rows=n,errors=errors,
                            accuracy=None if not n else (n-errors)/n,mode_errors=int((take&~r[field+'_mode_exact'].to_numpy()).sum())))
                    # All values of categorical fields and every observed numeric value remain available.
                    for value in sorted(set(values[population])):
                        take=population&(values==value);n=int(take.sum());errors=int((take&~r[field+'_exact'].to_numpy()).sum())
                        value_rows.append(dict(fold=fold,layer=layer,field=field,role=role,value=value,original_rows=n,errors=errors))
            r['all_observed_fields_exact']=np.logical_and.reduce([(~r[f+'_observed']|r[f+'_exact']).to_numpy() for f in FIELDS])
            r['all_field_states_exact']=np.logical_and.reduce([r[f+'_exact'].to_numpy() for f in FIELDS])
            r['known_ports']=r.src_port_fixed_observed&r.dst_port_fixed_observed
            r['ports_exact']=r.known_ports&r.src_port_fixed_exact&r.dst_port_fixed_exact
            r=r.join(support,on='row_position');r['known_578_cohort']=r.known_578_cohort.fillna(False).astype(bool)
            for role in ['readout_FIT','inner_HELD','outer_HELD']:
                population=r.readout_role.eq(role).to_numpy();wrong=(r.classifier_pred_frozen!=r.truth).to_numpy()
                strata={'all_original':np.ones(len(r),bool),'known_578':r.known_578_cohort.to_numpy(),
                    'known_ports':r.known_ports.to_numpy(),
                    'wrong_S':wrong&r.truth.eq(2).to_numpy(),'correct_S':~wrong&r.truth.eq(2).to_numpy(),
                    'correct_rare_same_class_S':~wrong&r.truth.eq(2).to_numpy()&r.same_class_roots.le(1).to_numpy(),
                    'correct_rare_same_class_M':~wrong&r.truth.eq(1).to_numpy()&r.same_class_roots.le(1).to_numpy(),
                    'canonical_unseen_decoder':~r.canonical_seen_by_decoder.to_numpy()}
                for name,mask in strata.items():
                    take=population&mask
                    scope_rows.append(dict(fold=fold,layer=layer,role=role,stratum=name,original_rows=int(take.sum()),
                        classifier_errors=int((take&wrong).sum()),all_observed_fields_exact=int((take&r.all_observed_fields_exact).sum()),
                        all_field_states_exact=int((take&r.all_field_states_exact).sum()),known_ports=int((take&r.known_ports).sum()),
                        complete_port_values_exact=int((take&r.ports_exact).sum()),
                        ports_exact_but_class_wrong=int((take&r.ports_exact&wrong).sum()),
                        observed_fields_exact_but_class_wrong=int((take&r.all_observed_fields_exact&wrong).sum())))
            bundles[layer].append(r[r.readout_role.eq('outer_HELD')].assign(decoder_layer=layer,decoder_fold=fold))
            for p in [folder/'fit.json',folder/'decoder.pt',folder/'all_original_field_ledger.parquet']:sources[p.relative_to(ROOT).as_posix()]=sha(p)
    fields=pd.DataFrame(field_rows);values=pd.DataFrame(value_rows);scopes=pd.DataFrame(scope_rows)
    fields.to_parquet(OUT/'all_field_state_metrics.parquet',index=False);values.to_parquet(OUT/'all_exact_value_metrics.parquet',index=False);scopes.to_parquet(OUT/'all_classifier_control_metrics.parquet',index=False)
    outer=[]
    for layer,parts in bundles.items():
        rows=pd.concat(parts,ignore_index=True);assert len(rows)==112807 and not rows.row_position.duplicated().any()
        hard=rows[rows.known_578_cohort];assert len(hard)==578
        rows.to_parquet(OUT/f'{layer}_whole_outer_development_ledger.parquet',index=False)
        outer.append(dict(layer=layer,whole_outer_rows=len(rows),known_578_original_rows=len(hard),known_578_class_errors=int((hard.classifier_pred_frozen!=hard.truth).sum()),
            known_578_complete_ports_exact=int(hard.ports_exact.sum()),known_578_all_observed_fields_exact=int(hard.all_observed_fields_exact.sum()),
            known_578_all_states_exact=int(hard.all_field_states_exact.sum()),
            known_578_ports_exact_but_class_wrong=int((hard.ports_exact&(hard.classifier_pred_frozen!=hard.truth)).sum())))
    primary=fields[fields.stratum.eq('all_observed')].groupby(['layer','role','field'],sort=True)[['original_rows','errors','mode_errors']].sum().reset_index()
    result=dict(status='six_frozen_factual_decoder_fits_audited_not_classifier_training',latest_actual_classifier='V146',
        actual_diagnostic_decoder_fits=6,normal_equation_solves=6,new_classifier_fits=0,new_classifier_gradients=0,new_classifier_updates=0,
        fitted_decoder_parameters_total=sum(read(OUT/f'fold{f}_{l}/fit.json')['decoder_coefficient_values'] for f in range(3) for l in LAYERS),
        representation_feature_forwards=3+sum(read(OUT/f'fold{f}_{l}/fit.json')['feature_model_forward_calls'] for f in range(3) for l in LAYERS),
        original_frequency_preserved=True,canonical_min_local_representative=True,classifier_unchanged_all_six=True,
        backbone_seen_inner_held_labels=True,outer_previously_inspected_development=True,blind_generalization=False,
        quality_acceptance=False,issue_solved=False,next_classifier_training_registered=False,whole_outer=outer,
        all_observed_field_metrics=primary.to_dict('records'),
        limits=['A single fixed ridge linear decoder only diagnoses linear readability; its errors do not prove absent information or impossibility of nonlinear decoding.',
            'Frozen classifier retains exact fact-input bypass. Hidden-field reading is not whole-model information loss or new M/S evidence.',
            'Backbone classification has seen inner-held labels; outer folds were previously inspected; no independent blind acceptance.',
            'Known 578 cohort and same/opposite class supports are post-fit controls only; not field, coefficient, layer, training or checkpoint selectors.',
            'One simultaneous 348-coordinate decoder target preserves unknown states; equal coordinate MSE is not 13 equal field weights or threat-class loss.',
            'Numeric accuracy requires complete reconstructed integer values, with zero/unknown/not-applicable strata; bit accuracy not used as complete-value accuracy.'],
        source_sha256={**sources,Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__)),
            'artifacts/v150_field_readout_diagnostic_20261001/run_seal.json':sha(OUT/'run_seal.json'),
            'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet':sha(ROOT/'artifacts/v147_independent_conditional_support_20261001/all_known_port_conditional_support.parquet')})
    save(OUT/'result_audit.json',result)
    print(json.dumps(dict(status=result['status'],actual_decoder_fits=6,classifier_fits=0,whole_outer=outer,all_observed_field_metrics=primary[primary.field.isin(['transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed'])].to_dict('records')),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
