"""Protocol-aware availability and actual full-fact legal positive-pair coverage."""
import json
import pandas as pd
from v142_runtime import ROOT,save,sha
from v143_fine_support_audit import OUT


def main():
    target=OUT/'support_detail.json'
    if target.exists():raise FileExistsError(target)
    d=pd.read_parquet(OUT/'fine_support_population.parquet');facts=d.facts_json.map(json.loads)
    d['protocol']=facts.map(lambda f:f.get('transport_protocol'))
    availability=d.groupby(['truth','protocol','parameter_observed','body_src_port_observed','record_src_port_observed'],dropna=False).agg(
        population=('truth','size'),errors=('wrong','sum'),sources=('root','nunique')).reset_index()
    # Full facts retain source port, flags, connection metadata and unknown masks.
    d['full_fact_key']=facts.map(lambda f:json.dumps(f,sort_keys=True,separators=(',',':')))
    detail=[];coverage=[]
    for fold in range(3):
        train=d[d.fold.ne(fold)].copy();held=d[d.fold.eq(fold)]
        assert not(set(train.root)&set(held.root))
        for parameter in ['known_destination','known_both_transport_ports']:
            q=train.parameter_observed if parameter=='known_destination' else train.protocol.isin(['tcp','udp'])&train.parameter_observed&train.body_src_port_observed
            a=train[q];s=a.groupby(['full_fact_key','truth']).agg(rows=('truth','size'),roots=('root','nunique'),locals=('local','nunique')).reset_index()
            positives=s[s.roots.ge(2)&s.locals.ge(2)]
            detail.append(dict(fold=fold,scope=parameter,original_rows=len(a),keys=a.full_fact_key.nunique(),
                same_class_multisource_distinct_input_groups=len(positives),positive_groups_M=int(positives.truth.eq(1).sum()),positive_groups_S=int(positives.truth.eq(2).sum()),
                positive_original_rows_M=int(positives.loc[positives.truth.eq(1),'rows'].sum()),positive_original_rows_S=int(positives.loc[positives.truth.eq(2),'rows'].sum())))
            for row in positives.itertuples(index=False):
                coverage.append(dict(training_role=fold,scope=parameter,key=row.full_fact_key,truth=row.truth,original_rows=row.rows,roots=row.roots,distinct_inputs=row.locals))
    pd.DataFrame(coverage).to_parquet(OUT/'legal_full_fact_positive_groups.parquet',index=False)
    save(target,dict(status='no_fit_parameter_and_positive_pair_detail',new_fits=0,new_updates=0,protocol_availability=availability.to_dict('records'),
        legal_TRAIN_full_fact_positive_coverage=detail,first_mastery_guard='training/v142_retention_check.py',
        second_issue_solved=False,limits=['ICMP has no transport port; missing record src_port is not an ICMP feature defect.',
        'Known destination is weaker than both known transport ports; neither scope invents masked values.',
        'These legal TRAIN positive groups are feasibility evidence only; no supervised contrastive objective is registered or fitted.',
        'Multi-source same-label full facts may still coexist with opposite-label rows; identity/context cannot be discarded.'],
        source_sha256=sha(__file__),population_sha256=sha(OUT/'fine_support_population.parquet')))
    print(json.dumps(detail,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
