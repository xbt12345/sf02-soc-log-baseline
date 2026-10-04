"""Check the scope of zero masked-fact distance; no model computations/fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import load_data, fit_context
from v148_observed_relation import FIELDS, observed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v148_factual_relation_qualification_20261001'


def main():
    target=OUT/'mask_scope_audit.json';assert not target.exists()
    capacity=json.loads((OUT/'capacity.json').read_text(encoding='utf-8'))
    _,d=load_data();source=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    trace=pd.read_parquet(source,columns=['row_position','facts_json']);assert np.array_equal(trace.row_position,d.row_position)
    facts=trace.facts_json.map(json.loads)
    reports=[];examples=[]
    for fold in range(3):
        frame,_,_,_,_=fit_context(d,fold)
        nodes=pd.read_parquet(OUT/f'fold{fold}_nodes.parquet');edges=pd.read_parquet(OUT/f'fold{fold}_edges.parquet')
        labels=frame.groupby(['local','root']).truth.agg(lambda x:tuple(sorted(set(x))))
        fact_strings=frame.assign(facts_json=trace.loc[frame.index,'facts_json']).groupby('local').facts_json.first()
        fs=[json.loads(fact_strings.loc[int(local)]) for local in nodes.local]
        active=capacity['folds'][fold]['active_fields']
        masks=np.array([[observed(f,k) for k in active] for f in fs])
        canonical=[json.dumps(f,sort_keys=True,separators=(',',':')) for f in fs]
        summary={};cross=0;sampled=set()
        for z in edges[edges.target_distance.eq(0)].itertuples(index=False):
            a,b=labels.loc[(z.left_local,z.left_root)],labels.loc[(z.right_local,z.right_root)]
            if len(a)!=1 or len(b)!=1 or a==b:continue
            cross+=1
            same_mask=bool(np.array_equal(masks[z.left_node],masks[z.right_node]));same_fact=canonical[z.left_node]==canonical[z.right_node]
            kind='same_complete_parsed_facts' if same_fact else ('different_observation_masks' if not same_mask else 'same_masks_different_full_facts')
            item=summary.setdefault(kind,dict(edges=0,original_anchor_weight=0.));item['edges']+=1;item['original_anchor_weight']+=z.weight
            if kind not in sampled:
                sampled.add(kind);examples.append(dict(fold=fold,scope=kind,left_root=z.left_root,right_root=z.right_root,
                    left_local=z.left_local,right_local=z.right_local,left_truth=a[0],right_truth=b[0],target_distance=0.,
                    left_canonical_key=nodes.canonical_key.iloc[z.left_node],right_canonical_key=nodes.canonical_key.iloc[z.right_node],
                    left_facts=canonical[z.left_node],right_facts=canonical[z.right_node]))
        assert cross==[3231,355,3307][fold]
        reports.append(dict(fold=fold,zero_distance_cross_threat_class_edges=cross,scope_partition=summary))
    pd.DataFrame(examples).to_parquet(OUT/'zero_distance_cross_class_scope_examples.parquet',index=False)
    result=dict(status='masked_fact_zero_not_whole_input_or_threat_equivalence',latest_actual_training='V146',
        new_fits=0,new_model_forwards=0,new_gradients=0,new_updates=0,folds=reports,
        limits=['Equal common observed fields does not assert equality of unobserved fields or observation state.',
            'Even equal complete parsed facts do not imply equal original model input or equal threat category.',
            'Examples are lawful TRAIN diagnostics, not a training whitelist or new threat labels.'],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92),'/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'training/v148_observed_relation.py',source,OUT/'capacity.json',OUT/'gradient_probe.json']})
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
