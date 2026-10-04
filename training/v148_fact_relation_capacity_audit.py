"""Actual legal TRAIN factual graph capacity; no forward/gradient/fit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import load_data, fit_context
from v148_observed_relation import make_graph, observed, slots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v148_factual_relation_qualification_20261001'


def main():
    assert not OUT.exists(), 'Do not overwrite observed qualification'
    _, d = load_data()
    source = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    trace = pd.read_parquet(source, columns=['row_position', 'facts_json'])
    facts = trace.facts_json.map(json.loads)
    assert np.array_equal(trace.row_position, d.row_position)
    raw = ROOT / 'data/official/train.parquet'
    truth = pd.read_parquet(raw, columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy(np.int8)
    assert len(truth) == 2056871 and np.array_equal(d.truth, truth[d.row_position])
    OUT.mkdir(); reports = []
    for fold in range(3):
        frame, _, _, _, _ = fit_context(d, fold)
        nodes, edges, info = make_graph(frame, facts)
        assert not (set(nodes.root) & set(d.loc[d.fold.eq(fold), 'root']))
        # Full data permutation proves class/error columns are not topology inputs.
        altered = frame.copy(); altered['truth'] = np.roll(altered.truth.to_numpy(), 1)
        n2, e2, i2 = make_graph(altered, facts)
        assert n2.equals(nodes) and e2.equals(edges) and i2 == info
        nodes.to_parquet(OUT/f'fold{fold}_nodes.parquet', index=False)
        edges.to_parquet(OUT/f'fold{fold}_edges.parquet', index=False)
        anchor = nodes[nodes.index.isin(edges.left_node)]
        ids = set(zip(anchor.local, anchor.root))
        covered = np.array([(local, root) in ids for local, root in zip(frame.local, frame.root)])
        counts = frame.groupby('truth').size()
        rows_by_class = [int((covered & frame.truth.eq(cl)).sum()) for cl in [0,1,2]]
        labels = frame.groupby(['local','root']).truth.agg(lambda x: tuple(sorted(set(x))))
        same, different, mixed = 0, 0, 0
        for z in edges.itertuples(index=False):
            a, b = labels.loc[(z.left_local,z.left_root)], labels.loc[(z.right_local,z.right_root)]
            if len(a) != 1 or len(b) != 1: mixed += 1
            elif a == b: same += 1
            else: different += 1
        info.update(fold=fold, original_classification_rows=len(frame), original_classification_class_mass=[int(counts.get(cl,0)) for cl in [0,1,2]],
            graph_original_anchor_mass_by_class=rows_by_class, label_permutation_graph_exact=True,
            same_threat_class_edges_diagnostic=same, different_threat_class_edges_diagnostic=different,
            mixed_numeric_label_edges_diagnostic=mixed, node_sha256=hashlib.sha256((OUT/f'fold{fold}_nodes.parquet').read_bytes()).hexdigest(),
            edge_sha256=hashlib.sha256((OUT/f'fold{fold}_edges.parquet').read_bytes()).hexdigest())
        reports.append(info)
        print(json.dumps(dict(stage='legal_observed_graph_audited', **info), ensure_ascii=False), flush=True)
    result = dict(status='legal_label_free_fact_relations_capacity_only', latest_actual_training='V146', new_fits=0,
        new_model_forwards=0,new_gradients=0,new_updates=0,official_rows=len(truth),ASA_rows=len(d),folds=reports,
        next_training_registered=False, issue_solved=False,
        limits=['Fact relation targets do not provide new suspicious/malicious labels or imply equal threat class.',
            'This is auxiliary graph capacity, not model learning, portability, or full quality acceptance.',
            'Unknown parameters never match as concrete equal values; classifier exposure keeps every original occurrence.',
            'All exact port/ICMP values, ranges and roles remain classifier inputs; no raw time/product/IP target is used.'],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92),'/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'training/v148_observed_relation.py',ROOT/'training/v135_runtime.py',source,raw]})
    (OUT/'capacity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'], classifier_fits=0, roles=3),ensure_ascii=False))


if __name__ == '__main__': main()
