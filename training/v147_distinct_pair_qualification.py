"""Legal TRAIN same-class capacity with real distinct numeric inputs; no fitting."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v147_nearest_support_audit_20261001'


def main():
    target = OUT / 'distinct_pair_capacity.json'
    assert not target.exists()
    q = pd.read_parquet(OUT / 'all_known_port_query_support.parquet')
    complete = pd.read_parquet(OUT / 'whole_ASA_support_population.parquet')
    assert not complete.loc[complete.port_pair_diagnostic_eligible, 'whole_numeric_label_conflict'].any()
    results = []
    for fold in range(3):
        train = q[q.fold.ne(fold)]
        held = q[q.fold.eq(fold)]
        assert not (set(train.root) & set(held.root))
        for cl in [1, 2]:
            source = train[train.truth.eq(cl)]
            cells = source.groupby('key').agg(rows=('truth', 'size'), roots=('root', 'nunique'),
                numeric=('canonical_key', 'nunique'))
            qualified = cells[cells.roots.ge(2) & cells.numeric.ge(2)]
            rows = source[source.key.isin(qualified.index)]
            h = held[held.truth.eq(cl)]
            results.append(dict(fold=fold, truth=cl, legal_original_rows=len(source),
                qualified_keys=len(qualified), qualified_original_rows=len(rows), qualified_roots=int(rows.root.nunique()),
                qualified_numeric_inputs=int(rows.canonical_key.nunique()),
                held_qualified_key_rows=int(h.key.isin(qualified.index).sum()),
                held_V142_errors_with_qualified_key=int((h.key.isin(qualified.index) & h.pred_V142.ne(h.truth)).sum())))
    report = dict(status='legal_same_class_distinct_numeric_cross_root_capacity_only', latest_actual_training='V146',
        new_fits=0, new_gradients=0, new_model_forwards=0, new_updates=0,
        criterion='Known body TCP/UDP ports; all parsed facts except exact source value retained; same true class, at least two TRAIN roots and two canonical numeric inputs. No same-key opposite-class requirement.',
        results=results, new_training_selected=False,
        limits=['This proves observed label/parsed-context/numeric diversity, not invariance of every original textual fact.',
                'Exact source-port value omission is explicit; values have not all been proven ephemeral or semantically interchangeable.',
                'No source/port/class rule is authorized; full classifier inputs and all unknown/conflicting originals remain unchanged.'],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92), '/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), OUT/'all_known_port_query_support.parquet', OUT/'whole_ASA_support_population.parquet', OUT/'whole_population_audit.json']})
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
