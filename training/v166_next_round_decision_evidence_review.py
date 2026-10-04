"""Read-only model audit for the next-training decision; saved arrays only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v166_next_round_decision_evidence_review_20261002'
TRIAL = ROOT / 'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
PRIOR = ROOT / 'artifacts/v164_short_supervised_trajectory_20261002'
COVER = ROOT / 'artifacts/v165_independent_blocker_coverage_review_20261002'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists(), 'Preserve the original decision review'
    bindings = read(COVER / 'source_bindings.json')['source_sha256']
    for relative, expected in bindings.items():
        assert sha(ROOT / relative) == expected, relative
    gold_path = ROOT / 'data/official/train.parquet'
    gold = pd.read_parquet(gold_path, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert np.bincount(gold, minlength=3).tolist() == [1899723, 111728, 45420]
    roles = []
    for role in range(3):
        folder = TRIAL / f'role{role}'
        before = pd.read_parquet(folder / 'baseline/OOF_original_rows.parquet')
        after = pd.read_parquet(folder / 'treatment/OOF_original_rows.parquet')
        restored = pd.read_parquet(folder / 'endpoint/OOF_original_rows.parquet')
        positions = before.row_position.to_numpy()
        assert before.row_position.is_unique
        for table in [before, after, restored]:
            assert np.array_equal(table.row_position, positions)
            assert np.array_equal(table.truth, gold[positions])
            assert np.array_equal(table.pred, table[['p0', 'p1', 'p2']].to_numpy().argmax(axis=1))
        assert np.array_equal(before.pred, restored.pred)
        old_state = torch.load(PRIOR / f'role{role}/endpoint.pt', map_location='cpu', weights_only=True)['state']
        new_state = torch.load(folder / 'endpoint.pt', map_location='cpu', weights_only=True)['state']
        assert old_state.keys() == new_state.keys()
        assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)
        old_wrong = before.pred.ne(before.truth)
        new_wrong = after.pred.ne(after.truth)
        classes = {}
        for cls, name in [(1, 'M'), (2, 'S')]:
            population = before.truth.eq(cls)
            count = int(population.sum())
            correct = int((population & ~old_wrong).sum())
            classes[name] = dict(original_rows=count, baseline_correct=correct,
                baseline_recall=correct / count, baseline_errors=count-correct,
                treatment_errors=int((population & new_wrong).sum()),
                repairs=int((population & old_wrong & ~new_wrong).sum()),
                regressions=int((population & ~old_wrong & new_wrong).sum()))
        coverage = read(COVER / f'role{role}_function_coverage.json')
        assert all(item['actual_argmax_correct'] for item in coverage['known_actual_margins'])
        assert classes['M']['regressions'] == coverage['review']['actual_blocking_original_rows']
        assert len(coverage['fresh_function_ids']) == coverage['review']['fresh_unmeasured_functions']
        diag = read(folder / 'diagnostic.json')
        assert not diag['candidate']['accepted']
        assert diag['new_fits'] == diag['permanent_updates'] == 0
        roles.append(dict(role=role, classes=classes,
            baseline_pure_errors=int((old_wrong & before.pure_current_input).sum()),
            treatment_pure_errors=int((new_wrong & before.pure_current_input).sum()),
            measured_functions=len(coverage['known_actual_margins']),
            uncovered_functions=len(coverage['fresh_function_ids']),
            full_original_row_decisions_restored=True, all_checkpoint_tensors_restored=True))
    qualification = read(ROOT / 'artifacts/v166_coverage_core_qualification_v3_20261002/qualification.json')
    assert qualification['official_heads'] == qualification['official_derivatives'] == 0
    result = dict(status='original_gold_saved_decisions_coverage_bindings_and_restore_rechecked',
        roles=roles, verified_coverage_source_files=len(bindings), source_sha256=bindings,
        gold_sha256=sha(gold_path), own_source_sha256=sha(Path(__file__).resolve()),
        next_decision='No full retraining; bounded observed-coverage diagnostic first',
        official_calls=0, fits=0, parameter_updates=0,
        independent_source_generalization_verified=False,
        qualification_scope='CPU core only, not official finite classification quality')
    OUT.mkdir()
    (OUT / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
