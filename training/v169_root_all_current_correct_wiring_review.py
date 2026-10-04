"""Actual frozen original rows and the new guard helper; no model evaluation."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_review import ROOT, read, sha, check_bindings
from v169_current_correct_context import apply

OUT = ROOT/'artifacts/v169_root_all_current_correct_wiring_review_20261002'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    registry_file = ROOT/'artifacts/v169_initial_current_correct_protection_20261002/review.json'
    registry = read(registry_file)
    check_bindings(registry['source_sha256'])
    gold_file = ROOT/'data/official/train.parquet'
    labels = pd.read_parquet(gold_file, columns=['label_binary']).label_binary
    gold = labels.map({'benign':0, 'malicious':1, 'suspicious':2}).to_numpy()
    assert len(gold) == 2056871 and not pd.isna(gold).any()
    files = [Path(__file__).resolve(), registry_file, gold_file,
             ROOT/'training/v169_current_correct_context.py',
             ROOT/'training/v169_prior_pair_training_entry_v9.py']
    cases = []
    for role, spec in enumerate(registry['roles']):
        source = ROOT/spec['reference']
        frame = pd.read_parquet(source)
        assert np.array_equal(frame.truth.to_numpy(), gold[frame.row_position.to_numpy()])
        correct = frame.pred.eq(frame.truth).to_numpy()
        for arm in ['A', 'B']:
            ctx = {'OOF_rows': frame.copy()}
            result = apply(ctx, role)
            assert np.array_equal(ctx['OOF_rows'].protected_correct.to_numpy(), correct)
            assert len(ctx['OOF_rows']) == len(frame)
            assert np.array_equal(ctx['OOF_rows'][['row_position','local','truth']].to_numpy(),
                                  frame[['row_position','local','truth']].to_numpy())
            assert result['additional_mixed_rows'] == [106,6,94][role]
            cases.append(dict(role=role, arm=arm, frozen_original_gold_checked=True,
                original_rows=len(frame), all_current_correct_protected=int(correct.sum()),
                additional_mixed_rows=result['additional_mixed_rows'],
                complete_scoring_population_unchanged=True))
        for mode in ['shuffled_row_identity', 'revoked_old_correct', 'protected_old_wrong']:
            mutated = frame.copy()
            if mode == 'shuffled_row_identity':
                mutated = mutated.iloc[::-1].reset_index(drop=True)
            elif mode == 'revoked_old_correct':
                index = mutated.index[mutated.protected_correct][0]
                mutated.loc[index, 'protected_correct'] = False
            else:
                index = mutated.index[mutated.pred.ne(mutated.truth)][0]
                mutated.loc[index, 'protected_correct'] = True
            try:
                apply({'OOF_rows':mutated}, role)
            except ValueError as exc:
                cases.append(dict(role=role, negative_case=mode,
                    refused=True, actual_exception=str(exc)))
            else:
                raise AssertionError('Malformed existing-correct guard accepted: '+mode)
        files.extend([source, ROOT/spec['ledger'], ROOT/spec['extra']])
    entry = (ROOT/'training/v169_prior_pair_training_entry_v9.py').read_text(encoding='utf-8')
    assert 'initial_protection=apply_all_current_correct(self.ctx,role)' in entry
    assert 'from v169_pair_lifecycle_v3 import' in entry
    assert "if global_stop:" in entry and "global_technical_or_storage_fault_stop_remaining_fits" in entry
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='all_current_correct_guard_helper_replayed_on_actual_gold_and_rows',
        all_checks_passed=True, cases=cases,
        additional_original_mixed_rows=[106,6,94],
        all_original_rows_and_labels_retained=True,
        entry_v9_helper_and_revised_lifecycle_import_wired=True,
        entry_v9_global_fault_abort_present_not_yet_backend_qualified=True,
        official_heads=0, official_features=0, official_derivatives=0,
        fits=0, permanent_updates=0, supports_physical_seal=False,
        source_sha256=bindings,
        scope='Actual original-row guard helper and entry source wiring only; no official feature/head/gradient calls. Full backend, physical resource budget and seal remain separate.')
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','cases']},ensure_ascii=False))


if __name__ == '__main__':
    main()
