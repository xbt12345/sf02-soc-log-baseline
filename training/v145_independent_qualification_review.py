"""Recount finite diagnostic classifications without model/gradient execution."""
import json
import numpy as np
import pandas as pd
from v144_independent_decision_review import ROOT, read, sha

RUN = ROOT/'artifacts/v145_class_direction_qualification_20261001'
BASE = ROOT/'artifacts/v142_second_layer_training_20261001'
OUT = ROOT/'artifacts/v145_independent_qualification_review_20261001'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve original independent evidence')
    q = read(RUN/'qualification.json')
    registration = read(RUN/'pre_registered_probe.json')
    assert q['new_classifier_fits'] == q['new_optimizer_steps'] == q['new_mutable_parameter_updates'] == 0
    assert q['fixed_functional_parameter_evaluations'] == registration['fixed_functional_parameter_evaluations'] == 6
    for path, expected in q['evidence_bindings'].items():
        assert sha(ROOT/path) == expected
    official = ROOT/'data/official/train.parquet'
    truth = pd.read_parquet(official, columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    v144 = read(ROOT/'artifacts/v144_aux_gradient_evidence_20261001/probe.json')
    files = [ROOT/'training/v145_independent_qualification_review.py', RUN/'qualification.json',
             RUN/'pre_registered_probe.json', official]
    results = []
    for f in q['folds']:
        role = f['fold']
        checkpoint = BASE/f'fold{role}_S2/endpoint.pt'
        old = pd.read_parquet(BASE/f'fold{role}_S2/endpoint_original_rows.parquet')
        assert f['parameter_sha256'] == v144['folds'][role]['parameter_sha256']
        assert sha(checkpoint) == v144['evidence_bindings'][str(checkpoint.relative_to(ROOT))]
        assert not old.row_position.duplicated().any()
        assert np.array_equal(old.truth, truth[old.row_position])
        files += [checkpoint, BASE/f'fold{role}_S2/endpoint_original_rows.parquet']
        for detail in f['directions']:
            arm = detail['arm']
            path = RUN/f'fold{role}_{arm}_functional_trial_TRAIN_rows.parquet'
            a = pd.read_parquet(path)
            assert not a.row_position.duplicated().any()
            assert set(a.row_position) == set(old.row_position)
            assert a.training_role.eq(role).all()
            assert np.array_equal(a.truth, truth[a.row_position])
            b = old.set_index('row_position').loc[a.row_position]
            wrong = a.pred.to_numpy() != a.truth.to_numpy()
            same = a.pred.to_numpy() == b.pred.to_numpy()
            assert same.all(), 'Finite diagnostic classifications actually changed'
            classes = []
            for cl in (1, 2):
                mask = a.truth.eq(cl).to_numpy()
                count = int((wrong & mask).sum())
                assert count == detail['actual_TRAIN_classification']['M_errors' if cl == 1 else 'S_errors']
                classes.append(dict(truth=cl, original_rows=int(mask.sum()), errors=count,
                                    repaired=0, new_errors=0,
                                    member_CE_before=detail['initial_M_S_member_CE'][cl-1],
                                    member_CE_after=detail['functional_trial_M_S_member_CE'][cl-1]))
            results.append(dict(role=role, arm=arm, same_original_classifications=True,
                                original_rows=len(a), classes=classes,
                                registered_class_CE_descent=detail['strict_finite_class_risk_descent']))
            files.append(path)
    assert len(results) == 6 and not q['qualification_passed']
    assert all(not r['registered_class_CE_descent'] for r in results)
    output = dict(status='finite_probe_classification_and_loss_recount_completed', latest_actual_training='V142',
                  new_fits=0, new_gradients=0, new_updates=0, results=results,
                  failed_registered_qualification_unchanged=True,
                  classification_retention_passed=True,
                  next_action='Separate failed per-class-loss qualification from preserved classifications; reconsider hard surrogate gates prospectively, never retrospectively declare this qualification passed.',
                  limits=['Original predictions and receipts recounted, no numerical model function replay here.',
                          'CE values are source receipts, not independently recomputed logits.',
                          'Zero classification changes at one step do not prove full-run or source-held quality.',
                          'The fixed probe remains failed under its registered dual-class CE condition.',
                          'This probe does not solve fine support or stable cross-source correction.'],
                  source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    OUT.mkdir()
    (OUT/'review.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
