"""Read saved journals and official labels only; no model, forward, fit or gradient."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from v144_independent_decision_review import ROOT, read, sha

RUN = ROOT/'artifacts/v146_guarded_pair_training_20261001'
BASE = ROOT/'artifacts/v142_second_layer_training_20261001'
OUT = ROOT/'artifacts/v146_independent_review_20261001'


def class_counts(truth, prediction):
    out = {}
    for y, name in enumerate(['N', 'M', 'S']):
        true = truth == y
        positive = prediction == y
        tp = int((true & positive).sum())
        support = int(true.sum())
        predicted = int(positive.sum())
        out[name] = dict(support=support, errors=support-tp,
                         precision=tp/max(predicted, 1), recall=tp/max(support, 1),
                         F1=2*tp/max(support+predicted, 1))
    return out


def main():
    plan = read(ROOT/'training/review_policy/v146_guarded_pair_plan.json')
    seal = read(RUN/'run_seal.json')
    assert seal['plan_sha256'] == sha(ROOT/'training/review_policy/v146_guarded_pair_plan.json')
    checked_sources = {}
    for p, expected in seal['source_sha256'].items():
        if p.startswith('training/') and not p.startswith('training/__pycache__/'):
            assert sha(ROOT/p) == expected, 'Sealed training source changed: '+p
            checked_sources[p] = expected
    truth = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(truth) == 2056871 and not pd.isna(truth).any()
    folds = pd.read_parquet(ROOT/'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet',
                            columns=['row_position', 'root', 'proposed_fold']).set_index('row_position')
    registry = read(BASE/'verified_TRAIN_mastery_registry.json')
    roles = []
    for f in range(3):
        for arm in ['A', 'B']:
            folder = RUN/f'fold{f}_{arm}'
            if not (folder/'fit.json').is_file():
                continue
            receipt = read(folder/'fit.json')
            for n, k in [('endpoint.pt', 'model_sha256'), ('sealed_all_prob.npy', 'probability_sha256'),
                         ('endpoint_original_rows.parquet', 'rows_sha256'), ('gradients.jsonl', 'gradients_sha256'),
                         ('proposals.jsonl', 'proposals_sha256'), ('progress.json', 'progress_sha256')]:
                assert sha(folder/n) == receipt[k]
            rows = pd.read_parquet(folder/'endpoint_original_rows.parquet')
            old = pd.read_parquet(BASE/f'fold{f}_S2/endpoint_original_rows.parquet').set_index('row_position')
            assert not rows.row_position.duplicated().any() and set(rows.row_position) == set(old.index)
            assert np.array_equal(rows.truth, truth[rows.row_position])
            ff = folds.loc[rows.row_position]
            assert ff.proposed_fold.ne(f).all() and np.array_equal(ff.root, rows.root)
            assert not set(rows.root) & set(folds.loc[folds.proposed_fold.eq(f), 'root'])
            assert rows.training_role.eq(f).all() and rows.arm.eq(arm).all()
            mass = [int(rows.truth.eq(y).sum()) for y in range(3)]
            assert mass == receipt['original_class_mass']
            q = np.load(folder/'sealed_all_prob.npy')
            assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
            assert np.max(np.abs(q.sum(1)-1)) < 1e-12
            assert np.array_equal(rows.pred, q[rows.local].argmax(1))
            assert np.array_equal(rows.pred_start, old.loc[rows.row_position, 'pred'])
            wrong = rows.pred.ne(rows.truth)
            newly_wrong = wrong & rows.pred_start.eq(rows.truth)
            pure = rows.groupby('local').truth.transform('nunique').eq(1)
            assert np.array_equal(pure, rows.pure_TRAIN_input)
            assert not newly_wrong.any() and not (wrong & pure).any()
            errors = [int((wrong & rows.truth.eq(y)).sum()) for y in [1, 2]]
            assert errors == [0, [22, 6, 28][f]]
            guard_info = registry['scopes'][f]
            assert sha(ROOT/guard_info['guard']) == guard_info['guard_sha256']
            guard = pd.read_parquet(ROOT/guard_info['guard'])
            guard_rows = rows.set_index('row_position').loc[guard.row_position]
            assert guard_rows.pred.eq(guard_rows.truth).all()
            gradients = [json.loads(s) for s in (folder/'gradients.jsonl').read_text().splitlines()]
            proposals = [json.loads(s) for s in (folder/'proposals.jsonl').read_text().splitlines()]
            history = read(folder/'progress.json')
            assert len(gradients) == receipt['full_gradient_evaluations'] <= plan['solver']['max_gradients']
            assert len(proposals) == receipt['proposal_evaluations'] <= plan['solver']['max_proposals']
            assert len(history) == receipt['accepted_updates'] <= plan['solver']['max_updates']
            assert [v['proposal'] for v in proposals] == list(range(1, len(proposals)+1))
            accepted = [v for v in proposals if v['accepted']]
            assert [v['parameter_sha256'] for v in accepted] == [v['parameter_sha256'] for v in history]
            lam = receipt['lambda_fixed'] if arm == 'B' else 0.
            previous_state = read(folder/'started.json')['initial_parameter_sha256']
            steps = []
            for idx, g in enumerate(gradients, 1):
                assert g['full_gradient_evaluation'] == idx and g['finite']
                assert g['parameter_sha256'] == previous_state
                assert g['original_class_mass_seen'] == mass
                assert abs(g['objective']-g['original_member_CE']-lam*g['auxiliary_loss']) < 1e-12
                group = [v for v in proposals if v['gradient'] == idx]
                assert sum(v['accepted'] for v in group) <= 1
                assert not any(v['accepted'] for v in group[:-1])
                for bt, v in enumerate(group):
                    assert v['backtrack'] == bt and v['original_class_mass_seen'] == mass
                    assert all(math.isfinite(v[k]) for k in ['objective', 'original_member_CE', 'auxiliary_loss', 'step'])
                    assert abs(v['objective']-v['original_member_CE']-lam*v['auxiliary_loss']) < 1e-12
                    expected_ce = np.dot(mass, v['per_class_member_CE'])/sum(mass)
                    assert abs(expected_ce-v['original_member_CE']) < 1e-12
                    bound = g['objective']-plan['solver']['armijo']*v['step']*g['gradient_norm']
                    assert abs(bound-v['Armijo_bound']) < 1e-15
                    guard_ok = v['stats']['mastered'] and v['stats']['new_errors_vs_start'] == 0
                    ok = guard_ok and v['objective'] < g['objective'] and v['objective'] <= bound
                    assert v['classification_guard'] == guard_ok and v['accepted'] == ok
                    if bt:
                        assert v['step'] == group[bt-1]['step']*plan['solver']['shrink']
                    if v['accepted']:
                        previous_state = v['parameter_sha256']
                        steps.append(v['step'])
            if history:
                assert history[-1]['parameter_sha256'] == previous_state
            roles.append(dict(fold=f, arm=arm, official_TRAIN_rows=len(rows), class_mass=mass,
                              errors_M_S=errors, preserved_guard_rows=len(guard), new_errors=0,
                              gradients=len(gradients), proposals=len(proposals), accepted=len(accepted),
                              rejected=len(proposals)-len(accepted), termination=receipt['termination'],
                              first_objective=gradients[0]['objective'] if gradients else None,
                              final_objective=history[-1]['values']['objective'] if history else None,
                              accepted_step_min=min(steps) if steps else None,
                              accepted_step_max=max(steps) if steps else None,
                              parameter_delta_L2=receipt['parameter_delta_L2'],
                              fit_receipt_sha256=sha(folder/'fit.json')))
    output = dict(status='completed_fit_receipts_recounted', completed_fits=len(roles),
                  all_six_fits_completed=len(roles) == 6, roles=roles,
                  new_fits=0, new_forward_passes=0, new_gradients=0, new_parameter_updates=0,
                  checked_sealed_training_sources=checked_sources,
                  validation_limits=['Saved probabilities, journals and official truth checked; no fresh model replay here.',
                                     'Rejected-state values and gradients are recorded receipts, not recomputed logits/derivatives.',
                                     'No HELD score read during partial training; final full-population review requires final_delivery.'])
    if (RUN/'final_delivery.json').is_file():
        assert len(roles) == 6
        full = pd.read_parquet(RUN/'full_prediction_ledger.parquet')
        assert len(full) == len(truth) and np.array_equal(full.row_position, np.arange(len(truth)))
        output['full_official_classes'] = {a: class_counts(truth, full['pred_'+a].to_numpy()) for a in ['A0', 'V142', 'A', 'B']}
        asa = pd.read_parquet(RUN/'ASA_prediction_ledger.parquet')
        assert np.array_equal(asa.truth, truth[asa.row_position])
        for a in ['A0', 'V142', 'A', 'B']:
            assert np.array_equal(asa['pred_'+a], full.loc[asa.row_position, 'pred_'+a])
        transitions = []
        for old_name in ['A0', 'V142', 'A']:
            for cls in [1, 2]:
                z = asa[asa.truth.eq(cls)]
                old_ok = z['pred_'+old_name].eq(cls)
                new_ok = z.pred_B.eq(cls)
                transitions.append(dict(reference=old_name, truth=cls, rows=len(z),
                                        repairs=int((~old_ok & new_ok).sum()),
                                        new_errors=int((old_ok & ~new_ok).sum()),
                                        reference_errors=int((~old_ok).sum()), B_errors=int((~new_ok).sum())))
        output['ASA_original_row_transitions'] = transitions
        output['final_delivery_sha256'] = sha(RUN/'final_delivery.json')
        output['status'] = 'training_journals_and_complete_official_prediction_recount_completed'
        source = []
        for (f, root, cls), z in asa.groupby(['fold', 'root', 'truth']):
            source.append(dict(fold=int(f), root=int(root), truth=int(cls), rows=len(z),
                               **{a+'_errors':int(z['pred_'+a].ne(cls).sum()) for a in ['A0', 'V142', 'A', 'B']}))
        OUT.mkdir(exist_ok=True)
        pd.DataFrame(source).to_csv(OUT/'source_error_transitions.csv', index=False)
    output['auditor_sha256'] = sha(Path(__file__))
    OUT.mkdir(exist_ok=True)
    target = OUT/'review.json'
    temp = OUT/'review.json.tmp'
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(target)
    print(json.dumps({k: output[k] for k in ['status', 'completed_fits', 'all_six_fits_completed', 'roles']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
