"""Read-only independent original-row recount after all V155 fits/evaluation.

No model construction, forward, gradient, fitting, or parameter update. Saved
ensemble probabilities are not substituted for independent-member CE gradients.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v155_guarded_full_gradient_sam_20261001'
OUT = ROOT / 'artifacts/v155_independent_endpoint_audit_20261001'
REF = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001'
OLD = ROOT / 'artifacts/v146_guarded_pair_training_20261001'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bindings(values):
    for name, expected in values.items():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        assert p.is_file() and sha(p) == expected, name


def classes(y, p):
    out = {}
    for c in range(3):
        support = int((y == c).sum())
        called = int((p == c).sum())
        tp = int(((y == c) & (p == c)).sum())
        out[str(c)] = dict(support=support, correct=tp, missed=support-tp,
                          false_called=called-tp,
                          recall=tp/support if support else None,
                          precision=tp/called if called else None,
                          f1=2*tp/(support+called) if support else None)
    return out


def changes(y, before, after):
    return {str(c): dict(support=int((y == c).sum()),
        repairs=int(((y == c) & (before != y) & (after == y)).sum()),
        new_errors=int(((y == c) & (before == y) & (after != y)).sum()),
        before_errors=int(((y == c) & (before != y)).sum()),
        after_errors=int(((y == c) & (after != y)).sum())) for c in range(3)}


def main():
    assert (RUN / 'final_delivery.json').is_file(), 'Wait for actual completed evaluation'
    assert not OUT.exists(), 'Preserve independent audit receipts'
    delivery = read(RUN / 'final_delivery.json')
    seal = read(RUN / 'run_seal.json')
    authorization = read(ROOT / 'artifacts/v155_independent_pretrain_review_20261001/review.json')
    bindings(authorization['source_sha256'])
    bindings(seal['source_sha256'])
    bindings(read(RUN / 'fit_activation.json')['source_sha256'])
    plan = read(ROOT / 'training/review_policy/v155_guarded_full_gradient_sam_plan.json')
    assert seal['plan_sha256'] == sha(ROOT / 'training/review_policy/v155_guarded_full_gradient_sam_plan.json')
    assert delivery['seal_sha256'] == sha(RUN / 'run_seal.json')
    assert sha(RUN / 'quality.json') == delivery['quality_sha256']
    assert sha(RUN / 'verification.json') == delivery['verification_sha256']
    assert plan['fits'] == 6 and not plan['HELD_used_for_fitting_or_selection']
    assert plan['candidate'] == 'B' and not plan['automatic_repeat']
    assert len(list(RUN.glob('fold*_*/started.json'))) == 6
    assert len(list(RUN.glob('fold*_*/fit.json'))) == 6
    original = pd.read_parquet(REF / 'all_original_classifier_gap_and_control_ledger.parquet')
    previous_train = pd.read_parquet(REF / 'both_arms_legal_TRAIN_outer_control.parquet')
    previous_train = previous_train[previous_train.arm.eq('A')].reset_index(drop=True)
    assert len(original) == 112807 and original.row_position.is_unique
    assert len(previous_train) == 225614
    identity = ['row_position', 'local', 'root', 'fold', 'truth', 'canonical_key', 'pure_TRAIN_input']
    summaries, roles, qouter = [], {a: [] for a in 'AB'}, {a: np.empty((112807, 3)) for a in 'AB'}
    for fold in range(3):
        train_reference = pd.read_parquet(RUN / f'fold{fold}_TRAIN_reference.parquet')
        prior = previous_train[previous_train.training_role.eq(fold)].reset_index(drop=True)
        assert train_reference[identity].equals(prior[identity])
        mass = np.bincount(prior.truth, minlength=3).tolist()
        zero = np.load(RUN / f'fold{fold}_zero_probability.npy')
        assert np.array_equal(zero, np.load(OLD / f'fold{fold}_A/sealed_all_prob.npy'))
        radius = read(RUN / f'fold{fold}_fixed_radius.json')['radius_L2']
        for arm in 'AB':
            folder = RUN / f'fold{fold}_{arm}'
            fit = read(folder / 'fit.json')
            assert (fit['arm'], fit['fold'], fit['status']) == (arm, fold, 'fit_executed')
            assert not fit['selected_by_score'] and fit['seal_sha256'] == sha(RUN / 'run_seal.json')
            files = dict(model_sha256='endpoint.pt', probability_sha256='sealed_all_prob.npy',
                         rows_sha256='endpoint_original_rows.parquet', gradients_sha256='gradients.jsonl',
                         proposals_sha256='proposals.jsonl', progress_sha256='progress.json')
            for key, name in files.items():
                assert sha(folder / name) == fit[key], (folder.name, name)
            steps = [json.loads(z) for z in (folder / 'gradients.jsonl').read_text().splitlines()]
            trials = [json.loads(z) for z in (folder / 'proposals.jsonl').read_text().splitlines()]
            history = read(folder / 'progress.json')
            attempts, updates = fit['outer_gradient_attempts'], fit['accepted_updates']
            assert 0 <= updates <= attempts <= 200 and updates <= len(trials) <= 600
            assert len(steps) == fit['full_gradient_evaluations'] == attempts*(1 if arm == 'A' else 2)
            assert len(trials) == fit['proposal_evaluations'] and len(history) == updates
            assert [z['full_gradient_evaluation'] for z in steps] == list(range(1, len(steps)+1))
            assert [z['proposal'] for z in trials] == list(range(1, len(trials)+1))
            assert [z['location'] for z in steps] == (['base'] if arm == 'A' else ['base', 'adversarial'])*attempts
            assert all(z['original_class_mass_seen'] == mass and z['radius_L2'] == radius and z['finite'] for z in steps)
            assert fit['ordinary_class_mass'] == mass and fit['radius_L2'] == radius
            accepted = [z for z in trials if z['accepted']]
            assert len(accepted) == updates
            assert [z['parameter_sha256'] for z in accepted] == [z['parameter_sha256'] for z in history]
            assert [z['accepted_update'] for z in history] == list(range(1, updates+1))
            for z in accepted:
                assert z['classification_guard'] and z['stats']['mastered'] and z['stats']['new_errors_vs_start'] == 0
                assert z['frozen_epsilon_proxy_CE'] < z['frozen_epsilon_base_proxy_CE']
                assert z['frozen_epsilon_proxy_CE'] <= z['proxy_Armijo_bound']
                assert not z['moving_epsilon_proxy_monotonic_claimed']
            assert fit['termination'] in ['outer_gradient_budget', 'proposal_budget', 'accepted_update_budget', 'zero_gradient', 'no_feasible_step']
            if fit['termination'] == 'accepted_update_budget':
                assert updates == 200
            elif fit['termination'] == 'outer_gradient_budget':
                assert attempts == 200
            elif fit['termination'] == 'proposal_budget':
                assert len(trials) == 600
            q = np.load(folder / 'sealed_all_prob.npy')
            assert q.shape == zero.shape and np.isfinite(q).all()
            assert (q >= 0).all() and (q <= 1).all() and np.allclose(q.sum(1), 1, rtol=0, atol=1e-12)
            rows = pd.read_parquet(folder / 'endpoint_original_rows.parquet')
            assert rows[identity].equals(train_reference[identity])
            assert np.array_equal(rows.pred_start, zero[rows.local].argmax(1))
            assert np.array_equal(rows[['p0', 'p1', 'p2']], q[rows.local])
            assert np.array_equal(rows.pred, q[rows.local].argmax(1))
            wrong, initial_correct = rows.pred.ne(rows.truth), rows.pred_start.eq(rows.truth)
            assert int((wrong & rows.pure_TRAIN_input).sum()) == 0
            assert int((wrong & initial_correct).sum()) == 0
            assert int((wrong & rows.truth.eq(1)).sum()) == 0
            assert int((wrong & rows.truth.eq(2)).sum()) == [22, 6, 28][fold]
            rolemask = original.fold.eq(fold).to_numpy()
            qouter[arm][rolemask] = q[original.loc[rolemask, 'local']]
            roles[arm].append(rows)
            rejects = [z for z in trials if not z['accepted']]
            summaries.append(dict(arm=arm, fold=fold, gradient_attempts=attempts, gradients=len(steps),
                proposals=len(trials), updates=updates, termination=fit['termination'], seconds=fit['seconds'],
                TRAIN_errors=int(wrong.sum()), initial_correct_rows=int(initial_correct.sum()),
                new_errors_on_initial_correct=0, rejected=int(len(rejects)),
                rejected_classification_guard=int(sum(not z['classification_guard'] for z in rejects)),
                endpoint_ordinary_member_CE=history[-1]['ordinary_member_CE'] if history else None,
                original_class_mass=mass, fit_sha256=sha(folder / 'fit.json')))
    official = pd.read_parquet(ROOT / 'data/official/train.parquet', columns=['event_id', 'label_binary'])
    rows = pd.read_parquet(ROOT / 'artifacts/v75_four_arm_20260921_r2/rows.parquet', columns=['row_position', 'event_id', 'label_index', 'route'])
    folds = pd.read_parquet(ROOT / 'artifacts/v106_frozen_audit_20260928/proposed_body_closed_folds.parquet', columns=['row_position', 'root', 'proposed_fold'])
    y = official.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy(np.int8)
    assert len(official) == len(rows) == len(folds) == 2056871
    assert np.array_equal(rows.row_position, np.arange(len(y))) and np.array_equal(folds.row_position, rows.row_position)
    assert np.array_equal(official.event_id, rows.event_id) and np.array_equal(y, rows.label_index)
    asa_positions = np.flatnonzero(rows.route.eq('asa'))
    assert np.array_equal(asa_positions, original.row_position) and np.array_equal(y[asa_positions], original.truth)
    assert np.array_equal(folds.root.to_numpy()[asa_positions], original.root)
    assert np.array_equal(folds.proposed_fold.to_numpy()[asa_positions], original.fold)
    full = pd.read_parquet(RUN / 'full_prediction_ledger.parquet')
    prior_full = pd.read_parquet(OLD / 'full_prediction_ledger.parquet')
    asa = pd.read_parquet(RUN / 'ASA_prediction_ledger.parquet')
    assert np.array_equal(full.row_position, rows.row_position) and np.array_equal(asa.row_position, original.row_position)
    assert np.array_equal(asa.truth, original.truth)
    assert np.array_equal(full.pred_A0, prior_full.pred_A0) and np.array_equal(full.pred_V146_A, prior_full.pred_A)
    not_asa = rows.route.ne('asa').to_numpy()
    quality = read(RUN / 'quality.json')
    ladder = pd.read_parquet(ROOT / 'artifacts/v123_targeted_plan_20260929/support_ladder.parquet')
    assert np.array_equal(ladder.row_position, original.row_position)
    headers = pd.read_parquet(ROOT / 'artifacts/v124_header_trial_20260929/header_span_ledger.parquet', columns=['row_position'])
    masks = dict(all_ASA=np.ones(len(original), bool), known_578=original.known_578_cohort.to_numpy(),
                 strict_correct_51=original.same_family_and_outer_fold_control_S.to_numpy())
    results, slices, pair_results, absolute_gates = {}, {}, {}, {}
    for arm in 'AB':
        pred = full['pred_' + arm].to_numpy()
        assert np.isin(pred, [0, 1, 2]).all()
        assert np.array_equal(pred[asa_positions], qouter[arm].argmax(1))
        assert np.array_equal(pred[asa_positions], asa['pred_' + arm])
        assert np.array_equal(pred[not_asa], prior_full.pred_A.to_numpy()[not_asa])
        assert int((pred[not_asa] != y[not_asa]).sum()) == 107
        train = pd.concat(roles[arm], ignore_index=True)
        saved = pd.read_parquet(RUN / f'{arm}_training_role_ledger.parquet')
        assert train.equals(saved)
        assert len(train) == 225614 and train.row_position.nunique() == 112807
        results[arm] = dict(full=classes(y, pred), ASA=classes(y[asa_positions], pred[asa_positions]),
                           TRAIN=classes(train.truth.to_numpy(), train.pred.to_numpy()),
                           TRAIN_pure_role_rows=int(train.pure_TRAIN_input.sum()),
                           TRAIN_initial_correct_role_rows=int(train.pred_start.eq(train.truth).sum()))
        assert results[arm]['full'] == quality[arm]['full_task']['B']
        assert results[arm]['ASA'] == quality[arm]['ASA']['B']
        assert results[arm]['TRAIN_initial_correct_role_rows'] == 225558
        slices[arm] = {name: classes(original.truth.to_numpy()[mask], pred[asa_positions][mask]) for name, mask in masks.items()}
        yp, ap, bp = original.truth.to_numpy(), full.pred_A0.to_numpy()[asa_positions], pred[asa_positions]
        base_full = classes(y, full.pred_A0.to_numpy())
        base_asa = classes(yp, ap)
        cc, cf = results[arm]['ASA'], results[arm]['full']
        oo = (yp == 2) & ~original.root.isin([21702, 20849, 29]).to_numpy()
        bysource = pd.DataFrame(dict(root=original.root.to_numpy()[yp == 2], correct=bp[yp == 2] == 2)).groupby('root').correct.agg(['mean', 'sum'])
        sources = dict(roots=len(bysource), mean_recall=float(bysource['mean'].mean()), zero_recall_roots=int(bysource['sum'].eq(0).sum()))
        assert sources == quality[arm]['S_sources']
        unk = ladder.diagnostic_bucket.eq('unknown_parameter_pooled_support').to_numpy() & (yp == 2)
        head = original.row_position.isin(headers.row_position).to_numpy()
        r2868 = original.root.eq(2868).to_numpy() & (yp == 1)
        gates = dict(complete_full_population=len(y) == 2056871,
            all_required_classes_present=all(base_full[str(c)]['support'] > 0 for c in range(3)),
            M_S_present_in_ASA=all(base_asa[str(c)]['support'] > 0 for c in [1, 2]),
            ASA_M_absolute=cc['1']['missed'] <= 318, ASA_S_absolute=cc['2']['missed'] <= 2074,
            ASA_total_absolute=int((bp != yp).sum()) <= 2170,
            paired_M_S_protected=all(cc[str(c)]['missed'] <= base_asa[str(c)]['missed'] for c in [1, 2]),
            multiple_folds_improve=sum(int((bp[m] != yp[m]).sum()) < int((ap[m] != yp[m]).sum()) for f in range(3) for m in [original.fold.eq(f).to_numpy()]) >= 2,
            outside_largest_S_groups_protected=int((bp[oo] == 2).sum()) >= int((ap[oo] == 2).sum()),
            full_class_recall_F1_protected=all(cf[str(c)][k] >= base_full[str(c)][k] for c in range(3) for k in ['recall', 'f1']),
            full_precision_recall_F1_protected=all(cf[str(c)][k] >= base_full[str(c)][k] for c in range(3) for k in ['precision', 'recall', 'f1']),
            S_source_mean_recall=sources['mean_recall'] >= .1151252713,
            S_zero_recall_roots=sources['zero_recall_roots'] <= 198,
            S_outside_top3=int((bp[oo] != 2).sum()) < 1373,
            unknown186_S=int((bp[unk] != 2).sum()) <= 184,
            header682=int((bp[head] != yp[head]).sum()) == 0,
            root2868_M=int((bp[r2868] != 1).sum()) <= 48,
            non_ASA_frozen_errors=int((pred[not_asa] != y[not_asa]).sum()) == 107)
        assert gates == quality[arm]['gates']
        assert all(gates.values()) == quality[arm]['quality_passed']
        absolute_gates[arm] = gates
    yy = original.truth.to_numpy()
    a, b = full.pred_A.to_numpy()[asa_positions], full.pred_B.to_numpy()[asa_positions]
    for name, before in [('B_vs_A', a), ('B_vs_V146_A', full.pred_V146_A.to_numpy()[asa_positions]), ('B_vs_A0', full.pred_A0.to_numpy()[asa_positions])]:
        pair_results[name] = changes(yy, before, b)
        assert pair_results[name] == delivery['original_pairs'][name]
    fold_results = [dict(fold=int(f), A_errors=int((a[mask] != yy[mask]).sum()), B_errors=int((b[mask] != yy[mask]).sum()))
                    for f in range(3) for mask in [original.fold.eq(f).to_numpy()]]
    outside = (yy == 2) & ~original.root.isin([21702, 20849, 29]).to_numpy()
    ca, cb = results['A']['ASA'], results['B']['ASA']
    matched = dict(M_no_increase=cb['1']['missed'] <= ca['1']['missed'], S_no_increase=cb['2']['missed'] <= ca['2']['missed'],
                   one_class_improves=any(cb[str(c)]['missed'] < ca[str(c)]['missed'] for c in [1, 2]),
                   two_folds_improve=sum(z['B_errors'] < z['A_errors'] for z in fold_results) >= 2,
                   outside_top3_S_no_increase=int((b[outside] != 2).sum()) <= int((a[outside] != 2).sum()))
    assert matched == quality['matched_gates']
    assert fold_results == quality['matched']['folds']
    assert sum(z['gradients'] for z in summaries) == delivery['full_gradient_evaluations'] <= 1800
    assert sum(z['proposals'] for z in summaries) == delivery['proposal_evaluations'] <= 3600
    assert sum(z['updates'] for z in summaries) == delivery['accepted_updates'] <= 1200
    changed = original[['row_position', 'root', 'fold', 'truth', 'canonical_key', 'known_578_cohort']].copy()
    for name in ['pred_A0', 'pred_V146_A', 'pred_A', 'pred_B']:
        changed[name] = full[name].to_numpy()[asa_positions]
    changed = changed[changed.pred_A.ne(changed.pred_B) | changed.pred_B.ne(changed.pred_V146_A)]
    result = dict(status='complete_actual_original_rows_and_endpoint_receipts_independently_recounted',
        physical_sealed_files=len(seal['source_sha256']), official_rows=len(y), ASA_rows=len(original),
        latest_actual='V155', own_model_forwards=0, own_gradients=0, own_fits=0, own_updates=0,
        fit_summaries=summaries, metrics=results, fixed_cohorts=slices, paired_changes=pair_results,
        folds=fold_results, matched_gates=matched, independent_absolute_quality_gates=absolute_gates,
        matched_effect_reported=delivery['matched_effect_passed'],
        full_quality_acceptance_reported=delivery['quality_acceptance'], model_promoted=delivery['model_promoted'],
        limits=['Saved ensemble probabilities and counts audited; actual model replay belongs to executor.',
                'No HELD calibration, new point selection, independent support, or external test created.'],
        source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [Path(__file__), RUN/'final_delivery.json', RUN/'quality.json', RUN/'verification.json', REF/'all_original_classifier_gap_and_control_ledger.parquet']})
    OUT.mkdir()
    changed.to_parquet(OUT / 'all_changed_original_ASA_rows.parquet', index=False)
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], ASA={a: results[a]['ASA'] for a in 'AB'},
        fixed_cohorts=slices, matched_gates=matched, acceptance=delivery['quality_acceptance']), ensure_ascii=False))


if __name__ == '__main__':
    main()
