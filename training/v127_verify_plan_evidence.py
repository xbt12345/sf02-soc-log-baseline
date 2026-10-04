"""Replay V127 review evidence. No fit, optimizer, training seal or deployment."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from v126_frozen_audit import ROOT, PARENT, read, sha, save
from v125_evaluate import TRACE, OFFICIAL
from v124_header import HEADER, transform, old_text

OUT = ROOT / 'artifacts/v127_plan_review_20260929'
PLAN = ROOT / 'training/review_policy/v127_next_training_plan.json'
REPORT = ROOT / 'docs/V127_EVIDENCE_BASED_PLAN_REFINEMENT.md'
RISKS = ROOT / 'training/review_policy/v127_risk_actions.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def review_plan(p):
    require(p['status'] == 'design_only_runtime_not_implemented_or_sealed', 'Runtime status inflated')
    require(p['latest_actual_training'] == 'V125' and p['classifier_fits_this_review'] == 0
            and p['optimizer_steps_this_review'] == 0 and not p['model_promoted'], 'Review is not training')
    require(set(p['arms']) == {'A0', 'AH', 'K', 'W', 'P'}, 'Missing or unregistered arm')
    b, t, q, k, d = p['base'], p['training'], p['primary_quality'], p['constant_control'], p['data']
    require(b['frozen_parameters'] and b['frozen_buffers'] and b['eval_mode']
            and b['shared_exact_base_across_arms'] and not b['optimizer_contains_base'], 'Base is not frozen')
    require('per-member' in b['cache'] and 'mean-member' in t['loss'] and 'AH canonical' in b['cache'],
            'Wrong base input or member objective')
    require(d['common_header_projection']['canonical_real_and_sanitized_clock_variants_equal']
            and d['common_header_projection']['raw_body_bytes_or_fact_values_changed'] == 0,
            'Header projection does not preserve the evidence contract')
    require(t['epochs'] == 50 and t['selector'].startswith('fixed epoch50')
            and t['seed'] == 12701, 'Unregistered endpoint or seed')
    require(not any(t[x] for x in ['class_or_group_reweighting', 'base_unfreezing', 'adaptive_outer_selection',
                                  'threshold_tuning', 'candidate_substitution', 'automatic_epoch_extension']),
            'Unregistered adaptation')
    require(q['candidate'] == 'P' and 'sole' in p['arms']['P']['role']
            and p['arms']['K']['role'].startswith('nonpromotable')
            and p['arms']['W']['role'].startswith('nonpromotable'), 'Control promoted')
    require(len(t['fit_order']) == 9 and set(map(tuple, t['fit_order'])) == {(f, a) for f in range(3) for a in 'KWP'},
            'Missing fold/control')
    require((t['primary_fits'], t['primary_network_fits'], t['primary_constant_offset_fits']) == (9, 6, 3),
            'Fits miscounted')
    require(t['expected_steps_per_arm'] == [3650, 1600, 3650] and t['expected_primary_steps'] == 17800,
            'Update budget differs')
    require((q['ASA_M_errors_max'], q['ASA_S_errors_max'], q['ASA_total_errors_max']) == (318, 2074, 2170)
            and q['hard_reference'].startswith('A0') and q['projection_cost_included'], 'Weakened baseline')
    require(all(q[x] for x in ['converged_K_required', 'ASA_total_errors_strictly_less_than_K',
                              'full_task_each_class_recall_precision_F1_no_regression',
                              'zero_prediction_flips_registered_harmless_header_variants',
                              'at_least_two_improving_folds', 'outside_top3_S_roots_no_regression',
                              'S_group_macro_no_regression', 'S_zero_recall_groups_no_increase']),
            'Incomplete quality protection')
    require(d['full_scoring_rows'] == 2056871 and d['ASA_original_rows'] == 112807
            and d['labels_unchanged'] and d['original_row_frequency_unchanged']
            and d['all_legally_available_fit_rows'] and not d['unknown_parameter_imputation'], 'Rows or truth changed')
    require(k['parameterization'] == 'b=(u,v,-u-v)' and k['l2_coefficient'] == 1e-6
            and k['dtype'] == 'float64' and k['independent_gradient_inf_max'] == 1e-7
            and k['finite_solution_required'] and not k['heldout_labels_or_calibration_set_used'],
            'Constant control lacks its finite stationary solution contract')
    sh = p['diagnostic_interventions']['permuted_residual']
    require(sh['unit'].startswith('individual original row') and not sh['uses_labels']
            and sh['preserves_original_row_marginal_residual_distribution'], 'Shuffle changes original-row marginal')
    require(not p['branch']['sequence_order_benefit_claim_allowed']
            and not t['learning_rate_causality_claim_allowed'], 'Unsupported causal claim')
    require(p['confirmation']['only_after_all_primary_quality_and_integrity_gates']
            and p['confirmation']['seeds'] == [12702, 12703], 'Confirmation gate changed')


def synthetic_constant_gradient():
    """No optimization. Check the exact proposed objective's chain rule numerically."""
    rng = np.random.default_rng(12701)
    z = rng.normal(size=(7, 3, 3))
    y = np.array([1, 2, 1, 2, 2, 1, 1])
    mass = np.array([1, 5, 2, 1, 8, 3, 1], dtype=float)
    lam = 1e-6
    def loss_grad(uv):
        b = np.array([uv[0], uv[1], -uv.sum()])
        v = z + b
        lse = logsumexp(v, axis=-1)
        loss = np.dot(mass, (lse - v[np.arange(len(y)), :, y]).mean(1)) / mass.sum() + lam / 2 * np.dot(b, b)
        prob = np.exp(v - lse[:, :, None])
        prob[np.arange(len(y)), :, y] -= 1
        gb = (prob.mean(1) * mass[:, None]).sum(0) / mass.sum() + lam * b
        return float(loss), np.array([gb[0] - gb[2], gb[1] - gb[2]])
    uv = np.array([.24, -.31]); step = 1e-5
    analytic = loss_grad(uv)[1]
    fd = np.array([(loss_grad(uv + step * e)[0] - loss_grad(uv - step * e)[0]) / (2 * step)
                   for e in np.eye(2)])
    error = float(np.max(np.abs(analytic - fd)))
    require(error < 1e-8, 'Synthetic K gradient incorrect')
    # Unequal duplicate masses: row shuffle preserves the marginal, local shuffle need not.
    row_values = np.array([1, 1, 1, 5]); permuted = row_values[rng.permutation(4)]
    require(np.array_equal(np.sort(row_values), np.sort(permuted)), 'Row permutation loses marginal')
    require(row_values.mean() != np.array([5, 5, 5, 1]).mean(), 'Counterexample is ineffective')
    return {'synthetic_gradient_max_abs_error': error, 'unequal_duplicate_shuffle_counterexample_passed': True,
            'scope': 'Synthetic formula and permutation checks only; no solver run or SOC fit'}


def main(record=False):
    receipt = OUT / 'verification.json'
    if record and receipt.exists():
        raise FileExistsError(receipt)
    parent = read(PARENT / 'delivery.json')
    bound = {**parent['artifact_sha256'], **parent['source_sha256'],
             parent['report']: parent['report_sha256'], parent['risk_addendum']: parent['risk_addendum_sha256']}
    prior_path = ROOT / 'artifacts/v126_frozen_review_20260929/verification.json'
    prior = read(prior_path)
    for rel, h in {**bound, **prior['artifact_sha256']}.items():
        require(sha(ROOT / rel) == h, 'Historical file changed: ' + rel)
    hp = ROOT / 'artifacts/v124_header_trial_20260929'
    header_preflight = read(hp / 'input_preflight.json')
    for name, h in header_preflight['output_sha256'].items():
        require(sha(hp / name) == h, 'Header input changed: ' + name)
    for rel, h in header_preflight['source_sha256'].items():
        require(sha(ROOT / rel) == h, 'Header source changed: ' + rel)

    d = pd.read_parquet(TRACE, columns=['row_position', 'local', 'fold', 'truth', 'raw_message'])
    saved = pd.read_parquet(PARENT / 'expert_ASA_predictions.parquet')
    require(len(d) == 112807 and d.row_position.equals(saved.row_position)
            and not d.row_position.duplicated().any(), 'Wrong original-row population')
    labels = pd.read_parquet(OFFICIAL, columns=['label_binary']).label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    y, local, folds = d.truth.to_numpy(), d.local.to_numpy(), d.fold.to_numpy()
    require(len(labels) == 2056871 and np.array_equal(labels[d.row_position], y), 'Official labels differ')
    mechanism, header = read(OUT / 'mechanism_probe.json'), read(OUT / 'header_projection.json')
    require(mechanism['source_sha256'] == sha(ROOT / 'training/v127_mechanism_probe.py')
            and header['source_sha256'] == sha(ROOT / 'training/v127_header_projection_probe.py')
            and mechanism['parent_review_sha256'] == sha(prior_path), 'Diagnostic provenance changed')
    require(mechanism['panel_sha256'] == sha(OUT / 'attention_panel.parquet'), 'Attention panel changed')
    panel = pd.read_parquet(OUT / 'attention_panel.parquet')
    require(not panel.duplicated(['fold', 'class_panel', 'local']).any(), 'Duplicate panel unit')
    for f in range(3):
        for c in (1, 2):
            available = np.unique(local[(folds != f) & (y == c)])
            chosen = np.sort(np.random.default_rng(12701 + 100 * f + c).choice(available, min(256, len(available)), replace=False))
            actual = panel.loc[(panel.fold == f) & (panel.class_panel == c), 'local'].to_numpy()
            require(np.array_equal(chosen, actual), 'Panel is not the registered train-only selection')
    attention = mechanism['attention']
    require(len(attention) == 18 and {(r['fold'], r['arm'], r['epoch']) for r in attention}
            == {(f, a, e) for f in range(3) for a in 'BC' for e in (0, 1, 25)}, 'Missing attention state')
    require(max(r['forward_max_abs_difference'] for r in attention) <= 2e-5, 'Probe forward not equivalent')
    fold1_heads = next(r for r in attention if (r['fold'], r['arm'], r['epoch']) == (1, 'B', 25))['layers'][1]['per_head_mean_entropy']
    require(min(fold1_heads) > .99, 'Observed counterexample changed; investigate, not a general entropy gate')

    changed = 0
    for raw in d.raw_message:
        m = HEADER.match(raw)
        require(m is not None, 'Unknown header')
        body = raw[m.end():]; text = transform(raw)[0]
        require(text == old_text('<164>Jan 01 2000 01:02:03: ' + body)
                and text == transform('<164>Dec 31 2001 USER-CRED-45: ' + body)[0], 'Variant projection differs')
        changed += text != old_text(raw)
    require(changed == header['changed_raw_inputs'] == 682, 'Projection population changed')
    original, canonical = np.empty(len(d), dtype=int), np.empty(len(d), dtype=int)
    gradient_summary = []
    for f in range(3):
        a = np.load(PARENT / f'fold{f}_A/epoch25_prob.npy')
        h = np.load(OUT / f'fold{f}_A_header_probability.npy')
        require(a.shape == h.shape == (22546, 3) and np.isfinite(h).all()
                and np.allclose(h.sum(1), 1, atol=1e-6), 'Invalid probabilities')
        held = folds == f
        original[held], canonical[held] = a[local[held]].argmax(1), h[local[held]].argmax(1)
        row = {'fold': f}
        for name, prob in [('A0', a), ('AH', h)]:
            masses = [float((2 * (1 - prob[local[(~held) & (y == c)], c].astype(float))).sum()) for c in (1, 2)]
            row[name + '_S_gradient_l1_mass_fraction'] = masses[1] / sum(masses)
            if name == 'A0':
                reported = mechanism['zero_residual_gradient'][f]
                require(np.allclose(masses, [r['gradient_l1_mass'] for r in reported['classes']], rtol=1e-12), 'Gradient mass differs')
        gradient_summary.append(row)
        per_fold = header['by_fold'][f]
        require(int((original[held] != y[held]).sum()) == per_fold['original_errors']
                and int((canonical[held] != y[held]).sum()) == per_fold['canonical_errors'], 'Fold errors differ')
    require(np.array_equal(original, saved.expert_pred_A.to_numpy()), 'Original baseline decision changed')
    for c in (1, 2):
        mask = y == c
        counts = {'rows': int(mask.sum()), 'original_errors': int((original[mask] != c).sum()),
                  'canonical_errors': int((canonical[mask] != c).sum()),
                  'repaired': int((mask & (original != c) & (canonical == c)).sum()),
                  'regressed': int((mask & (original == c) & (canonical != c)).sum())}
        require(counts == header['per_class'][str(c)], 'Projection transitions differ')
    require(header['per_class']['2']['regressed'] == 40 and header['per_class']['1']['regressed'] == 0,
            'Projection cost changed')

    p = read(PLAN); review_plan(p)
    mutations = [
        ('missing_K', lambda v: v['arms'].pop('K')),
        ('trainable_base', lambda v: v['base'].update(optimizer_contains_base=True)),
        ('control_promotion', lambda v: v['primary_quality'].update(candidate='W')),
        ('weaker_AH_quality_reference', lambda v: v['primary_quality'].update(hard_reference='AH')),
        ('ignore_projection_cost', lambda v: v['primary_quality'].update(projection_cost_included=False)),
        ('normal_missing_no_finite_K', lambda v: v['constant_control'].update(l2_coefficient=0)),
        ('unverified_K_stationarity', lambda v: v['primary_quality'].update(converged_K_required=False)),
        ('unique_local_shuffle', lambda v: v['diagnostic_interventions']['permuted_residual'].update(unit='unique local')),
        ('label_based_shuffle', lambda v: v['diagnostic_interventions']['permuted_residual'].update(uses_labels=True)),
        ('hidden_reweight', lambda v: v['training'].update(class_or_group_reweighting=True)),
        ('early_checkpoint_selection', lambda v: v['training'].update(epochs=25)),
        ('hide_member_loss', lambda v: v['base'].update(cache='AH canonical mean probability')),
        ('omit_header_gate', lambda v: v['primary_quality'].update(zero_prediction_flips_registered_harmless_header_variants=False)),
    ]
    rejected = []
    for name, mutate in mutations:
        bad = copy.deepcopy(p); mutate(bad)
        try:
            review_plan(bad)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError('Invalid plan accepted: ' + name)
    for case in read(RISKS)['cases']:
        require((ROOT / case['evidence']).is_file(), 'Missing risk evidence')
    synthetic = synthetic_constant_gradient()
    files = [f for f in OUT.iterdir() if f.is_file() and f.name != receipt.name]
    files += [PLAN, REPORT, RISKS, Path(__file__), ROOT / 'training/v127_mechanism_probe.py',
              ROOT / 'training/v127_header_projection_probe.py', prior_path, hp / 'input_preflight.json',
              PARENT / 'delivery.json']
    hashes = {f.relative_to(ROOT).as_posix(): sha(f) for f in files}
    result = {'status': 'V127_plan_evidence_verified_not_training_runtime', 'latest_actual_training': 'V125',
              'classifier_fits': 0, 'optimizer_steps': 0, 'quality_acceptance': False, 'model_promoted': False,
              'training_runtime_implemented': False, 'historical_delivery_files_verified': len(bound),
              'parent_review_files_verified': len(prior['artifact_sha256']), 'official_rows': len(labels),
              'ASA_original_rows_recounted': len(d), 'header_variants_rechecked': len(d),
              'attention_saved_states_checked': len(attention), 'fold1_B_final_layer2_entropy': fold1_heads,
              'gradient_mass_recomputed': gradient_summary, 'projection_transition_counts': header['per_class'],
              'negative_plan_cases_rejected': rejected, **synthetic, 'artifact_sha256': hashes,
              'limits': 'Attention aggregate provenance and panel checked, not a second GPU attention replay. Header model inference was executed by the probe; this verifier independently recounts saved probabilities and official truth. Plan validation is not the unimplemented training runtime seal; no independent blind test.'}
    if receipt.exists():
        require(read(receipt) == result, 'Review receipt no longer matches')
    if record:
        save(receipt, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--record', action='store_true')
    main(parser.parse_args().record)
