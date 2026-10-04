"""Read-only audit of v69 saved HGB representations. Never calls fit.

Only --out writes this audit's new JSON; prior model/data files are immutable.
Empirical collision bounds concern this representation on observed rows, not
the Bayes error or attainable performance on future subjects.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from run_v69_support_control import ARMS, load
from run_v67_targeted import design

RUN = ROOT / 'artifacts/v69_support_control_20260921'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(values):
    x = np.array(values, dtype=np.float64, copy=True)
    # All NaNs express the same missing state. Normalize signed zero too.
    x[np.isnan(x)] = np.inf
    x[x == 0] = 0
    return np.unique(x, axis=0, return_inverse=True)[1]


def collision_summary(key, y, source):
    m = np.bincount(key, weights=y == 1)
    s = np.bincount(key, weights=y == 2, minlength=len(m))
    m = np.pad(m, (0, max(0, len(s) - len(m))))
    mixed = (m > 0) & (s > 0)
    mask = mixed[key]
    return {
        'unique_full_views': int(np.unique(key).size),
        'mixed_full_views': int(mixed.sum()),
        'mixed_rows': int(mask.sum()),
        'M_rows_in_mixed_views': int((mask & (y == 1)).sum()),
        'S_rows_in_mixed_views': int((mask & (y == 2)).sum()),
        'S_sources_in_mixed_views': int(np.unique(source[mask & (y == 2)]).size),
        'minimum_empirical_row_errors': int(np.minimum(m, s).sum()),
    }, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.out:
        assert not args.out.exists(), 'Do not overwrite audit evidence'
    f, context, manifest = load()
    allpos = pd.Index(f.row_position)
    target = set(manifest.loc[manifest.target_578, 'row_position'])
    bindings = {str(p.relative_to(ROOT)): digest(p) for p in [
        RUN / 'preregistered.json', RUN / 'target_actual_model_inputs.parquet',
        ROOT / 'artifacts/v61_source_factorial_20260914_r2/records.parquet',
        ROOT / 'artifacts/v61_source_factorial_20260914_r2/context.npz',
        ROOT / 'training/run_v67_targeted.py',
        ROOT / 'training/run_v69_support_control.py',
    ]}
    result = {
        'scope': 'Saved-model inference and representation audit only; no training or selection',
        'new_fits': 0, 'sklearn_version': sklearn.__version__,
        'limitations': [
            'Empirical minimum row errors are representation-specific sample bounds, not causal label-noise claims or future error floors.',
            'A numeric feature losing distinct values is not itself proof of a new full-input M/S collision.',
            'Bin collisions demonstrate representational loss; correcting it need not improve held-out classification.',
            'Leaf limits and training loss do not establish that larger trees will generalize.',
            'All targets and evaluation sources are previously inspected development data.',
            'Source groups are anonymized symbols and do not prove distinct physical entities.',
        ],
        'input_sha256': bindings, 'models': [],
    }
    with threadpool_limits(limits=4):
        for fold in range(3):
            roles = pd.read_parquet(RUN / f'fold{fold}_roles.parquet')
            assert np.array_equal(roles.row_position, f.row_position)
            for arm in ARMS:
                folder = RUN / f'fold{fold}' / arm
                path = folder / 'model.joblib'
                before = digest(path)
                bundle = joblib.load(path)
                model = bundle['model']
                fit = roles[arm].to_numpy()
                ix = np.flatnonzero(fit)
                assert np.array_equal(bundle['fit_positions'], f.row_position[fit])
                x, _, names, categorical = design(f, context, 'context_numeric', ix, encoder=bundle['encoder'])
                assert names == bundle['names']
                # Follow the actual 1.6.1 predict path, including categorical
                # re-encoding/reordering, before applying the training bin map.
                xp = model._preprocess_X(x, reset=False)
                remapped_names = [n for n, cat in zip(names, categorical) if cat]
                remapped_names += [n for n, cat in zip(names, categorical) if not cat]
                assert np.array_equal(model._is_categorical_remapped,
                                      np.r_[np.ones(sum(categorical), bool), np.zeros(len(names)-sum(categorical), bool)])
                xb = model._bin_mapper.transform(xp)
                role_replays = {}
                for role, mask in [('fit', fit), ('calibration', roles.calibration.to_numpy()), ('evaluation', roles.evaluation.to_numpy())]:
                    saved = pd.read_parquet(folder / (role + '.parquet'))
                    assert np.array_equal(saved.row_position, f.row_position[mask])
                    ordinary = model.predict_proba(x[mask])[:, 1]
                    raw = np.zeros((int(mask.sum()), model.n_trees_per_iteration_), dtype=model._baseline_prediction.dtype, order='F')
                    raw += model._baseline_prediction
                    model._predict_iterations(np.ascontiguousarray(xb[mask]), model._predictors, raw, True, 4)
                    binned_probability = expit(raw[:, 0])
                    np.testing.assert_array_equal(ordinary, saved.score.to_numpy())
                    np.testing.assert_array_equal(binned_probability, ordinary)
                    role_replays[role] = {'rows': int(mask.sum()), 'ordinary_vs_saved_max_abs': 0., 'binned_vs_ordinary_max_abs': 0.}
                # Use exact equality grouping, not hashes, for full-view bounds.
                fit_y = f.label.to_numpy()[fit]
                fit_g = f.group.to_numpy()[fit]
                exact_key = ids(x[fit]); pre_key = ids(xp[fit]); bin_key = ids(xb[fit])
                exact, exact_mixed = collision_summary(exact_key, fit_y, fit_g)
                pre, pre_mixed = collision_summary(pre_key, fit_y, fit_g)
                bins, bin_mixed = collision_summary(bin_key, fit_y, fit_g)
                assert exact == pre
                assert np.array_equal(exact_mixed, pre_mixed)
                assert bins['minimum_empirical_row_errors'] >= exact['minimum_empirical_row_errors']
                new = bin_mixed & ~exact_mixed
                newly_pure_groups = np.unique(bin_key[bin_mixed])
                contain_old_mixed = set(bin_key[exact_mixed])
                new_mixed_group_count = sum(int(k not in contain_old_mixed) for k in newly_pure_groups)
                feature_rows = []
                for j, name in enumerate(remapped_names):
                    v = xp[fit, j]; b = xb[fit, j]
                    observed = ~np.isnan(v)
                    pairs = np.unique(np.c_[v[observed], b[observed]], axis=0)
                    sizes = pd.Series(pairs[:, 1]).value_counts()
                    mixed_bins = set(sizes[sizes > 1].index)
                    feature_rows.append({
                        'name': name, 'is_categorical': bool(model._is_categorical_remapped[j]),
                        'distinct_nonmissing_values': int(np.unique(v[observed]).size),
                        'distinct_nonmissing_bins': int(np.unique(b[observed]).size),
                        'bins_merging_distinct_values': int((sizes > 1).sum()),
                        'maximum_distinct_values_per_bin': int(sizes.max()) if len(sizes) else 0,
                        'training_rows_in_merged_nonmissing_bins': int(np.isin(b, list(mixed_bins)).sum()),
                        'missing_fit_rows': int(np.isnan(v).sum()),
                    })
                targets = roles.evaluation.to_numpy() & f.row_position.isin(target).to_numpy()
                # Group joint fit + target matrices; target labels never enter fit.
                joint = fit | targets
                jfit = fit[joint]; jy = f.label.to_numpy()[joint]
                supports = {}
                for label, view in [('exact', x), ('binned', xb)]:
                    key = ids(view[joint]); same_m = set(key[jfit & (jy == 1)]); same_s = set(key[jfit & (jy == 2)])
                    tkeys = key[~jfit]
                    supports[label] = {'M': np.isin(tkeys, list(same_m)), 'S': np.isin(tkeys, list(same_s))}
                tg = f.group.to_numpy()[targets]
                tpos = f.row_position.to_numpy()[targets]
                new_m = supports['binned']['M'] & ~supports['exact']['M']
                cats = [n for n, cat in zip(names, categorical) if cat]
                category_oov = {}
                for name, vocab in zip(cats, bundle['encoder'].categories_):
                    unknown = ~f[name].isin(vocab).to_numpy()
                    category_oov[name] = {'fit_rows': int((fit & unknown).sum()),
                                          'evaluation_rows': int((roles.evaluation.to_numpy() & unknown).sum()),
                                          'target_rows': int((targets & unknown).sum())}
                predictors = [tree for step in model._predictors for tree in step]
                leaves = np.array([int(tree.nodes['is_leaf'].sum()) for tree in predictors])
                leaf_counts = np.concatenate([tree.nodes['count'][tree.nodes['is_leaf'].astype(bool)] for tree in predictors])
                split_counts = np.zeros(len(names), int)
                for tree in predictors:
                    for j in tree.nodes['feature_idx'][~tree.nodes['is_leaf'].astype(bool)]: split_counts[j] += 1
                fd = pd.read_parquet(folder / 'fit.parquet')
                score_key = ids(fd[['score']].to_numpy())
                scores, _ = collision_summary(score_key, fit_y, fit_g)
                pred = np.where(fd.score.to_numpy() >= .5, 2, 1)
                entry = {
                    'fold': fold, 'arm': arm, 'model_sha256': before,
                    'preprocessing': 'model._preprocess_X(reset=False), then saved model._bin_mapper.transform',
                    'predict_path_replay': role_replays,
                    'fit_rows': int(fit.sum()), 'fit_S_rows': int((fit_y == 2).sum()),
                    'exact_design': exact, 'post_preprocessor': pre, 'post_binning': bins,
                    'binning_new_mixed_groups_without_preexisting_mixed_subgroup': new_mixed_group_count,
                    'binning_added_minimum_empirical_errors': bins['minimum_empirical_row_errors'] - exact['minimum_empirical_row_errors'],
                    'previously_pure_exact_S_rows_now_mixed': int((new & (fit_y == 2)).sum()),
                    'previously_pure_exact_S_sources_now_mixed': int(np.unique(fit_g[new & (fit_y == 2)]).size),
                    'feature_binning': feature_rows, 'category_OOV': category_oov,
                    'target': {'rows': int(targets.sum()),
                        'same_exact_fit_M_rows': int(supports['exact']['M'].sum()),
                        'same_binned_fit_M_rows': int(supports['binned']['M'].sum()),
                        'new_same_fit_M_rows_due_to_binning': int(new_m.sum()),
                        'new_same_fit_M_sources_due_to_binning': int(np.unique(tg[new_m]).size),
                        'new_same_fit_M_row_positions': tpos[new_m].tolist(),
                        'same_exact_fit_S_rows': int(supports['exact']['S'].sum()),
                        'same_binned_fit_S_rows': int(supports['binned']['S'].sum())},
                    'capacity': {'iterations': int(model.n_iter_), 'max_bins': int(model.max_bins),
                        'max_leaf_nodes': int(model.max_leaf_nodes), 'min_samples_leaf': int(model.min_samples_leaf),
                        'leaf_count_min_median_max': [int(leaves.min()), float(np.median(leaves)), int(leaves.max())],
                        'trees_reaching_max_leaf_nodes': int((leaves == model.max_leaf_nodes).sum()),
                        'minimum_observed_training_leaf_support': int(leaf_counts.min()),
                        'split_counts_by_feature': dict(zip(remapped_names, split_counts.tolist())),
                        'saved_score_empirical_collisions': scores,
                        'default_threshold_fit_errors': int((pred != fit_y).sum()),
                        'default_threshold_fit_M_errors': int(((fit_y == 1) & (pred != 1)).sum()),
                        'default_threshold_fit_S_errors': int(((fit_y == 2) & (pred != 2)).sum())},
                }
                assert digest(path) == before
                result['models'].append(entry)
                print(f"AUDITED fold={fold} arm={arm}: exact_min={exact['minimum_empirical_row_errors']}, binned_min={bins['minimum_empirical_row_errors']}, new_target_M={int(new_m.sum())}", file=sys.stderr, flush=True)
    result['models_audited'] = len(result['models'])
    result['prediction_rows_replayed'] = sum(v['rows'] for e in result['models'] for v in e['predict_path_replay'].values())
    result['audit_script_sha256'] = digest(__file__)
    for name, value in bindings.items():
        assert digest(ROOT / name) == value
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
