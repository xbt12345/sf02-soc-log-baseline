"""Read-only v65 fixed-terminal prediction audit against v69 B.

No model loading, fit, current-calibration thresholding, checkpoint selection,
or deployable combination. Evaluation-answer oracle cutoffs are diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
V65 = ROOT / 'artifacts/v65_rank_heads_20260920'
V69 = ROOT / 'artifacts/v69_support_control_20260921'
DATA = ROOT / 'artifacts/v61_source_factorial_20260914_r2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def source_weights(d):
    size = d.groupby(['label', 'group']).label.transform('size')
    sources = d.groupby('label').group.nunique()
    return .5 / size.to_numpy() / d.label.map(sources).to_numpy()


def oracle_cut(d, scores, budget=.01):
    """Lowest whole-tie cutoff meeting observed M row/source errors <= budget."""
    mask = d.label.eq(1).to_numpy()
    m = d.loc[mask]; s = scores[mask]
    counts = m.groupby('group').size()
    limits = []
    for w in [np.ones(len(m)) / len(m), 1 / len(counts) / m.group.map(counts).to_numpy()]:
        blocks = pd.DataFrame({'score': s, 'weight': w}).groupby('score').weight.sum().sort_index(ascending=False)
        crossing = np.flatnonzero(blocks.cumsum().to_numpy() > budget + 1e-12)
        limits.append(float(np.nextafter(blocks.index[crossing[0]], np.inf)) if len(crossing) else 0.)
    return max(limits)


def effects(d, candidate, reference, target):
    y = d.label.to_numpy()
    out = {'target_repairs': int((target & (candidate == 2)).sum())}
    for label, name in [(1, 'M'), (2, 'S')]:
        mask = y == label
        c = candidate[mask] == label; b = reference[mask] == label
        sub = d.loc[mask, ['group']].copy(); sub['candidate'] = c; sub['reference'] = b
        by_source = sub.groupby('group')[['candidate', 'reference']].mean()
        delta = by_source.candidate - by_source.reference
        only = mask & (candidate == label) & (reference != label)
        lost = mask & (candidate != label) & (reference == label)
        out[name] = {'candidate_errors': int((~c).sum()), 'reference_errors': int((~b).sum()),
                     'candidate_only_correct_rows': int(only.sum()), 'reference_only_correct_rows': int(lost.sum()),
                     'candidate_only_correct_sources': int(d.loc[only, 'group'].nunique()),
                     'reference_only_correct_sources': int(d.loc[lost, 'group'].nunique()),
                     'source_recall_candidate': float(by_source.candidate.mean()),
                     'source_recall_reference': float(by_source.reference.mean()),
                     'sources_improved': int((delta > 0).sum()), 'sources_worsened': int((delta < 0).sum()),
                     'source_recall_gain': float(delta.mean()),
                     'gain_without_best_source': float(np.delete(delta.to_numpy(), np.argmax(delta.to_numpy())).mean()),
                     'candidate_only_correct_row_positions': d.loc[only, 'row_position'].astype(int).tolist() if name == 'S' else []}
    return out


def source_rank_detail(d, score):
    """AUC for each S source against the common source-balanced M reference."""
    mask = d.label.eq(1).to_numpy(); m = d.loc[mask]; z = score[mask]
    sizes = m.groupby('group').size(); w = 1 / len(sizes) / m.group.map(sizes).to_numpy()
    order = np.argsort(z); z = z[order]; c = np.r_[0., np.cumsum(w[order])]
    ss = d.loc[~mask, ['row_position', 'group']].copy(); s = score[~mask]
    left = c[np.searchsorted(z, s, side='left')]
    right = c[np.searchsorted(z, s, side='right')]
    ss['AUC_against_source_balanced_M'] = (left + right) / 2
    return ss.groupby('group').AUC_against_source_balanced_M.mean().to_dict()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path); args = ap.parse_args()
    if args.out: assert not args.out.exists(), 'Do not overwrite prior evidence'
    inputs = {}
    def bind(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    delivery = json.loads((ROOT/'evidence/2026-09-20/v65_rank_heads/delivery.json').read_text(encoding='utf-8'))
    def check_v65(path):
        key = path.relative_to(ROOT).as_posix(); digest = sha(path)
        assert delivery['artifact_sha256'][key] == digest, key
        inputs[key] = digest
    reg = json.loads((V65/'preregistered.json').read_text(encoding='utf-8'))
    assert reg['epochs'] == 12
    assert sha(ROOT/'training/train_v65_rank_heads.py') == reg['sources']['train_v65_rank_heads.py']
    bind(ROOT/'training/train_v65_rank_heads.py'); check_v65(V65/'preregistered.json')
    cache = json.loads((V65/'text_cache_receipt.json').read_text(encoding='utf-8'))
    assert cache['labels_used'] is False and cache['encoder_frozen'] is True
    check_v65(V65/'text_cache_receipt.json')
    asset = ROOT/'artifacts/v61_asset_cache/securebert2/download_receipt.json'
    assert sha(asset) == cache['asset_receipt_sha256']; bind(asset)
    ar = json.loads(asset.read_text(encoding='utf-8'))
    for name, digest in ar['sha256'].items():
        path = asset.parent/name; assert sha(path) == digest; bind(path)
    for name, field in [('upstream_text_features.npy', 'features_sha256'), ('unique_text.parquet', 'texts_sha256')]:
        assert sha(V65/name) == cache[field]; check_v65(V65/name)
    manifest = pd.read_parquet(V65/'fold_manifest.parquet'); check_v65(V65/'fold_manifest.parquet')
    original = pd.read_parquet(DATA/'records.parquet').set_index('row_position').loc[manifest.row_position].reset_index()
    original['fold'] = manifest.fold.to_numpy()
    assert np.array_equal(original.label, manifest.label) and np.array_equal(original.group, manifest.group)
    bind(DATA/'records.parquet')
    targets = pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet').set_index('row_position').target_578
    bind(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    original['target_578'] = targets.loc[original.row_position].to_numpy()
    h = original.action.eq('deny') & original.outcome.eq('blocked') & original.src_role.eq('outside') & original.dst_role.eq('dmz') & original.transport_protocol.isin(['tcp', 'udp']) & original.label.isin([1, 2])
    hard = original[h].copy().reset_index(drop=True)
    assert len(hard) == 42542 and hard.label.eq(2).sum() == 652 and hard.target_578.sum() == 578
    b = pd.read_parquet(V69/'B_small_sources_budget0.01_oof.parquet'); bind(V69/'B_small_sources_budget0.01_oof.parquet')
    assert np.array_equal(b.row_position, manifest.row_position)
    assert np.array_equal(b.fold, manifest.fold) and np.array_equal(b.label, manifest.label)
    scores = {'B69': b.hard_score.to_numpy()[h]}; fixed = {'B69': b.pred.to_numpy()[h]}
    provenance = []; role_overlap = []
    for fold in range(3):
        roles = pd.read_parquet(V69/f'fold{fold}_roles.parquet'); bind(V69/f'fold{fold}_roles.parquet')
        expected = original.loc[h & original.fold.eq(fold), 'row_position'].to_numpy()
        assert np.array_equal(expected, roles.loc[roles.evaluation, 'row_position'].to_numpy())
        cal_positions = set(roles.loc[roles.calibration, 'row_position'])
        old_train_positions = set(manifest.loc[manifest.fold.ne(fold), 'row_position'])
        role_overlap.append({'fold': fold, 'current_calibration_rows_seen_in_v65_head_training': len(cal_positions & old_train_positions),
                             'old_train_rows': len(old_train_positions), 'current_calibration_rows': len(cal_positions),
                             'common_evaluation_rows': len(expected)})
    for arm in ['H0', 'H1']:
        path = V65/f'{arm}_last_oof.parquet'; check_v65(path)
        old = pd.read_parquet(path)
        for col in ['row_position', 'label', 'group', 'fold']: assert np.array_equal(old[col], manifest[col])
        for fold in range(3):
            folder = V65/f'fold{fold}'/arm
            curve = json.loads((folder/'curve.json').read_text(encoding='utf-8')); check_v65(folder/'curve.json')
            last = curve[-1]
            assert last['epoch_fraction'] == 12. and last['step'] == max(x['step'] for x in curve)
            pred_path = folder/last['predictions_file']; model_path = folder/last['model_file']
            check_v65(pred_path); check_v65(model_path)
            pred = pd.read_parquet(pred_path); oo = old[old.fold.eq(fold)]
            for col in ['row_position', 'label', 'group', 'p_B', 'p_M', 'p_S']:
                np.testing.assert_array_equal(pred[col].to_numpy(), oo[col].to_numpy())
            provenance.append({'fold': fold, 'arm': arm, 'epoch': 12, 'step': last['step'],
                               'checkpoint_file': str(model_path.relative_to(ROOT)), 'checkpoint_sha256': sha(model_path),
                               'predictions_file': str(pred_path.relative_to(ROOT)), 'predictions_exactly_match_terminal_oof': True})
        # p_S/(p_M+p_S) is a monotone transform of logit_S-logit_M.
        # It does not fit a calibration or select a threshold.
        p = old[['p_B', 'p_M', 'p_S']].to_numpy()
        scores[arm] = (p[:, 2] / (p[:, 1] + p[:, 2]))[h]
        fixed[arm] = p.argmax(1)[h]
    oracle = {name: np.zeros(len(hard), dtype=int) for name in scores}
    cells = []; sources = []
    for fold in range(3):
        for protocol in ['tcp', 'udp']:
            ix = hard.fold.eq(fold).to_numpy() & hard.transport_protocol.eq(protocol).to_numpy()
            d = hard[ix]; w = source_weights(d); y = d.label.eq(2).to_numpy(); detail = {}
            for name, score in scores.items():
                s = score[ix]; cut = oracle_cut(d, s)
                pred = np.where(s >= cut, 2, 1); oracle[name][ix] = pred
                ranked_sources = source_rank_detail(d, s)
                dd = d[['group', 'label']].copy(); dd['ok'] = pred == d.label.to_numpy()
                recalls = dd.groupby(['label', 'group']).ok.mean().groupby('label').mean()
                mrows = (pred[~y] != 1).mean(); msource = 1 - float(recalls.loc[1])
                assert max(mrows, msource) <= .01 + 1e-12
                detail[name] = {'source_AUC': float(roc_auc_score(y, s, sample_weight=w)),
                                'source_standardized_pAUC_1pct': float(roc_auc_score(y, s, sample_weight=w, max_fpr=.01)),
                                'oracle_cut_NOT_DEPLOYABLE': cut,
                                'oracle_S_source_recall': float(recalls.loc[2]),
                                'oracle_S_rows_correct': int((y & (pred == 2)).sum()),
                                'oracle_target_repairs': int((d.target_578.to_numpy() & (pred == 2)).sum()),
                                'oracle_M_row_error': float(mrows), 'oracle_M_source_error': msource}
                for group, auc in ranked_sources.items():
                    src = d.group.eq(group).to_numpy() & y
                    sources.append({'fold': fold, 'protocol': protocol, 'group': int(group), 'model': name,
                                    'S_rows': int(src.sum()), 'AUC_against_source_balanced_M': float(auc),
                                    'oracle_S_recall': float((pred[src] == 2).mean())})
            cells.append({'fold': fold, 'protocol': protocol,
                          'M_rows': int((~y).sum()), 'S_rows': int(y.sum()),
                          'M_sources': int(d.loc[~y, 'group'].nunique()), 'S_sources': int(d.loc[y, 'group'].nunique()),
                          'models': detail})
    result = {
        'scope': 'Read-only fixed v65 epoch-12 OOF development diagnostics; no training and no current-calibration thresholding',
        'new_fits': 0, 'checkpoint_selection': 'Original predeclared terminal epoch 12 for both H0/H1, not best checkpoints',
        'encoder_provenance': {'revision': cache['revision'], 'original_upstream_asset_hashes_verified': True,
                               'supervised_v61_checkpoint_used': False, 'frozen_features_sha256': cache['features_sha256']},
        'checkpoint_provenance': provenance, 'role_overlap': role_overlap,
        'rows': len(hard), 'S_rows': int(hard.label.eq(2).sum()), 'S_sources': int(hard.loc[hard.label.eq(2), 'group'].nunique()),
        'cells': cells, 'per_S_source_ranking': sources,
        'mean_six_cell_pAUC': {name: float(np.mean([c['models'][name]['source_standardized_pAUC_1pct'] for c in cells])) for name in scores},
        'comparisons': {},
        'limitations': [
            'Identical evaluation rows/folds do not imply matched training population, calibration roles, or model budgets.',
            'v65 trained on some rows now assigned to v69 calibration, so no new thresholds are fitted using that calibration.',
            'Terminal epoch choice is fixed here; all methods and development data have still been inspected adaptively.',
            'Evaluation-answer oracle cutoffs are only fixed-score diagnostic upper frontiers, never deployable estimates.',
            'Argmax neural decisions and B frozen-budget decisions do not have matched operating costs.',
            'Per-S-source AUC uses that S source against source-balanced M; it is not a within-single-class AUC.',
            'Source symbols need not denote independent physical hosts, and row-level unique corrections can share source fingerprints.',
            'No ensemble weights, best checkpoint, seed, budget, or router have been selected by this audit.',
        ],
        'input_sha256': inputs, 'script_sha256': sha(__file__),
    }
    for name in ['H0', 'H1']:
        result['comparisons'][name] = {
            'fixed_argmax_vs_B_frozen_budget_UNMATCHED': effects(hard, fixed[name], fixed['B69'], hard.target_578.to_numpy()),
            'own_evaluation_oracle_vs_B_own_evaluation_oracle_NOT_DEPLOYABLE': effects(hard, oracle[name], oracle['B69'], hard.target_578.to_numpy()),
            'own_evaluation_oracle_vs_B_frozen_budget_UNMATCHED': effects(hard, oracle[name], fixed['B69'], hard.target_578.to_numpy()),
        }
    for path, digest in inputs.items(): assert sha(ROOT/path) == digest
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(output + '\n', encoding='utf-8')
    else: print(output)


if __name__ == '__main__': main()
