"""Read-only audit of saved v69 scores; never fit or alter original artifacts.

Writes one audit JSON alongside this script. Per-role oracle thresholds use that
role's labels solely to describe saved-score limitations; they are not deployable
or validation estimates. Row loss/residual accounting describes final fit scores,
not the unrecorded optimization trajectory.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/v61_source_factorial_20260914_r2'
RUN = ROOT / 'artifacts/v69_support_control_20260921'
ARMS = ['A_original_S', 'B_small_sources', 'C_all_sources']
ROLES = ['fit', 'calibration', 'evaluation']
FIELDS = ['action', 'outcome', 'transport_protocol', 'src_role', 'dst_role',
          'src_port_fixed', 'dst_port_fixed', 'src_port_range', 'dst_port_range',
          'icmp_type', 'icmp_code', 'icmp_message', 'icmp_unreachable']
BUDGET = .01
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def cut_at_budget(d, budget=BUDGET):
    """Independent score-block cumulative implementation of two M budgets."""
    m = d[d.label.eq(1)]
    if m.empty:
        return None
    ns = m.group.nunique()
    sizes = m.groupby('group').score.transform('size')
    blocks = pd.DataFrame({'s': m.score, 'row': 1 / len(m),
                           'source': 1 / ns / sizes}).groupby('s').sum()
    blocks = blocks.sort_index(ascending=False)
    passed = (blocks.cumsum() <= budget + 1e-12).all(axis=1).to_numpy()
    bad = np.flatnonzero(~passed)
    return float(np.nextafter(float(blocks.index[bad[0]]), np.inf)) if len(bad) else 0.


def rates(d, cutoff):
    result = {}
    for label, name in [(1, 'M'), (2, 'S')]:
        z = d[d.label.eq(label)]
        result[name + '_rows'] = len(z)
        result[name + '_sources'] = int(z.group.nunique())
        if z.empty or cutoff is None:
            result[name + '_row_recall'] = None
            result[name + '_source_recall'] = None
            continue
        ok = (z.score.ge(cutoff) if label == 2 else z.score.lt(cutoff)).astype(float)
        result[name + '_row_recall'] = float(ok.mean())
        result[name + '_source_recall'] = float(ok.groupby(z.group).mean().mean())
    result['cutoff'] = cutoff
    return result


def ranking(d):
    if d.label.nunique() != 2:
        return {'source_AUC': None, 'source_pAUC_1pct': None}
    ct = d.groupby(['label', 'group']).score.transform('size')
    ns = d.groupby('label').group.nunique()
    weights = 1 / ct / d.label.map(ns)
    y = d.label.eq(2)
    return {'source_AUC': float(roc_auc_score(y, d.score, sample_weight=weights)),
            'source_pAUC_1pct': float(roc_auc_score(y, d.score, sample_weight=weights, max_fpr=.01))}


def geometry(d, calibrated_cut):
    if not len(d):
        return None
    own = cut_at_budget(d)
    result = {'rows': len(d), 'sources': int(d.group.nunique()),
              'fixed_calibration_cut': rates(d, calibrated_cut),
              'own_role_or_slice_oracle_1pct': rates(d, own), **ranking(d)}
    for label, name in [(1, 'M'), (2, 'S')]:
        z = d[d.label.eq(label)]
        result[name + '_score_quantiles'] = (
            {str(q): float(z.score.quantile(q)) for q in [0, .1, .5, .9, .99, 1]} if len(z) else None)
    if own is not None:
        rr = result['own_role_or_slice_oracle_1pct']
        assert max(1 - rr['M_row_recall'], 1 - rr['M_source_recall']) <= BUDGET + 1e-10
    return result


def main():
    f = pd.read_parquet(record(DATA / 'records.parquet'))
    with np.load(record(DATA / 'context.npz')) as context:
        f['empty_context'] = context['stats'][:, 0] == 0
    f['source_port_visible'] = f.src_port_fixed.astype(str).str.fullmatch(r'\d+')
    f['destination_port_visible'] = f.dst_port_fixed.astype(str).str.fullmatch(r'\d+')
    f['single_view'] = f[FIELDS].astype(str).agg('|'.join, axis=1)
    f['support_origin'] = np.where(f.role.eq('fit'), 'original',
                                    np.where(f.group.eq(3225), 'large_auxiliary', 'small_auxiliary'))
    meta = f[['row_position', 'label', 'group', 'transport_protocol', 'empty_context',
              'source_port_visible', 'destination_port_visible', 'single_view', 'support_origin']]
    output = {'scope': 'Saved v69 score diagnosis only; zero new fits; not a new model or blind validation',
              'oracle_scope': 'Per-role/slice label-informed upper envelope of monotone thresholds on these saved scores only; not deployable',
              'loss_scope': 'Unweighted objective mass and terminal binary log loss/logit residual, not optimization trajectory or causal attribution',
              'budget': BUDGET, 'cells': [], 'slices': [], 'training_loss': [], 'origin_cells': [],
              'source_training': [], 'fit_to_eval_contrasts': []}
    all_cells = {}
    for fold in range(3):
        for arm in ARMS:
            folder = RUN / f'fold{fold}' / arm
            cuts = json.loads(record(folder / 'thresholds.json').read_text(encoding='utf-8'))['0.01']
            frames = {}
            for role in ROLES:
                d = pd.read_parquet(record(folder / (role + '.parquet')))
                d = d.merge(meta, on=['row_position', 'label', 'group', 'transport_protocol'], validate='one_to_one')
                frames[role] = d
                for proto, z in d.groupby('transport_protocol'):
                    cutoff = cuts[proto]['threshold']
                    if role == 'calibration':
                        assert cut_at_budget(z) == cutoff
                    cell = {'fold': fold, 'arm': arm, 'role': role, 'protocol': proto,
                            **geometry(z, cutoff)}
                    output['cells'].append(cell)
                    all_cells[(fold, arm, role, proto)] = cell
                    axes = [('context', 'empty_context'), ('src_port', 'source_port_visible'),
                            ('dst_port', 'destination_port_visible')]
                    for axis, col in axes:
                        for value, zz in z.groupby(col):
                            output['slices'].append({'fold': fold, 'arm': arm, 'role': role,
                                'protocol': proto, 'axis': axis, 'value': bool(value), **geometry(zz, cutoff)})
                    for (empty, src, dst), zz in z.groupby(['empty_context', 'source_port_visible', 'destination_port_visible']):
                        output['slices'].append({'fold': fold, 'arm': arm, 'role': role, 'protocol': proto,
                            'axis': 'joint_context_ports', 'value': [bool(empty), bool(src), bool(dst)], **geometry(zz, cutoff)})
                for origin, z in d.groupby('support_origin'):
                    for proto, zz in z.groupby('transport_protocol'):
                        output['origin_cells'].append({'fold': fold, 'arm': arm, 'role': role,
                            'origin': origin, 'protocol': proto, **geometry(zz, cuts[proto]['threshold'])})
            d = frames['fit'].copy()
            yy = d.label.eq(2).astype(float).to_numpy()
            pp = np.clip(d.score.to_numpy(), 1e-15, 1 - 1e-15)
            d['log_loss'] = -(yy * np.log(pp) + (1 - yy) * np.log1p(-pp))
            d['abs_logit_residual'] = np.abs(pp - yy)
            d['partition'] = np.where(d.label.eq(1), 'M_all', 'S_' + d.support_origin)
            total_loss = float(d.log_loss.sum())
            total_residual = float(d.abs_logit_residual.sum())
            for partition, z in d.groupby('partition'):
                sources = z.groupby('group').size()
                output['training_loss'].append({'fold': fold, 'arm': arm, 'partition': partition,
                    'rows': len(z), 'sources': int(z.group.nunique()), 'single_views': int(z.single_view.nunique()),
                    'unweighted_row_mass': len(z) / len(d), 'max_source_rows': int(sources.max()),
                    'within_partition_source_mass_HHI': float(np.square(sources / sources.sum()).sum()),
                    'mean_terminal_log_loss': float(z.log_loss.mean()), 'sum_terminal_log_loss': float(z.log_loss.sum()),
                    'share_terminal_log_loss': float(z.log_loss.sum()) / total_loss,
                    'sum_absolute_logit_residual': float(z.abs_logit_residual.sum()),
                    'share_absolute_logit_residual': float(z.abs_logit_residual.sum()) / total_residual})
            for (origin, group), z in d[d.label.eq(2)].groupby(['support_origin', 'group']):
                output['source_training'].append({'fold': fold, 'arm': arm, 'origin': origin, 'group': int(group),
                    'rows': len(z), 'mean_score': float(z.score.mean()), 'mean_terminal_log_loss': float(z.log_loss.mean()),
                    'calibrated_recall': float(np.mean([s >= cuts[p]['threshold'] for s, p in zip(z.score, z.transport_protocol)]))})
            for proto in ['tcp', 'udp']:
                cells = [all_cells[(fold, arm, role, proto)] for role in ROLES]
                output['fit_to_eval_contrasts'].append({'fold': fold, 'arm': arm, 'protocol': proto,
                    'fixed_calibration_cut_S_source_recalls': {role: c['fixed_calibration_cut']['S_source_recall'] for role, c in zip(ROLES, cells)},
                    'own_oracle_same_1pct_cost_S_source_recalls': {role: c['own_role_or_slice_oracle_1pct']['S_source_recall'] for role, c in zip(ROLES, cells)},
                    'source_pAUC_1pct': {role: c['source_pAUC_1pct'] for role, c in zip(ROLES, cells)}})
    output['inputs_sha256'] = INPUTS
    output['audit_source_sha256'] = sha(Path(__file__))
    dest = Path(__file__).with_suffix('.json')
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'written': str(dest), 'input_files': len(INPUTS),
                      'cells': len(output['cells']), 'slices': len(output['slices']),
                      'fit_to_eval_contrasts': output['fit_to_eval_contrasts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
