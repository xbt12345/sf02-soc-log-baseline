"""Frozen V116 diagnosis: no fitting, threshold search, or checkpoint selection."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from v116_preflight import ROOT, DEST as PREV, TRACE, save, sha

DEST = ROOT / 'artifacts/v117_selection_failure_review_20260929'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def describe(d, p, scope, fold, epoch, role):
    out = []
    for cls in (1, 2):
        mask = d.truth.eq(cls).to_numpy()
        q = d[mask]
        if not len(q):
            continue
        prob = p[mask].astype(np.float64)
        target = prob[:, cls]
        ce = -np.log(np.clip(target, 1e-15, 1))
        # Independently recompute CE with sklearn, not only our formula.
        assert np.isclose(ce.mean(), log_loss(np.full(len(q), cls), prob, labels=[0, 1, 2]), atol=1e-6)
        residual = prob - np.eye(3)[np.full(len(q), cls)]
        root_mass = q.groupby('root').size().to_numpy(dtype=float)
        out.append(dict(scope=scope, fold=fold, epoch=epoch, role=role, truth=cls,
                        rows=len(q), unique_inputs=q.local.nunique(), roots=q.root.nunique(),
                        correct=int((prob.argmax(1) == cls).sum()), mean_target_probability=float(target.mean()),
                        row_CE=float(ce.mean()), multiclass_brier=float((residual ** 2).sum(1).mean()),
                        source_equal_CE=float(pd.DataFrame({'root': q.root.to_numpy(), 'ce': ce}).groupby('root').ce.mean().mean()),
                        inverse_Herfindahl_roots=float(root_mass.sum() ** 2 / (root_mass ** 2).sum())))
    return out


def main():
    assert not DEST.exists(), 'Review output is immutable; do not silently rerun.'
    receipt = read(PREV / 'delivery.json')
    hashes = receipt.get('artifact_sha256', receipt.get('file_sha256', {}))
    assert hashes, list(receipt)
    for rel, digest in hashes.items():
        assert sha(ROOT / rel) == digest, rel
    manifest = pd.read_parquet(PREV / 'inner_split_manifest.parquet')
    trace = pd.read_parquet(TRACE, columns=['row_position', 'local', 'root', 'fold', 'truth', 'facts_json'])
    facts = trace.facts_json.map(json.loads)
    trace['protocol'] = facts.map(lambda f: f.get('transport_protocol', '<absent>'))
    trace['roles'] = facts.map(lambda f: str(f.get('src_role', '<absent>')) + '->' + str(f.get('dst_role', '<absent>')))
    m = manifest.merge(trace[['row_position', 'protocol', 'roles']], on='row_position', validate='many_to_one')
    records, stability, sources = [], [], [PREV / 'delivery.json', PREV / 'inner_split_manifest.parquet', TRACE,
                                            PREV / 'OOF_ASA_decisions.parquet', Path(__file__)]
    for fold in range(3):
        d = m[m.outer_fold.eq(fold) & ~m.role.eq('outer_test')].reset_index(drop=True)
        folder = PREV / f'inner_fold{fold}'
        ids = np.load(folder / 'inference_local_ids.npy')
        lookup = pd.Series(np.arange(len(ids)), index=ids)
        pos = lookup.loc[d.local].to_numpy()
        assert np.array_equal(ids[pos], d.local)
        saved = read(folder / 'checkpoints.json')
        sources.extend([folder / 'checkpoints.json', folder / 'inference_local_ids.npy'])
        predictions = {}
        for item in saved:
            epoch = item['epoch']
            path = folder / f'epoch{epoch}_prob.npy'
            assert sha(path) == item['prob_sha256']
            sources.append(path)
            p = np.load(path)[pos]
            assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
            predictions[epoch] = p.argmax(1)
            for role in ('fit', 'K', 'U', 'validation_collateral'):
                mask = d.role.eq(role).to_numpy()
                records.extend(describe(d[mask], p[mask], 'inner', fold, epoch, role))
        for role in ('fit', 'K', 'U', 'validation_collateral'):
            for cls in (1, 2):
                mask = (d.role.eq(role) & d.truth.eq(cls)).to_numpy()
                a, b = predictions[15][mask], predictions[25][mask]
                stability.append(dict(fold=fold, role=role, truth=cls, rows=int(mask.sum()),
                                      changed=int((a != b).sum()), learned_15_to_25=int(((a != cls) & (b == cls)).sum()),
                                      lost_15_to_25=int(((a == cls) & (b != cls)).sum())))
    # Only fold 1 has both outer checkpoints saved; do not fabricate other epochs.
    d = m[m.outer_fold.eq(1)].reset_index(drop=True)
    folder = PREV / 'outer_fold1'
    ids = np.load(folder / 'inference_local_ids.npy')
    positions = pd.Series(np.arange(len(ids)), index=ids).loc[d.local].to_numpy()
    for item in read(folder / 'checkpoints.json'):
        epoch = item['epoch']
        path = folder / f'epoch{epoch}_prob.npy'
        assert sha(path) == item['prob_sha256']
        sources.append(path)
        p = np.load(path)[positions]
        for role, mask in [('fit_all', d.fold.ne(1).to_numpy()), ('outer_test', d.fold.eq(1).to_numpy())]:
            records.extend(describe(d[mask], p[mask], 'outer', 1, epoch, role))
    decisions = pd.read_parquet(PREV / 'OOF_ASA_decisions.parquet')
    joined = decisions.merge(trace[['row_position', 'protocol', 'roles']], on='row_position', validate='one_to_one')
    population = m.assign(parameter_key=m.parameter.fillna('<missing>')).groupby(
        ['outer_fold', 'role', 'truth', 'protocol', 'parameter_key', 'roles'], dropna=False).agg(
        rows=('local', 'size'), roots=('root', 'nunique'), unique_inputs=('local', 'nunique')).reset_index()
    changes = joined.assign(parameter_key=joined.parameter.fillna('<missing>')).groupby(
        ['truth', 'protocol', 'parameter_key', 'roles'], dropna=False).agg(
        rows=('local', 'size'), repaired=('repaired', 'sum'), regressed=('regressed', 'sum')).reset_index()
    params = []
    f1 = m[m.outer_fold.eq(1)]
    for role in ('fit', 'K', 'U', 'validation_collateral', 'outer_test'):
        q = f1[f1.role.eq(role)]
        for cls in (1, 2):
            z = q[q.truth.eq(cls)]
            params.append(dict(role=role, truth=cls, rows=len(z), missing=int(z.parameter.isna().sum()),
                               icmp_3_13=int(z.parameter.eq('["icmp",3,13]').sum())))
    inner_unique = f1.loc[f1.role.eq('fit'), 'local'].nunique()
    outer_unique = f1.loc[~f1.role.eq('outer_test'), 'local'].nunique()
    summary = {'status': 'frozen_selection_failure_audit', 'classifier_fits': 0, 'optimizer_steps': 0,
               'checkpoint_reselection': False, 'all_prior_delivery_hashes_verified': True,
               'fold1_populations': params, 'fold1_step_counts': {
                   'inner_unique': inner_unique, 'outer_unique': outer_unique,
                   'inner_15_steps': math.ceil(inner_unique / 256) * 15,
                   'inner_25_steps': math.ceil(inner_unique / 256) * 25,
                   'outer_15_steps': math.ceil(outer_unique / 256) * 15,
                   'outer_25_steps': math.ceil(outer_unique / 256) * 25},
               'S_regressions': int(joined.loc[joined.truth.eq(2), 'regressed'].sum()),
               'S_regressions_missing_parameter': int(joined.loc[joined.truth.eq(2) & joined.parameter.isna(), 'regressed'].sum()),
               'K_U_missing_parameter_rows': int(m.loc[m.role.isin(['K', 'U']), 'parameter'].isna().sum()),
               'interpretation_limits': ['All outer folds are previously inspected development evidence.',
                   'Inverse Herfindahl measures concentration, not a count of independent organizations.',
                   'Post-hoc CE/collateral diagnostics do not retroactively authorize a different selected epoch.',
                   'A missing value subgroup absent from validation is a coverage defect, not proof that including it alone would prevent regression.']}
    DEST.mkdir()
    pd.DataFrame(records).to_csv(DEST / 'checkpoint_probability_diagnostics.csv', index=False)
    pd.DataFrame(stability).to_csv(DEST / 'inner_15_to_25_transitions.csv', index=False)
    population.to_csv(DEST / 'population_by_parameter.csv', index=False)
    changes.to_csv(DEST / 'changes_by_parameter.csv', index=False)
    save(DEST / 'audit.json', summary)
    save(DEST / 'input_receipt.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in dict.fromkeys(sources)})
    print(json.dumps(summary, ensure_ascii=False, default=int))
    result = pd.DataFrame(records)
    print(result[(result.fold.eq(1)) & result.epoch.isin([15, 25])].to_string(index=False))


if __name__ == '__main__':
    main()
