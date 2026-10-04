"""No-fit verification of v72 diagnostic receipts and primary saved decisions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_mapping(mapping):
    for path, digest in mapping.items():
        assert sha(ROOT/path) == digest, path
    return len(mapping)


def main():
    counts = {}
    for v, folder in [('v69','v69_support_control'), ('v71','v71_resolution')]:
        delivery = read(ROOT/f'evidence/2026-09-21/{folder}/delivery.json')
        counts[v] = verify_mapping(delivery['artifact_sha256'])
    audits = {}
    for name in ['audit_fixed_blend', 'audit_terminal_gradients', 'v65_complement_audit']:
        data = read(HERE/f'{name}.json')
        assert data['new_fits'] == 0
        assert data['script_sha256'] == sha(HERE/f'{name}.py')
        counts[name] = verify_mapping(data.get('inputs_sha256', data.get('input_sha256', {})))
        audits[name] = data
    pred = pd.read_parquet(HERE/'fixed_blend_decisions.parquet')
    manifest = pd.read_parquet(ROOT/'artifacts/v67_targeted_20260920/manifest.parquet')
    baseline = pd.read_parquet(ROOT/'artifacts/v69_support_control_20260921/B_small_sources_budget0.01_oof.parquet')
    np.testing.assert_array_equal(pred.row_position, manifest.row_position)
    np.testing.assert_array_equal(pred.label, manifest.label)
    np.testing.assert_array_equal(pred.row_position, baseline.row_position)
    y = pred.label.to_numpy(); p = pred.pred.to_numpy(); b = baseline.pred.to_numpy()
    primary = audits['audit_fixed_blend']['results']['0.01']
    errors = {str(c): int(((y == c) & (p != y)).sum()) for c in [0,1,2]}
    assert errors == {'0':0, '1':600, '2':590}
    assert int((p != y).sum()) == primary['vs_historical']['total_errors'] == 1190
    target = manifest.target_578.to_numpy()
    assert int((target & (p == 2)).sum()) == primary['vs_historical']['target_fixed'] == 56
    for c, name in [(0,'B'), (1,'M'), (2,'S')]:
        k = y == c
        fixed = int((k & (p==y) & (b!=y)).sum())
        broken = int((k & (p!=y) & (b==y)).sum())
        assert (fixed,broken) == (primary['vs_B69'][name]['fixed'],primary['vs_B69'][name]['broken'])
    for item in audits['audit_terminal_gradients']['entries']:
        path = ROOT/f"artifacts/v71_resolution_20260921/fold{item['fold']}/{item['arm']}/fit.parquet"
        f = pd.read_parquet(path); k = f.label.eq(item['label']).to_numpy()
        g = f.score.to_numpy()-f.label.eq(2).astype(float).to_numpy()
        observed = np.abs(g[k]).sum()/np.abs(g).sum()
        assert abs(observed-item['terminal_absolute_gradient_fraction']) < 1e-12
    neural = audits['v65_complement_audit']
    for model in ['B69','H0','H1']:
        average = np.mean([c['models'][model]['source_standardized_pAUC_1pct'] for c in neural['cells']])
        assert abs(average-neural['mean_six_cell_pAUC'][model]) < 1e-12
        udp = [c for c in neural['cells'] if c['protocol']=='udp']
        assert len(udp)==3 and all(c['models'][model]['oracle_S_source_recall']==0 for c in udp)
    result = dict(all_checks_passed=True, new_fits=0, unchanged_bound_files_and_verified_inputs=counts,
        primary_blend_errors=errors, primary_blend_target_correct=56,
        model_quality_acceptance=False, accepted_repairs=0,
        scope='Artifact integrity, saved primary counts, terminal derivatives, neural audit aggregate consistency. Not independent model-quality validation.',
        local_artifact_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='verification.json'})
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='local_artifact_sha256'},ensure_ascii=False))


if __name__ == '__main__':
    main()
