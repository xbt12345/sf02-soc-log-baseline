"""Independent coverage-qualified split and original-row transfer verification."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
import run_v54 as v
import run_v56 as t
import run_v57_port_transfer as run
from verify_v56 import independent_keys
from verify_v55 import manual
from verify_v57_stage1 import codes, add_offset, independent_gate
from verify_v57_stage2 import row_objective

OUT = t.ROOT / 'evidence/2026-09-14/v57_port_verification'

def active(x):
    return np.isin(x.transport_protocol.to_numpy(), ['tcp', 'udp']) & (x.src_port_fixed.to_numpy() == v.MISSING) & (x.dst_port_fixed.to_numpy() != v.MISSING)

def main():
    assert not OUT.exists()
    c, r, x = run.load()
    fine = np.asarray(independent_keys(x)['fine'])
    asa = r.route.eq('asa').to_numpy()
    context = (x.transport_protocol.eq('tcp') & x.src_role.eq('outside') & x.dst_role.eq('dmz')).to_numpy()
    natural = context & active(x)
    donors = context & x.src_port_fixed.ne(v.MISSING).to_numpy() & x.dst_port_fixed.ne(v.MISSING).to_numpy()
    expected_protocols = []
    for protocol in t.PROTOCOLS:
        old = pd.read_parquet(t.RUN / (protocol + '_split.parquet')).role.to_numpy()
        idx = np.flatnonzero((old != 'evaluation') & asa)
        selected = None
        for j, (aa, bb) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=20260916).split(idx, r.label_index.to_numpy()[idx], fine[idx])):
            candidate = old.copy(); candidate[idx[aa]] = 'fit'; candidate[idx[bb]] = 'calibration'
            donor = int(((candidate == 'fit') & donors).sum())
            target = int(((candidate == 'calibration') & natural).sum())
            expected = {'fold': j, 'fit_TCP_donors': donor, 'calibration_natural_TCP_source_missing': target, 'coverage_qualified': donor > 0 and target > 0}
            assert expected == c['coverage_audits'][protocol]['folds'][j]
            if donor > 0 and target > 0 and selected is None:
                selected = j
                selected_roles = candidate.copy()
        assert selected == c['coverage_audits'][protocol]['selected_fold']
        if selected is not None:
            expected_protocols.append(protocol)
            np.testing.assert_array_equal(selected_roles, pd.read_parquet(run.RUN / (protocol + '_split.parquet')).role)
        else:
            assert not (run.RUN / protocol).exists()
    assert expected_protocols == c['protocols']
    OUT.mkdir(parents=True)
    records, replayed, maxp, maxbase = [], 0, 0., 0.
    for protocol in c['protocols']:
        folder = run.RUN / protocol
        complete = v.read(folder / 'complete.json')
        for p, h in complete['bindings'].items():
            assert v.sha(folder / p) == h
        before = v.read(folder / 'before_evaluation.json')
        assert not before['evaluation_generated'] and not any(p.endswith('evaluation.parquet') for p in before['bindings'])
        roles = pd.read_parquet(run.RUN / (protocol + '_split.parquet')).role.to_numpy()
        for left, right in [('fit', 'calibration'), ('fit', 'evaluation'), ('calibration', 'evaluation')]:
            assert set(r.loc[roles == left, 'body_group']).isdisjoint(r.loc[roles == right, 'body_group'])
        assert set(fine[asa & (roles == 'fit')]).isdisjoint(fine[asa & (roles == 'calibration')])
        base = joblib.load(folder / 'base.joblib')
        assert base['model'].t_ == (roles == 'fit').sum() * c['epochs']
        for j, field in enumerate(v.FIELDS):
            assert base['input']['seen'][j] == set(x.loc[roles == 'fit', field])
        selection = v.read(folder / 'selection.json')
        original_cal = pd.read_parquet(folder / 'base_calibration.parquet')
        choices, objectives = {}, {}
        for name in c['variants']:
            bundle = joblib.load(folder / (name + '.joblib'))
            parent = active(x) if name == 'natural_only' else np.isin(x.transport_protocol.to_numpy(), ['tcp', 'udp']) & x.dst_port_fixed.ne(v.MISSING).to_numpy()
            participating = (roles == 'fit') & parent
            np.testing.assert_array_equal(bundle['participating_row_positions'], r.loc[participating, 'row_position'])
            assert len(set(bundle['participating_row_positions'])) == participating.sum()
            xx = x.loc[participating].copy(); xx[['src_port_fixed', 'src_port_range']] = v.MISSING
            val, grad = row_objective(bundle, base, xx, r.loc[participating, 'label_index'], r.loc[participating, 'body_group'], c['strength'])
            assert bundle['base_sha256'] == v.sha(folder / 'base.joblib')
            objectives[name] = {'original_row_mean_objective': val, 'scaled_gradient': grad, 'parents': int(participating.sum())}
            predcal = pd.read_parquet(folder / (name + '_calibration.parquet'))
            gate = independent_gate(original_cal, predcal)
            assert gate == selection['candidates'][name]['gate']
            z = predcal[predcal.route == 'asa']; y = z.label_index.to_numpy(); pp = z[['p_0', 'p_1', 'p_2']].to_numpy()
            loss = float(-np.log(np.maximum(pp[np.arange(len(y)), y], 1e-30)).mean())
            passed = bundle['optimizer']['eligible'] and gate['quality_passed']
            assert passed == selection['candidates'][name]['accepted']
            assert loss == selection['candidates'][name]['calibration_log_loss']
            if passed:
                choices[name] = loss
            for role in ['calibration'] + (['evaluation'] if name == selection['selected'] else []):
                mask = roles == role
                basepred = pd.read_parquet(folder / ('base_' + role + '.parquet'))
                pred = predcal if role == 'calibration' else pd.read_parquet(folder / 'evaluation.parquet')
                for view, fields in v.VIEWS.items():
                    xx = x.loc[mask].reset_index(drop=True).copy(); xx[fields] = v.MISSING
                    b = basepred[basepred.scenario == view].reset_index(drop=True)
                    a = pred[pred.scenario == view].reset_index(drop=True)
                    pd.testing.assert_frame_equal(a[r.columns], r.loc[mask].reset_index(drop=True))
                    bp = manual(base, xx)
                    np.testing.assert_allclose(bp, b[['p_0', 'p_1', 'p_2']], atol=3e-6, rtol=0)
                    np.testing.assert_array_equal(bp.argmax(axis=1), b.pred)
                    maxbase = max(maxbase, float(abs(bp - b[['p_0', 'p_1', 'p_2']].to_numpy()).max()))
                    used = active(xx)
                    p = b[['p_0', 'p_1', 'p_2']].to_numpy().astype(float)
                    yp = b.pred.to_numpy(copy=True)
                    if used.any():
                        p[used] = softmax(add_offset(p[used], codes(bundle['design'], xx.loc[used]), bundle['weights']), axis=1)
                        yp[used] = p[used].argmax(axis=1)
                    np.testing.assert_array_equal(p[~used], a.loc[~used, ['p_0', 'p_1', 'p_2']])
                    np.testing.assert_array_equal(yp, a.pred)
                    np.testing.assert_allclose(p, a[['p_0', 'p_1', 'p_2']], atol=1e-12, rtol=0)
                    maxp = max(maxp, float(abs(p - a[['p_0', 'p_1', 'p_2']].to_numpy()).max()))
                    replayed += len(a)
        selected = min(choices, key=lambda name: (choices[name], c['variants'].index(name))) if choices else 'zero'
        assert selected == selection['selected'] == complete['selected']
        assert (folder / 'evaluation.parquet').exists() == (selected != 'zero')
        records.append({'protocol': protocol, 'selected': selected, 'objectives': objectives})
    result = {'all_checks_passed': True, 'coverage_qualified_protocols': expected_protocols, 'records': records, 'new_base_fits': len(records), 'new_adapter_fits': 2 * len(records), 'score_rows_replayed': replayed, 'max_probability_difference': maxp, 'max_manual_base_difference': maxbase, 'scope': 'Independent coverage-first split choice, no repeated parent rows, original-row objective/support/gradient, model forward passes and calibration-before-evaluation selection. No test scores used to choose inner fold.'}
    v.save(OUT / 'verification.json', result)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    v.save(OUT / 'receipt.json', {'files': {p.name: v.sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({k: a for k, a in result.items() if k != 'records'}), flush=True)

if __name__ == '__main__':
    main()
