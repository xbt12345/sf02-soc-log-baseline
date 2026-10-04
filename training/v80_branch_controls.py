"""Zero-fit branch controls. Altered-input scores are diagnostics, not candidates."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from run_v75 import OUT, adapter, load_sparse, save, sha, metrics
from v80_failure_attribution import DEST, LAST
from v79_execute import rows


def main():
    path = DEST / 'branch_controls.json'
    if path.exists():
        raise FileExistsError(path)
    adapter()
    from v38_learning import BIT_FIELDS
    r = rows()
    fid = np.load(LAST / 'row_feature_id.npy', mmap_mode='r')
    x = load_sparse(LAST / 'X')
    enc = joblib.load(OUT / 'facts_encoder.joblib')
    default = normalize(enc.transform([{}]).astype(np.float32), copy=False)
    model = joblib.load(LAST / 'P4_cef_fields_whole_format.joblib')
    take = np.flatnonzero(r.route.eq('cef_fields').to_numpy() & r.fold.isin([0, 2]).to_numpy())
    xx = x[fid[take]]
    defaults = sparse.vstack([default] * len(take), format='csr')
    result = {}
    for facts_default in (False, True):
        for clear_conflict in (False, True):
            meta = xx[:, -18:].tolil()
            if clear_conflict:
                meta[:, 17] = 0
            factblock = defaults if facts_default else xx[:, 65792:66269]
            variant = sparse.hstack([xx[:, :65792], factblock, meta.tocsr()], format='csr')
            pred = (variant @ model['coef'] + model['intercept']).argmax(1)
            key = f'default_facts_{facts_default}_clear_conflict_{clear_conflict}'
            result[key] = {}
            for fold, role in [(0, 'H'), (2, 'C')]:
                mask = r.fold.to_numpy()[take] == fold
                result[key][role] = metrics(confusion_matrix(r.label_index.to_numpy()[take][mask], pred[mask], labels=[0, 1, 2]))
    # Reproduce the previous paired alteration, rather than interpreting it as one-factor evidence.
    prior = json.loads((LAST / 'P4_transfer.json').read_text())['formats']['cef_fields']['whole_format']
    assert result['default_facts_False_clear_conflict_False']['H']['cm'] == prior['H_metrics']['cm']
    assert result['default_facts_True_clear_conflict_True']['H']['cm'] == prior['unknown_syntax_no_adapter']['H_metrics']['cm']

    take = np.flatnonzero(r.fold.eq(2).to_numpy() & r.route.eq('asa').to_numpy() & r.label_index.eq(2).to_numpy())
    model = joblib.load(LAST / 'P1_B_ovr.joblib')
    xx = x[fid[take]]
    pred = (xx @ model['coef'] + model['intercept']).argmax(1)
    take = take[pred != 2]
    xx = xx[pred != 2]
    facts = pd.read_parquet(OUT / 'projections.parquet', columns=['facts']).facts
    fs = [json.loads(facts.iloc[int(i)]) for i in r.projection_id.to_numpy()[take]]
    names = np.asarray(enc.names()).astype(str)
    dw = model['coef'][:, 1] - model['coef'][:, 2]
    missing = {}
    for key, (width, sentinel) in BIT_FIELDS.items():
        positions = np.flatnonzero(np.char.startswith(names, key + ':bit'))
        unavailable = np.array([key not in f or f[key] == sentinel for f in fs])
        absent = np.array([key not in f for f in fs])
        contributions = np.asarray(xx[:, 65792 + positions] @ dw[65792 + positions]).ravel()
        missing[key] = {'absent_rows': int(absent.sum()), 'unavailable_rows': int(unavailable.sum()),
                        'all_error_rows': len(take), 'mean_unavailable_margin_contribution_over_all_errors': float(np.mean(contributions * unavailable)),
                        'default_active_value_bits': int(enc.transform([{}])[:, positions].nnz)}
    payload = {'new_fits': 0, 'source_sha256': sha(__file__), 'cef_two_by_two': result,
               'asa_wrong_S_missing_value_contributions': missing,
               'scope': 'Exact fixed-classifier decomposition and altered-input stress only. No improved trained classifier. Missingness attribution is relative to the feature basis and does not prove causality or label validity.'}
    save(path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
