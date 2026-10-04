"""Read-only postselection source diagnosis; no new fit or V tuning."""
import json

import numpy as np
import pandas as pd

from run_v75 import ROOT, save, sha
from v100_prepare import DEST, V92


def main():
    dest = DEST/'source_diagnosis.json'
    if dest.exists():
        raise FileExistsError(dest)
    s = pd.read_parquet(DEST/'source_rows.parquet')
    asa = s[s.is_ASA].copy()
    oof = pd.read_parquet(DEST/'oof_predictions.parquet')
    assert np.array_equal(asa.row_position.to_numpy(), oof.row_position.to_numpy())
    truth = asa.label_index.to_numpy()
    old = np.load(ROOT/'artifacts/v97_matched_teacher_20260928/MAG_prediction.npy')
    fid = asa.fid.to_numpy()
    current = old[fid]
    frequency = asa.groupby(['fid', 'label_index']).size().unstack(fill_value=0)
    conflicting = set(frequency.index[(frequency > 0).sum(1) > 1])
    historical = (current != truth) & ~asa.fid.isin(conflicting).to_numpy()
    assert int(historical.sum()) == 229
    six = json.loads((ROOT/'artifacts/v98_boundary_and_gate_audit_20260928/six_supported_training_errors.json').read_text())
    focus_ids = {int(item['R0']) for item in six}
    focus = asa.fid.isin(focus_ids).to_numpy()
    assert int(focus.sum()) == 6
    ids = np.load(V92/'ASA_input_ids.npy')
    lookup = np.full(int(ids.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    local = lookup[fid]
    assert (local >= 0).all()
    endpoint = {}
    for view in ('R0', 'N1'):
        for arm in ('ERM', 'MAG'):
            for tag in ('200', 'long'):
                name = f'{view}_{arm}_{tag}'
                oof_pred = oof[name].to_numpy()
                records = []
                for fold in range(3):
                    scores = np.load(DEST/f'fold{fold}_{view}'/f'{arm}_{tag}_ASA_scores.npy')
                    pred = scores[local].argmax(1)
                    fit = asa.fold.to_numpy() != fold
                    for label, mask in [('historical_unconflicted_229', historical),
                                        ('six_supported_S', focus)]:
                        for role, role_mask in [('fit', fit), ('hold', ~fit)]:
                            subset = mask & role_mask
                            records.append({'fold': fold, 'slice': label, 'role': role,
                                'rows': int(subset.sum()),
                                'correct': int((pred[subset] == truth[subset]).sum())})
                endpoint[name] = {'historical_229_OOF_correct': int((oof_pred[historical] == truth[historical]).sum()),
                    'six_supported_S_OOF_correct': int((oof_pred[focus] == truth[focus]).sum()),
                    'fold_exposure_records': records}
    reference = oof['R0_teacher'].to_numpy()
    best_branch = oof['R0_MAG_200'].to_numpy()
    n1_stable = oof['N1_MAG_long'].to_numpy()
    components = {}
    for name, pred in [('R0_teacher', reference), ('R0_MAG_200', best_branch),
                       ('N1_MAG_long', n1_stable)]:
        wrong = asa[pred != truth]
        comp = wrong.groupby(['component', 'label_index']).size().sort_values(ascending=False)
        top = []
        for (c, label), n in comp.head(12).items():
            total = int(((asa.component == c)&(asa.label_index == label)).sum())
            top.append({'component': int(c), 'label': int(label),
                        'errors': int(n), 'support': total})
        components[name] = {'error_rows': int((pred != truth).sum()),
            'error_components': int(wrong.component.nunique()),
            'top_12_component_class_errors': top,
            'largest_component_error_fraction': float(comp.iloc[0]/len(wrong)) if len(wrong) else 0}
    source_s_by_fold = {}
    for fold in range(3):
        subset = asa[(asa.fold == fold)&(asa.label_index == 2)]
        counts = subset.groupby('component').size().sort_values(ascending=False)
        source_s_by_fold[str(fold)] = {'rows': len(subset), 'components': len(counts),
            'largest_component_rows': int(counts.iloc[0]),
            'largest_component_share': float(counts.iloc[0]/len(subset))}
    result = {'status': 'source_only_postselection_diagnosis',
        'source_sha256': sha(__file__), 'historic_v97_error_rows_without_R0_label_conflict': int(historical.sum()),
        'six_supported_S_rows': int(focus.sum()), 'A_B_R0_conflict_floor': int(frequency.min(axis=1).sum()),
        'endpoint_behavior': endpoint, 'error_components': components,
        'OOF_S_component_exposure': source_s_by_fold,
        'no_independent_incident_truth': True, 'V_and_historic_labels_used_for_method_selection': False,
        'scope': 'Diagnosis on repeatedly inspected source labels; trained vs held-out rows separated per fold. No claim that a model correction is independently true attack detection.'}
    save(dest, result)
    print(json.dumps({'stage': 'source_diagnosis',
        'historical_229_OOF_correct': {k:v['historical_229_OOF_correct'] for k,v in endpoint.items()},
        'six_S_OOF_correct': {k:v['six_supported_S_OOF_correct'] for k,v in endpoint.items()},
        'S_component_support': source_s_by_fold}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
