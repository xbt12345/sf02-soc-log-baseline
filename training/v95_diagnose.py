"""Post-selection paired errors, historic guard, behavior coverage and component bootstrap."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, read, save, sha
from v89_common import data, raw_counts
from v85_protection import changes, cm_from_counts
from v81_training_contract import compare
from v92_common import DEST as V92
from v95_prepare import DEST
from v95_four_arm import class_metrics


def role_report(r, y, fid, baseline, candidate, mask):
    counts = raw_counts(fid, y, mask, len(baseline))
    result = changes(counts, baseline, candidate)
    return {**class_metrics(np.asarray(result['cm'])),
            'positive_flips': result['positive_flips'],
            'negative_flips': result['negative_flips'],
            'positive_flips_by_class': result['positive_flips_by_class'],
            'negative_flips_by_class': result['negative_flips_by_class'],
            'guard': compare(cm_from_counts(counts, baseline), np.asarray(result['cm']), True)}


def main():
    assert not (DEST / 'diagnosis.json').exists()
    selection = read(DEST / 'selection.json')
    assert selection['status'] == 'V_selection_frozen_before_inner_C_H'
    r, y, fid, _, old, _, fit = data()
    roles = pd.read_parquet(DEST / 'full_format_roles.parquet', columns=['role']).role.to_numpy()
    teacher = np.load(DEST / 'teacher_prediction.npy')
    trial = [('E00', 100), ('E10', 200), ('E01', 50), ('E11', 50)]
    assert selection['V_selected']['arm'] == 'E10' and selection['V_selected']['step'] == 200
    masks = {'A': roles == 'A', 'B': roles == 'B', 'V': roles == 'V',
             'inner': r.fold.eq(1).to_numpy(), 'C': r.fold.eq(2).to_numpy(), 'H': r.fold.eq(0).to_numpy(),
             'old_full_fit': fit}
    role_table = {}
    for base_name, baseline in [('A_teacher', teacher), ('historic_v85_teacher', old)]:
        role_table[base_name] = {role: role_report(r, y, fid, baseline, baseline, mask)
                                 for role, mask in masks.items()}
    predictions = {}
    for arm, step in trial:
        name = f'{arm}_step{step:03}'
        p = DEST / f'{name}_prediction.npy'
        snap = next(s for s in read(DEST / f'{arm}_snapshots.json') if s['step'] == step)
        assert sha(p) == snap['prediction_sha256']
        predictions[name] = np.load(p)
        role_table[name] = {
            'versus_A_teacher': {role: role_report(r, y, fid, teacher, predictions[name], mask)
                                  for role, mask in masks.items()},
            'versus_historic_v85_teacher': {role: role_report(r, y, fid, old, predictions[name], mask)
                                           for role, mask in masks.items()}}
    save(DEST / 'paired_role_metrics.json', role_table)
    selected = predictions['E10_step200']
    baseline = predictions['E00_step100']
    prior = np.load(V92 / 'U0R_prediction.npy')
    inner = masks['inner']
    bad_m = np.flatnonzero(inner & (y == 1) & (old[fid] == 1) & (prior[fid] != 1))
    gain_s = np.flatnonzero(inner & (y == 2) & (old[fid] != 2) & (prior[fid] == 2))
    assert len(bad_m) == 500 and len(gain_s) == 58
    flow = {}
    for name, pred in [('A_teacher', teacher), ('matched_ERM_E00_100', baseline),
                       ('V_selected_E10_200', selected), ('historic_U0R', prior)]:
        flow[name] = {'old_500_M_still_correct': int((pred[fid[bad_m]] == 1).sum()),
                      'old_58_S_repair_still_correct': int((pred[fid[gain_s]] == 2).sum())}
    save(DEST / 'historic_500_58_flow.json', flow)
    aux = pd.read_parquet(ROOT / 'artifacts/v94_research_mechanism_20260928/prospective_ASA_roles.parquet')
    eligible = read(ROOT / 'artifacts/v94_research_mechanism_20260928/auxiliary_eligibility.json')['eligible_behaviors']
    slices = []
    for behavior in eligible:
        for role in 'ABV':
            for cls in [1, 2]:
                ix = aux[(aux.behavior == behavior) & (aux.proposed_role == role) & (aux.label_index == cls)].row_position.to_numpy()
                if not len(ix): continue
                slices.append({'behavior': behavior, 'role': role, 'class': cls, 'rows': len(ix),
                               'components': int(r.component.iloc[ix].nunique()),
                               'A_teacher_correct': int((teacher[fid[ix]] == cls).sum()),
                               'E00_correct': int((baseline[fid[ix]] == cls).sum()),
                               'E10_correct': int((selected[fid[ix]] == cls).sum()),
                               'E01_correct': int((predictions['E01_step050'][fid[ix]] == cls).sum()),
                               'E11_correct': int((predictions['E11_step050'][fid[ix]] == cls).sum())})
    pd.DataFrame(slices).to_csv(DEST / 'auxiliary_behavior_classwise.csv', index=False)
    v = np.flatnonzero(masks['V'])
    diff = (selected[fid[v]] == y[v]).astype(np.int8) - (baseline[fid[v]] == y[v]).astype(np.int8)
    groups, inverse = np.unique(r.component.iloc[v].to_numpy(), return_inverse=True)
    component_delta = np.bincount(inverse, weights=diff, minlength=len(groups)).astype(np.int32)
    rng = np.random.default_rng(9501)
    bootstrap = np.empty(1000, dtype=np.int64)
    for i in range(len(bootstrap)):
        bootstrap[i] = component_delta[rng.integers(0, len(groups), len(groups))].sum()
    np.save(DEST / 'V_E10_vs_E00_component_bootstrap.npy', bootstrap)
    comparison = {'selected_vs_matched_ERM_V_correct_delta': int(diff.sum()),
                  'V_components': len(groups), 'changed_components': int((component_delta != 0).sum()),
                  'bootstrap_component_1000_q025_q50_q975': np.quantile(bootstrap, [.025, .5, .975]).tolist(),
                  'bootstrap_probability_delta_positive': float((bootstrap > 0).mean()),
                  'scope': 'Development component bootstrap; repeated inspected split, not independent external generalization'}
    save(DEST / 'paired_component_uncertainty.json', comparison)
    chosen = role_table['E10_step200']['versus_historic_v85_teacher']
    failures = {role: {'errors': stat['errors'], 'negative_flips': stat['negative_flips'],
                       'positive_flips': stat['positive_flips'], 'class_correct': stat['correct'],
                       'guard_passed': stat['guard']['eligible']}
                for role, stat in chosen.items() if role in ['inner', 'C', 'H', 'old_full_fit']}
    passed = all(stat['negative_flips'] == 0 and stat['guard_passed'] for stat in failures.values())
    result = {'status': 'locked_diagnosis_completed_after_V_selection',
              'V_selected': selection['V_selected'], 'historic_v85_guards': failures,
              'historic_all_guards_passed': passed,
              'two_rotation_training_triggered': passed,
              'model_promoted': False, 'quality_acceptance': False,
              'bootstrap': comparison,
              'scope': 'All roles are inspected official development; teacher A is a different anchor than historic v85. Locked H/C cannot retroactively change V selection.'}
    save(DEST / 'diagnosis.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
