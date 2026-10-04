"""Trace concentrated S failures and actual support without fitting or changing selection."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT, read, save, sha
from v89_common import data, raw_counts
from v95_prepare import DEST


def main():
    assert read(DEST / 'selection.json')['V_selected'] == {'arm': 'E10', 'step': 200,
                                                              'errors': 737, 'positive_flips': 18,
                                                              'negative_flips': 0, 'M_correct': 8672, 'S_correct': 609}
    assert not (DEST / 'failure_audit.json').exists()
    r, y, fid, _, old, _, fit = data()
    roles = pd.read_parquet(DEST / 'full_format_roles.parquet', columns=['role']).role.to_numpy()
    teacher = np.load(DEST / 'teacher_prediction.npy')
    e00 = np.load(DEST / 'E00_step100_prediction.npy')
    e10 = np.load(DEST / 'E10_step200_prediction.npy')
    aux = pd.read_parquet(ROOT / 'artifacts/v94_research_mechanism_20260928/prospective_ASA_roles.parquet')
    v = roles == 'V'; vs = v & (y == 2)
    frame = r.loc[vs, ['row_position', 'component', 'route']].copy()
    frame['feature_id'] = fid[vs]
    frame['teacher_error'] = teacher[fid[vs]] != 2
    frame['e00_error'] = e00[fid[vs]] != 2
    frame['e10_error'] = e10[fid[vs]] != 2
    frame['old_error'] = old[fid[vs]] != 2
    frame = frame.merge(aux[['row_position', 'behavior']], on='row_position', how='left', validate='one_to_one')
    counts = raw_counts(fid, y, np.isin(roles, ['A', 'B']), len(teacher))
    frame['training_M_rows_same_R0'] = counts[fid[vs], 1]
    frame['training_S_rows_same_R0'] = counts[fid[vs], 2]
    components = frame.groupby('component', dropna=False).agg(
        rows=('row_position', 'size'), teacher_errors=('teacher_error', 'sum'),
        e00_errors=('e00_error', 'sum'), e10_errors=('e10_error', 'sum'),
        old_errors=('old_error', 'sum'), distinct_R0=('feature_id', 'nunique'),
        same_R0_training_S_rows=('training_S_rows_same_R0', 'sum'))
    components = components.sort_values(['teacher_errors', 'rows'], ascending=False)
    components.reset_index().to_csv(DEST / 'V_S_component_errors.csv', index=False)
    top = components.head(10).reset_index().to_dict('records')
    empty = frame[frame.behavior.eq('{}')]
    main_behavior = '{"action": "deny", "dst_role": "dmz", "outcome": "blocked", "src_role": "outside", "transport_protocol": "tcp"}'
    main = frame[frame.behavior.eq(main_behavior)]
    def summarize(frame):
        return {'rows': len(frame), 'components': int(frame.component.nunique()),
                'distinct_R0': int(frame.feature_id.nunique()),
                'teacher_errors': int(frame.teacher_error.sum()),
                'E00_errors': int(frame.e00_error.sum()),
                'E10_errors': int(frame.e10_error.sum()),
                'old_v85_errors': int(frame.old_error.sum()),
                'exact_R0_with_any_training_S': int((frame.training_S_rows_same_R0 > 0).sum()),
                'exact_R0_with_any_training_M': int((frame.training_M_rows_same_R0 > 0).sum())}
    repairs = np.flatnonzero(vs & (teacher[fid] != 2) & (e10[fid] == 2))
    correction = pd.DataFrame({'component': r.component.iloc[repairs].to_numpy(),
                               'route': r.route.iloc[repairs].to_numpy()})
    result = {'status': 'audited_after_selection', 'V_S_total': int(vs.sum()),
              'V_S_teacher_errors': int(frame.teacher_error.sum()),
              'V_S_E10_errors': int(frame.e10_error.sum()),
              'V_S_top_error_components': top,
              'empty_known_fact_group': summarize(empty),
              'supported_denial_group': summarize(main),
              'all_V_S': summarize(frame),
              'E10_V_S_repairs': len(repairs),
              'E10_repair_components': correction.component.value_counts().to_dict(),
              'E10_repair_routes': correction.route.value_counts().to_dict(),
              'V_is_A_teacher_unseen_but_historic_v85_teacher_training_exposed': True,
              'caveat': 'Empty known-fact group still has R0 bytes; exact-R0 absence is not proof of indistinguishable semantics or model impossibility.',
              'new_classifier_fits': 0, 'source_sha256': sha(__file__)}
    save(DEST / 'failure_audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
