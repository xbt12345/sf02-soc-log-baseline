"""Label-independent nested role preparation; no model fit or selection."""
import hashlib
import json
import math
import pandas as pd
from v128_no_fit_review import ROOT, OUT, TRACE, sha, write


def group_partition(outer_fold, root):
    s = f'V128|split=12801|outer={outer_fold}|root={root}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'big') % 3


def main():
    target = OUT / 'nested_role_preparation.json'
    if target.exists():
        raise FileExistsError(target)
    d = pd.read_parquet(TRACE, columns=['row_position', 'local', 'root', 'fold', 'truth'])
    frames = []; teachers = []
    for f in range(3):
        legal = d[d.fold != f].copy()
        legal['outer_fold'] = f
        legal['inner_fold'] = legal.root.map(lambda r: group_partition(f, r))
        legal['crossfit_teacher'] = legal.inner_fold
        legal['matched_infit_teacher'] = (legal.inner_fold + 1) % 3
        assert legal.groupby('root').inner_fold.nunique().max() == 1
        assert legal.groupby('local').inner_fold.nunique().max() == 1
        outer_roots = set(d.loc[d.fold == f, 'root'])
        assert not set(legal.root) & outer_roots
        for j in range(3):
            fit = legal[legal.inner_fold != j]
            oof = legal[legal.inner_fold == j]
            isin = legal[legal.matched_infit_teacher == j]
            assert not set(fit.root) & set(oof.root)
            assert set(isin.root) <= set(fit.root)
            assert all((fit.truth == c).any() and (oof.truth == c).any() for c in (1, 2))
            teachers.append({'outer_fold': f, 'teacher_excluded_inner_fold': j,
                'fit_rows': len(fit), 'fit_roots': fit.root.nunique(), 'fit_locals': fit.local.nunique(),
                'fit_M': int((fit.truth == 1).sum()), 'fit_S': int((fit.truth == 2).sum()),
                'OOF_prediction_rows': len(oof), 'OOF_M': int((oof.truth == 1).sum()), 'OOF_S': int((oof.truth == 2).sum()),
                'matched_IS_prediction_rows': len(isin),
                'planned_25epoch_steps': math.ceil(fit.local.nunique()/256)*25})
        frames.append(legal)
    m = pd.concat(frames, ignore_index=True)
    assert len(m) == 225614 and not m.duplicated(['outer_fold', 'row_position']).any()
    m.to_parquet(OUT / 'nested_score_roles.parquet', index=False)
    result = {'status': 'roles_prepared_no_teachers_trained', 'classifier_fits': 0, 'optimizer_steps': 0,
        'split_rule': 'sha256(V128|split=12801|outer={outer_fold}|root={root}) first8bytes big-endian modulo3',
        'label_independent_split': True, 'reroll_seeds': False, 'teacher_count': 9,
        'teachers': teachers, 'teacher_25epoch_steps_total': sum(x['planned_25epoch_steps'] for x in teachers),
        'legal_corrector_row_roles': len(m), 'unique_ASA_rows': m.row_position.nunique(),
        'network_correction_fits': 6, 'constant_control_fits': 6,
        'planned_primary_fits_total': 21, 'corrector_50epoch_steps_total': 17800,
        'preparation_scope': 'Role and class-presence checks only. New trainer, feature replay and runtime source seal still required before parameter updates.',
        'teacher_population_warning': 'Inner teachers have fewer original rows and uneven class/source coverage than the frozen outer base. Matched IS controls share the same teacher bank, but this does not eliminate every train-size or teacher-identity difference.',
        'source_sha256': sha(__file__), 'manifest_sha256': sha(OUT/'nested_score_roles.parquet'),
        'input_sha256': sha(TRACE)}
    write(target, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
