"""Freeze the V-only choice before opening locked development roles."""
import json
from run_v75 import ROOT, read, save, sha
from v95_prepare import DEST


def main():
    assert not (DEST / 'selection.json').exists()
    reg = read(DEST / 'registration.json')
    teacher = read(DEST / 'teacher_fit.json')
    assert teacher['actual_classifier_fits'] == 1
    trace = []
    initial = []
    for arm in reg['branch']['arms']:
        fit = read(DEST / f'{arm}_fit.json')
        assert fit['actual_classifier_fits'] == 1 and fit['steps_executed'] == 200
        assert fit['source_sha256'] == sha(ROOT / 'training/v95_four_arm.py')
        assert fit['teacher_model_sha256'] == teacher['model_sha256']
        initial.append(fit['initial_state_sha256'])
        snaps = read(DEST / f'{arm}_snapshots.json')
        assert [s['step'] for s in snaps] == [0, 25, 50, 100, 200]
        assert all(sha(DEST / f"{arm}_step{s['step']:03}_model.pt") == s['model_sha256'] for s in snaps)
        assert all(sha(DEST / f"{arm}_step{s['step']:03}_prediction.npy") == s['prediction_sha256'] for s in snaps)
        trace.extend(snaps)
    assert len(set(initial)) == 1
    candidates = [s for s in trace if s['V_eligibility']['eligible']]
    selected = min(candidates, key=lambda s: (s['roles']['V']['errors'], s['step'], s['arm'])) if candidates else None
    report = {'status': 'V_selection_frozen_before_inner_C_H',
              'criterion': reg['selection'],
              'actual_classifier_fits': 5, 'actual_calibration_fits': 0,
              'all_arm_initial_state_equal': True,
              'V_selected': None if selected is None else {'arm': selected['arm'], 'step': selected['step'],
                                      'errors': selected['roles']['V']['errors'],
                                      'positive_flips': selected['roles']['V']['positive_flips'],
                                      'negative_flips': selected['roles']['V']['negative_flips'],
                                      'M_correct': selected['roles']['V']['correct'][1],
                                      'S_correct': selected['roles']['V']['correct'][2]},
              'eligible_snapshots': [{'arm': s['arm'], 'step': s['step'], 'V_errors': s['roles']['V']['errors'],
                                       'V_positive': s['roles']['V']['positive_flips'],
                                       'V_negative': s['roles']['V']['negative_flips']}
                                      for s in candidates],
              'inner_C_H_used_for_selection': False,
              'model_promoted': False,
              'quality_acceptance': False}
    save(DEST / 'selection.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
