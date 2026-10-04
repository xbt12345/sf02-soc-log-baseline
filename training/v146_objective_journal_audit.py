"""Post-fit reconstruction of every sealed V146 proposal; no model fitting."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v146_guarded_pair_training_20261001'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    delivery = read(OUT / 'final_delivery.json')
    reports, joined = [], []
    for fold in range(3):
        fixed = read(OUT / f'fold{fold}_fixed_lambda.json')
        for arm in ['A', 'B']:
            folder = OUT / f'fold{fold}_{arm}'
            receipt = read(folder / 'fit.json')
            started = read(folder / 'started.json')
            assert receipt['lambda_fixed'] == started['lambda_fixed'] == fixed['coefficient']
            lam = fixed['coefficient'] if arm == 'B' else 0.0
            gradients = [json.loads(s) for s in (folder / 'gradients.jsonl').read_text().splitlines()]
            proposals = [json.loads(s) for s in (folder / 'proposals.jsonl').read_text().splitlines()]
            assert gradients[0]['parameter_sha256'] == started['initial_parameter_sha256']
            previous_hash = started['initial_parameter_sha256']
            previous_classes = None
            class_rises = [0, 0, 0]
            rejected_guard = rejected_objective = 0
            gap = 0.0
            for g in gradients:
                assert g['parameter_sha256'] == previous_hash
                reconstructed = g['original_member_CE'] + lam * g['auxiliary_loss']
                gap = max(gap, abs(reconstructed - g['objective']))
                assert abs(reconstructed - g['objective']) < 1e-12
                group = [z for z in proposals if z['gradient'] == g['full_gradient_evaluation']]
                accepted = [z for z in group if z['accepted']]
                assert len(accepted) <= 1 and (not accepted or group[-1]['accepted'])
                for z in group:
                    rebuilt = z['original_member_CE'] + lam * z['auxiliary_loss']
                    gap = max(gap, abs(rebuilt - z['objective']))
                    assert math.isfinite(rebuilt) and abs(rebuilt - z['objective']) < 1e-12
                    bound = g['objective'] - 1e-4 * z['step'] * g['gradient_norm']
                    assert abs(bound - z['Armijo_bound']) < 1e-12
                    expected = z['classification_guard'] and rebuilt < g['objective'] and rebuilt <= bound
                    assert z['accepted'] == expected
                    assert z['original_class_mass_seen'] == receipt['original_class_mass']
                    if not z['accepted']:
                        rejected_guard += not z['classification_guard']
                        rejected_objective += not (rebuilt < g['objective'] and rebuilt <= bound)
                    joined.append(dict(fold=fold, arm=arm, lambda_effective=lam,
                        registered_aux_coefficient=fixed['coefficient'],
                        lambda_binding=f'fold{fold}_fixed_lambda.json', base_objective=g['objective'],
                        reconstructed_objective=rebuilt, **z))
                if accepted:
                    z = accepted[0]
                    assert z['stats']['mastered'] and z['stats']['new_errors_vs_start'] == 0
                    classes = z['per_class_member_CE']
                    if previous_classes is not None:
                        for c in [1, 2]:
                            class_rises[c] += classes[c] > previous_classes[c]
                    previous_classes = classes
                    previous_hash = z['parameter_sha256']
            assert len(proposals) == receipt['proposal_evaluations']
            assert sum(z['accepted'] for z in proposals) == receipt['accepted_updates']
            assert previous_hash == read(folder / 'progress.json')[-1]['parameter_sha256']
            reports.append(dict(fold=fold, arm=arm, proposals=len(proposals),
                accepted=receipt['accepted_updates'], rejected_by_classification_guard=rejected_guard,
                rejected_by_objective=rejected_objective, objective_reconstruction_max_gap=gap,
                class_CE_rise_between_accepted_states=class_rises,
                classification_new_errors=receipt['endpoint_stats']['new_errors_vs_start']))
    with (OUT / 'all_proposals_with_bound_lambda.jsonl').open('w', encoding='utf-8') as stream:
        for z in joined:
            stream.write(json.dumps(z, ensure_ascii=False) + '\n')
    result = dict(status='all_actual_proposals_objective_and_acceptance_reconstructed',
        fits=6, new_fits=0, new_updates=0, proposals=len(joined),
        accepted=sum(z['accepted'] for z in reports), reports=reports,
        paired_effect_passed=delivery['matched_effect_passed'],
        quality_acceptance=delivery['quality_acceptance'],
        original_journals_changed=False,
        scope='Fixed lambda joined from pre-update sealed file; original journals and models preserved.')
    (OUT / 'objective_journal_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
