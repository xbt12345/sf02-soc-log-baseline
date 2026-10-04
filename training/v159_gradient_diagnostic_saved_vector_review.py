"""Recompute real diagnostic vectors/costs; zero new model/feature/grad calls."""
import json
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v159_class_direction import class_direction

OUT=ROOT/'artifacts/v159_gradient_diagnostic_saved_vector_review_20261002'
DIAG=ROOT/'artifacts/v159_bound_gradient_repeat_diagnostic_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve(),ROOT/'training/v159_class_direction.py',DIAG/'qualification.json',DIAG/'run_seal.json',DIAG/'actual_calls.jsonl',ROOT/'artifacts/v159_preflight_failure_cost_receipt_20261002/receipt.json']+list(DIAG.glob('repetition*_complete_gradient.npy'))+list(DIAG.glob('repetition*_receipt.json'))
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    (OUT/'pre_saved_vector_bindings.json').write_text(json.dumps(dict(status='before_vector_decode_and_direction_arithmetic_no_model_calls',source_sha256=bound),indent=2)+'\n',encoding='utf-8')
    vectors=[[np.load(DIAG/f'repetition{rep}_class{cls}_complete_gradient.npy') for cls in [1,2]] for rep in range(2)]
    assert all(g.shape==(1060832,) and np.isfinite(g).all() and not np.count_nonzero(g[:-48]) for gg in vectors for g in gg)
    comparison=[]
    for j,cls in enumerate([1,2]):
        a,b=vectors[0][j],vectors[1][j];d=a-b;maximum=float(np.abs(d).max());scale=max(float(np.abs(a).max()),float(np.abs(b).max()))
        comparison.append(dict(class_id=cls,maximum_absolute=maximum,relative_to_global_scale=maximum/scale,relative_L2=float(np.linalg.norm(d)/max(np.linalg.norm(a),np.linalg.norm(b))),cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),changed_components=int(np.count_nonzero(d)),sign_changes=int(np.count_nonzero(np.sign(a)!=np.sign(b))),all_changes_within_last48_output_weight=True,
            eight_float64_eps_times_gradient_scale=8*np.finfo(np.float64).eps*scale,within_eight_eps_scale=maximum<=8*np.finfo(np.float64).eps*scale))
    directions=[]
    for arm in ['A','B']:
        pair=[class_direction(*g,58840,33497,arm) for g in vectors]
        directions.append(dict(arm=arm,statuses=[p['status'] for p in pair],M_weight_difference=abs(pair[0]['M_weight']-pair[1]['M_weight']),maximum_direction_difference=float(np.abs(pair[0]['direction']-pair[1]['direction']).max()),class_slopes=[p['class_slopes'] for p in pair],maximum_unit_step_Armijo_bound_difference=float(np.abs(np.asarray(pair[0]['class_slopes'])-np.asarray(pair[1]['class_slopes'])).max()*1e-4)))
    events=[json.loads(s) for s in (DIAG/'actual_calls.jsonl').read_text().splitlines()]
    diagcounts={k:sum(e['kind']==k and e['event']=='attempt' for e in events) for k in ['head','feature','full_class_gradient']}
    assert diagcounts==dict(head=40,feature=40,full_class_gradient=4) and all(diagcounts[k]==sum(e['kind']==k and e['event']=='completed' for e in events) for k in diagcounts)
    receipts=[read(DIAG/f'repetition{rep}_class{cls}_receipt.json') for rep in range(2) for cls in [1,2]];assert len({r['parameters_sha256'] for r in receipts})==1
    diagnostic=read(DIAG/'qualification.json');prior=read(ROOT/'artifacts/v159_preflight_failure_cost_receipt_20261002/receipt.json')
    summary=dict(status='real_preflight_failed_new_bounded_diagnostic_vectors_recomputed_not_training_qualified',latest_actual_training='V158',original_failure=dict(head_calls=84,opinion_feature_calls=84,full_class_gradients=4,unsaved_gradient_magnitude_unknown=True),technical_diagnostic=dict(head_calls=40,opinion_feature_calls=40,full_class_gradients=4),
        cumulative_actual=dict(head_calls=124,opinion_feature_calls=124,full_class_gradients=8,fits=0,updates=0),remaining_original_total=dict(head_calls=77948,opinion_feature_calls=77948,full_class_gradients=2404),remaining_original_preflight=dict(head_calls=212,full_class_gradients=4),
        class_comparisons=comparison,direction_comparisons=directions,all_parameters_unchanged=True,all_four_probability_and_log_probability_hashes_identical=diagnostic['probability_and_log_probability_all_four_identical'],opinion_outputs_identical_by_chunk=diagnostic['opinion_feature_hashes_each_chunk_match_between_repetitions'],
        cause='not certified; sparse observation projection or reduction paths remain candidates, not a proven installed-wheel cause',new_repeat_policy_active=False,first_training_issue_passed=False,quality_acceptance=False,goal_status='active',
        own_model_calls=0,own_feature_calls=0,own_gradients=0,own_fits=0,own_updates=0,source_bindings_sha256=sha(OUT/'pre_saved_vector_bindings.json'),diagnostic_large_physical_source_seal_sha256=sha(DIAG/'run_seal.json'))
    check_bindings(bound);(OUT/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
