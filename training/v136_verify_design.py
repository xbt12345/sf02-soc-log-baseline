"""Evidence and adversarial design preflight; deliberately not a training runtime."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v135_runtime import ROOT,OUT,read,save,sha,require_run_seal,load_data,fit_context

DEST=ROOT/'artifacts/v136_root_review_20260930'
PLAN=ROOT/'training/review_policy/v136_readout_diagnosis_plan.json'

def need(value,message):
    if not value:raise ValueError(message)

def validate(p):
    need(p['status']=='designed_not_trained' and p['latest_actual_training']=='V135','Design cannot be training evidence')
    need(p['new_fits_in_design_turn']==p['new_updates_in_design_turn']==0,'Unreported training')
    r=p['primary'];s=p['selection'];q=p['quality'];m=p['mastery'];f=p['feasibility_probe']
    need(r['fits_max']==6 and r['folds']==[0,1,2] and r['candidate']==s['candidate']=='H_L','Posthoc candidate or missing role')
    need(r['trainable_modules']==['head','facts_direct'] and r['frozen_modules']==['first','second'],'Scope confounded')
    need(r['features']==66287 and r['members']==16 and r['hidden']==128,'Input or capacity changed')
    need(r['loss']=='Original-row mean independent-member CE' and r['all_original_label_mass'],'Wrong objective or mass')
    need(not r['pure_class_auxiliary'] and not r['duplicate_downweight'] and not r['new_data'],'Unregistered supervision')
    need(set(r['arms'])=={'H_A','H_L'} and all(z['full_role_gradient'] and z['weight_decay']==0 for z in r['arms'].values()),'Mismatched gradient/objective')
    need(r['arms']['H_A']['full_gradient_evaluations']==200 and r['arms']['H_L']['full_gradient_evaluations_max']==200 and r['full_role_gradient_evaluations_max']==1200,'Unbounded budget')
    need(s['no_other_arm_promotion'] and s['freeze_TRAIN_review_before_new_HELD_join'],'Posthoc selection')
    need(not s['HELD_labels_in_fit_or_stopping'] and not s['score_stopping'] and not s['posthoc_checkpoint_selection'],'Held or score stopping')
    need(s['numerical_stop_with_errors_is_unresolved'] and s['closed_sources_seen_by_backbone_not_unseen_validation'],'False learning or validation identity')
    need(m['pure_TRAIN_M_errors_max']==m['pure_TRAIN_S_errors_max']==m['full_TRAIN_M_errors_max']==0 and m['full_TRAIN_S_errors_max_by_fold']==[22,6,28],'Weakened classification acceptance')
    need(m['last_effective_accepted_states']==5 and not m['no_op_or_repeated_replay_counts_as_learning_state'] and not m['panel_only_acceptance'],'Artificial stable window')
    need(q['full_original_rows']==2056871 and q['classes']==[0,1,2] and q['all_class_precision_recall_f1_A0_protected'],'Partial quality')
    need((q['ASA_M_errors_max'],q['ASA_S_errors_max'],q['ASA_total_errors_max'])==(318,2074,2170),'Changing quality after failure')
    need(q['root2868_M_errors_max']==48 and q['header682_errors_max']==0 and q['non_ASA_errors_equal']==107,'Lost protected cases')
    need(q['matched_H_A_M_S_both_protected'] and q['matched_at_least_one_class_strictly_improves'],'Class tradeoff accepted')
    need(f['whole_population_constraint_replay_required'] and f['supervised_solver_count_separate'] and not f['probe_solution_may_promote'],'False capacity success')
    need(f['timeout_or_infeasible_does_not_prove_model_impossible'],'Sufficient condition mistaken for necessity')
    need(not p['confirmation']['automatic'] and not p['confirmation']['same_frozen_backbone_new_head_seed_is_independent'],'False independent confirmation')
    need(not p['runtime_readiness']['trainer_implemented'] and not p['runtime_readiness']['run_seal_created'],'False runtime readiness')

def adversaries(p):
    edits=[('correct member oracle promotion',lambda v:v['selection'].update(no_other_arm_promotion=False)),
        ('early score stopping',lambda v:v['selection'].update(score_stopping=True)),
        ('train-seen inner validation',lambda v:v['selection'].update(closed_sources_seen_by_backbone_not_unseen_validation=False)),
        ('change weight instead of solver',lambda v:v['primary'].update(pure_class_auxiliary=True)),
        ('drop original duplicate frequency',lambda v:v['primary'].update(duplicate_downweight=True)),
        ('no-op convergence window',lambda v:v['mastery'].update(no_op_or_repeated_replay_counts_as_learning_state=True)),
        ('solver tiny gradient called learning',lambda v:v['selection'].update(numerical_stop_with_errors_is_unresolved=False)),
        ('new M loss hidden by S gain',lambda v:v['quality'].update(ASA_M_errors_max=1748)),
        ('partial task scoring',lambda v:v['quality'].update(full_original_rows=112807)),
        ('LP budget called impossibility',lambda v:v['feasibility_probe'].update(timeout_or_infeasible_does_not_prove_model_impossible=False)),
        ('same backbone seed confirmation',lambda v:v['confirmation'].update(same_frozen_backbone_new_head_seed_is_independent=True)),
        ('design pretends executable',lambda v:v['runtime_readiness'].update(trainer_implemented=True))]
    passed=[]
    for name,edit in edits:
        v=copy.deepcopy(p);edit(v)
        try:validate(v)
        except ValueError:passed.append({'case':name,'rejected':True})
        else:raise AssertionError('Invalid design accepted '+name)
    return passed

def main():
    p=read(PLAN);validate(p);require_run_seal(ROOT/'training/v135_train.py')
    for rel,h in p['evidence_sha256'].items():need(sha(ROOT/rel)==h,'Changed evidence '+rel)
    for name in ['diagnosis_receipt.json','support_receipt.json']:
        z=read(DEST/name)
        for rel,h in z.get('source_sha256',{}).items():need(sha(ROOT/rel)==h,'Changed source '+rel)
        for file,h in z['output_sha256'].items():need(sha(DEST/file)==h,'Changed output '+file)
    x,d=load_data();summaries=pd.read_json(DEST/'member_diagnostics.json');replays=[];aggregate={}
    for fold in range(3):
        fit,c,pure,_,ids=fit_context(d,fold);y=fit.truth.to_numpy();ii=fit.local.to_numpy();pr=pure[ii].astype(bool)
        for arm in ['R_const','R_decay','O_const','O_decay']:
            cached=np.load(DEST/f'fold{fold}_{arm}_frozen.npz');q=cached['mean_probability'];mp=cached['member_probability'];pred=mp.argmax(-1)
            need(np.array_equal(pred,cached['member_pred']),'Member logits/probabilities disagree')
            need(float(np.abs(mp.mean(1)-q).max())<2e-6,'Wrong mean probability')
            saved=np.load(OUT/f'fold{fold}_{arm}/sealed_all_prob.npy')
            need(float(np.abs(saved-q).max())<2e-6 and np.array_equal(saved.argmax(1),q.argmax(1)),'Actual frozen versus sealed all-input replay')
            votes=np.stack([(pred==cl).sum(1) for cl in range(3)],1);vp=votes.argmax(1)[ii];qp=q[ii].argmax(1)
            for cl in [1,2]:
                m=pr&(y==cl);own=summaries[(summaries.fold==fold)&(summaries.arm==arm)&(summaries['class']==cl)&(summaries.status=='all')&(summaries.population=='TRAIN_pure')].iloc[0]
                actual=[int((m&(qp!=y)).sum()),int((m&(vp!=y)).sum()),int((m&(qp==y)&(vp!=y)).sum())]
                need(actual==[own.probability_mean_errors,own.vote_errors,own.vote_regressions_vs_mean],'Wrong original-row vote diagnosis')
                key=(arm,cl);aggregate.setdefault(key,np.zeros(3,dtype=np.int64));aggregate[key]+=actual
            replays.append({'fold':fold,'arm':arm,'all_inputs':len(q),'fit_original_role_rows':len(fit),'sealed_predictions_match':True})
    need(aggregate[('R_decay',2)].tolist()==[76,80,10],'Actual voting counterexample changed')
    ctrl=read(DEST/'canonical_support_controls.json');r=ctrl['residuals']['R_decay']
    need((r['original_role_rows'],r['independent_original_rows'],r['canonical_inputs'])==(80,56,27),'Residual identity changed')
    need(sum(z['sparse_single_source_correct'] for z in ctrl['TRAIN_controls'])==126882,'Correct sparse controls omitted')
    need(sum(z['other_errors'] for z in ctrl['TRAIN_controls'])==0,'Sparse-support claim not supported')
    report={'status':'design_and_actual_evidence_preflight_passed','new_fits':0,'new_updates':0,
        'quality_passed':False,'trainer_ready':False,'actual_training_version':'V135',
        'candidate_next_design':'H_L','actual_data_counterexamples':['R_decay pure S vote 76 to 80 with 10 regressions',
            'Sparse support also has 126882 correct TRAIN-role original rows','V135 original M loss despite S gain'],
        'adversarial_design_rejections':adversaries(p),'cached_models_vs_sealed_all_input_replays':replays,
        'plan_sha256':sha(PLAN),'source_sha256':sha(Path(__file__)),
        'limits':'Preflight is not a training runtime seal, optimizer correctness, model quality or blind validation.'}
    target=DEST/'design_preflight.json'
    if target.exists():raise FileExistsError(target)
    save(target,report);print(json.dumps({'invalid_designs_rejected':12,'actual_model_output_replays':12,'new_fits':0,'new_updates':0}),flush=True)

if __name__=='__main__':main()
