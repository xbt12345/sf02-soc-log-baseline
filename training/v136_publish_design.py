"""Publish a zero-fit design without changing the latest actual training authority."""
import json
from pathlib import Path
from v135_runtime import ROOT,OUT,read,save,sha,require_run_seal

DEST=ROOT/'artifacts/v136_root_review_20260930'
PLAN=ROOT/'training/review_policy/v136_readout_diagnosis_plan.json'
DOC=ROOT/'docs/V136_ROOT_DIAGNOSIS_AND_TARGETED_TRAINING_PLAN.md'

def main():
    require_run_seal(ROOT/'training/v135_train.py')
    if PLAN.exists():raise FileExistsError('Preserve previous registered design')
    paths=[OUT/'final_delivery.json',OUT/'quality.json',OUT/'learning_qualification.json',OUT/'run_seal.json',
        OUT/'postflight_audit/audit.json',ROOT/'training/review_policy/v135_observed_cases.json',
        DEST/'diagnosis_receipt.json',DEST/'support_receipt.json',DEST/'member_diagnostics.json',
        DEST/'canonical_support_controls.json',DEST/'frozen_gradient_diagnostics.json',
        ROOT/'docs/EXPERIMENT_REVIEW_RULES.md']
    for fold in range(3):paths.append(OUT/f'fold{fold}_R_decay/epoch100_model.pt')
    p={'version':'V136-readout-diagnosis','status':'designed_not_trained','latest_actual_training':'V135',
       'scope':'Official ASA source-closed TRAIN; other task routes frozen; no new external data',
       'new_fits_in_design_turn':0,'new_updates_in_design_turn':0,
       'evidence_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in paths},
       'primary':{'folds':[0,1,2],'fits_max':6,'candidate':'H_L',
           'start_state':'Each matching-fold V135 R_decay epoch100 model',
           'trainable_modules':['head','facts_direct'],'frozen_modules':['first','second'],
           'hidden':128,'members':16,'features':66287,'facts':495,
           'cache':'Full original member h2 and facts, float64 common readout; actual replay required',
           'zero_step_prediction_exact':True,'zero_step_probability_tolerance':2e-6,
           'loss':'Original-row mean independent-member CE','all_original_label_mass':True,
           'pure_class_auxiliary':False,'duplicate_downweight':False,'new_data':False,
           'arms':{'H_A':{'optimizer':'Adam','lr':.002,'weight_decay':0,'full_role_gradient':True,'fixed_updates':200,'full_gradient_evaluations':200},
                   'H_L':{'optimizer':'LBFGS','lr':1,'history_size':20,'line_search':'strong_wolfe','weight_decay':0,'full_role_gradient':True,'full_gradient_evaluations_max':200,'accepted_updates_max':200}},
           'full_role_gradient_evaluations_max':1200,
           'endpoint':'Last accepted state at preregistered solver budget or numerical termination; never score-selected',
           'same_frozen_state_and_loss_across_arms':True,'comparison_is_budgeted_solver_method_not_equal_update_count':True},
       'selection':{'candidate':'H_L','no_other_arm_promotion':True,'freeze_TRAIN_review_before_new_HELD_join':True,
           'HELD_labels_in_fit_or_stopping':False,'score_stopping':False,'posthoc_checkpoint_selection':False,
           'numerical_stop_with_errors_is_unresolved':True,'closed_sources_seen_by_backbone_not_unseen_validation':True},
       'mastery':{'pure_TRAIN_M_errors_max':0,'pure_TRAIN_S_errors_max':0,'full_TRAIN_M_errors_max':0,
           'full_TRAIN_S_errors_max_by_fold':[22,6,28],'last_effective_accepted_states':5,
           'no_op_or_repeated_replay_counts_as_learning_state':False,'panel_only_acceptance':False,
           'all_sparse_correct_controls_included':True},
       'quality':{'full_original_rows':2056871,'classes':[0,1,2],'all_class_precision_recall_f1_A0_protected':True,
           'ASA_M_errors_max':318,'ASA_S_errors_max':2074,'ASA_total_errors_max':2170,'improved_outer_folds_min':2,
           'S_source_mean_recall_min':.1151252713,'S_zero_recall_roots_max':198,
           'S_outside_big3_errors_max':1372,'root2868_M_errors_max':48,'header682_errors_max':0,
           'unknown186_S_errors_max':184,'non_ASA_errors_equal':107,
           'matched_H_A_M_S_both_protected':True,'matched_at_least_one_class_strictly_improves':True},
       'feasibility_probe':{'condition':'H_L still has pure TRAIN errors','solver_runs_max':3,
           'kind':'Original member heads and shared facts_direct affine-margin sufficient-condition feasibility',
           'pure_only_hard_constraints':True,'mixed_original_rows_still_train_and_score':True,
           'all_original_pure_inputs_all_members_two_competitors':True,'margin':1.,'margin_tolerance':1e-6,
           'generation_iterations_max':50,'violating_constraints_added_per_member_max':64,'tie_order':'input_hash',
           'seconds_max_per_fold':600,'whole_population_constraint_replay_required':True,
           'supervised_solver_count_separate':True,'timeout_or_infeasible_does_not_prove_model_impossible':True,
           'probe_solution_may_promote':False},
       'confirmation':{'automatic':False,'requires_new_registration':True,'same_frozen_backbone_new_head_seed_is_independent':False},
       'runtime_readiness':{'trainer_implemented':False,'run_seal_created':False,'old_fixed_epoch_guard_applicable':False,
           'requires_budgeted_solver_guard_and_history_replays':True},
       'known_limits':['Repeatedly observed development source folds, not blind or real-network generalization.',
           'Per-member affine margin feasibility is sufficient but not necessary for correct probability-averaged decisions.',
           'Unregularized frozen-readout diagnostic is not original DFR and may overfit.',
           'Frozen endpoint gradient variability is diagnostic, not a causal attribution.',
           'Future source-validation backbone must be refit without those validation source labels.']}
    save(PLAN,p)
    diagnosis={'status':'zero_fit_actual_model_review_and_design','new_fits':0,'new_updates':0,
        'latest_actual_training':'V135','models_replayed':12,'quality_acceptance':False,'model_promoted':False,
        'direction_doc':DOC.relative_to(ROOT).as_posix(),'plan':PLAN.relative_to(ROOT).as_posix(),
        'runtime_ready':False,'primary_future_fits_max':6,'future_supervised_feasibility_probes_max':3,
        'implementation_issues':[{'stage':'before any result files','problem':'CSR len is ambiguous','resolution':'Use shape[0]'},
            {'stage':'before any result files','problem':'NumPy row/member broadcasting','resolution':'Use explicit row column axis'}],
        'evidence_sha256':p['evidence_sha256'],'plan_sha256':sha(PLAN),'direction_sha256':sha(DOC)}
    save(DEST/'review_delivery.json',diagnosis)
    prefix='当前方向（V136，仅诊断与方案，未训练）：**12个冻结模型重放确认R_decay80次纯TRAIN残错中72次只有少数成员正确；投票使纯S错76→80。残错仅2–4条同类原行/单来源支持，但同样稀缺正确对照126882次。下一轮拟冻结表示、原行CE全量末层Adam/L-BFGS两臂三折最多6拟合；先分离表示与读出求解，再验跨来源。新训练器与封存未实现，本轮0拟合/0更新。** [本质问题、研究反证与针对性训练方案](docs/V136_ROOT_DIAGNOSIS_AND_TARGETED_TRAINING_PLAN.md)。最新实际仍V135质量失败，无模型晋升。\n\n'
    rp=ROOT/'README.md';text=rp.read_text(encoding='utf-8')
    if '当前方向（V136' in text:raise ValueError('Duplicate direction')
    rp.write_text(text.replace('# SOC 日志威胁检测项目\n','# SOC 日志威胁检测项目\n\n'+prefix,1),encoding='utf-8')
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path);project=catalog['project']
    if project['authoritative_delivery_id']!='v135-delivery':raise ValueError('Actual authority changed')
    project['authoritative_direction_id']='v136-review'
    project['current_direction']=['V136本轮0拟合0更新；12个模型重放及canonical支持核查已完成。',
        '下一轮只做冻结表示下六个全量末层求解对照；H_L候选预登记，训练器/执行封存未实现。',
        '正确成员主要为少数，投票已被真实退化反证；不新增S权重或直接联合CE。',
        'TRAIN、表示可读出与来源外质量分开；保留原A0逐类、2868和头部保护。',
        '内来源验证需重新拟合未见该来源的骨干；不把已见表示当独立验证。']
    additions=[('v136-review','V136本质问题诊断、研究反证与下一轮末层对照方案',DOC,'current_direction',
        '零拟合：80次残错72次少数成员正确；投票反例和稀缺正确对照；六拟合末层方案尚未执行。'),
        ('v136-diagnosis','V136十二模型零拟合审查凭据',DEST/'review_delivery.json','diagnostic_evidence',
         '12个冻结实际模型与原行支持复核；0拟合0更新，不改变最新实际V135质量失败。')]
    ids={z['id'] for z in catalog['documents']}
    for id,title,path,category,summary in additions:
        if id in ids:raise ValueError('Duplicate catalog entry')
        catalog['documents'].append({'id':id,'title':title,'path':path.relative_to(ROOT).as_posix(),
            'sha256':sha(path),'category':category,'summary':summary,
            'keywords':['V136','当前','下一轮','ASA','M/S','成员','训练','求解','泛化','方案']})
    save(catalog_path,catalog)
    require_run_seal(ROOT/'training/v135_train.py')
    print(json.dumps({'design_version':'V136','actual_training_authority':'v135-delivery','new_fits':0,'new_updates':0}),flush=True)

if __name__=='__main__':main()
