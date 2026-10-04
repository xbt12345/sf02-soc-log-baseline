"""Zero-fit single-issue design and real retention-compatibility audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from v135_runtime import ROOT,OUT,read,save,sha,require_run_seal,load_data
from v125_input import normalize_body

DEST=ROOT/'artifacts/v137_single_issue_design_20260930'
PLAN=ROOT/'training/review_policy/v137_single_issue_plan.json'
DOC=ROOT/'docs/V137_SINGLE_ISSUE_TRAINING_AND_RETENTION.md'

def main():
    require_run_seal(ROOT/'training/v135_train.py')
    if DEST.exists() or PLAN.exists() or DOC.exists():raise FileExistsError('Preserve prior design')
    DEST.mkdir();x,d=load_data()
    ledger_path=OUT/'ASA_prediction_ledger.parquet';a=pd.read_parquet(ledger_path)
    a=a.merge(d[['row_position','canonical_key']],on='row_position',validate='one_to_one')
    kept=a[a.pred_A0.eq(a.truth)]
    labels=kept.groupby(['fold','canonical_key']).truth.nunique()
    conflict_keys=labels[labels>1].reset_index()[['fold','canonical_key']]
    rows=a.merge(conflict_keys,on=['fold','canonical_key'],how='inner',validate='many_to_one')
    raw_path=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    raw=pd.read_parquet(raw_path,columns=['row_position','raw_message','facts_json'])
    rows=rows.merge(raw,on='row_position',validate='one_to_one');summaries=[]
    for (fold,key),part in rows.groupby(['fold','canonical_key']):
        locs=part.local.unique()
        if any((x[locs[0]]!=x[i]).nnz for i in locs[1:]):raise ValueError('Canonical key does not represent identical model inputs')
        mass=part.truth.value_counts();correct=part[part.pred_A0.eq(part.truth)].truth.value_counts()
        total=int(mass.sum());max_correct=int(mass.max());old_correct=int(correct.sum())
        summaries.append({'fold':int(fold),'canonical_key':key,'original_rows':total,
            'original_class_mass':{str(k):int(v) for k,v in mass.items()},
            'A0_correct_original_rows':old_correct,'A0_correct_class_mass':{str(k):int(v) for k,v in correct.items()},
            'identical_input_classifier_max_correct_rows':max_correct,
            'minimum_errors_current_input':total-max_correct,
            'minimum_regressions_on_A0_correct_rows':old_correct-int(correct.max()),
            'normalized_body_variants':int(part.raw_message.map(lambda t:normalize_body(t)[0]).nunique()),
            'parsed_fact_variants':int(part.facts_json.nunique()),
            'limits':'Identical current features do not prove full official fields or legitimate contextual behavior are indistinguishable.'})
    if len(summaries)!=1 or summaries[0]['original_class_mass']!={'1':72,'2':6} or summaries[0]['A0_correct_original_rows']!=74:raise ValueError('Retention counterexample changed')
    export=rows.drop(columns=['raw_message','facts_json']).copy()
    export['normalized_body_sha256']=rows.raw_message.map(lambda t:__import__('hashlib').sha256(normalize_body(t)[0].encode()).hexdigest())
    export['facts_sha256']=rows.facts_json.map(lambda t:__import__('hashlib').sha256(t.encode()).hexdigest())
    export.to_parquet(DEST/'retention_collision_original_rows.parquet',index=False)
    save(DEST/'retention_compatibility_audit.json',{'status':'actual_input_retention_counterexample_checked','new_fits':0,'new_updates':0,
        'counterexamples':summaries,'do_not_restore_clock_or_identity_shortcut':True,'labels_unchanged':True,
        'no_exception_rows_removed_from_primary_score':True,
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ledger_path,raw_path,ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz']}})
    evidence=[OUT/'final_delivery.json',OUT/'run_seal.json',OUT/'quality.json',
        ROOT/'training/review_policy/v136_readout_diagnosis_plan.json',
        ROOT/'artifacts/v136_root_review_20260930/completion_receipt.json',
        ROOT/'artifacts/v136_root_review_20260930/canonical_support_controls.json',
        DEST/'retention_compatibility_audit.json',DEST/'retention_collision_original_rows.parquet']
    original=read(ROOT/'training/review_policy/v136_readout_diagnosis_plan.json')
    plan={'version':'V137-single-issue-control','status':'designed_not_trained','latest_actual_training':'V135',
        'new_fits':0,'new_updates':0,'primary_issue':{'id':'TRAIN-PURE-READOUT','question':'Current representation: all pure TRAIN M/S classification, not transfer or input reconstruction',
        'baseline_pure_role_errors':80,'baseline_independent_original_error_rows':56,'baseline_canonical_error_inputs':27,
        'goal_pure_M_errors':0,'goal_pure_S_errors':0,'maximum_experiment_rounds':2,
        'cannot_guarantee_success_by_round_two':True,'round_counter_may_reset_by_version_rename':False,
        'scope_may_expand_inside_round':False},
        'round1':{'registered_fits_max':6,'parent_plan_path':'training/review_policy/v136_readout_diagnosis_plan.json',
            'parent_plan_sha256':sha(ROOT/'training/review_policy/v136_readout_diagnosis_plan.json'),
            'candidate':'H_L','method_only_factor':'Frozen-readout solver Adam versus LBFGS',
            'primary':original['primary'],'mastery':original['mastery'],
            'conditional_feasibility_probe':original['feasibility_probe'],
            'acceptance_source_label_purity':'Legal TRAIN only; independent official row truth',
            'no_inputs_or_class_weights_or_backbone_changes':True},
        'round2':{'automatic':False,'needs_new_bound_runtime_and_specific_round1_evidence':True,'fits_max':6,
            'candidate':'C_margin','single_factor':'Add legal TRAIN classification-margin constraints to the same frozen readout',
            'eligibility':'Round1 full-current-readout feasibility certificate plus observed optimizer/risk failure; whole TRAIN majority decisions feasible before fit',
            'constraints':'Every pure TRAIN input true class; mixed TRAIN input strict original-mass majority; all members margin >=1',
            'all_original_true_label_CE_retained':True,'mixed_labels_not_rewritten':True,'HELD_labels_in_constraints':False,
            'full_gradient_evaluations_max_per_fit':200,'supervised_solver_seconds_max_per_fold':600,
            'feasibility_all_original_training_inputs_replay':True,'probe_inconclusive_disallows_second_round':True,
            'new_target_or_trainable_scope_disallows_second_round':True,
            'no_posthoc_threshold_seed_or_fold_routing':True},
        'problem_budget':{'rounds_max':2,'primary_fits_max':12,'new_initialization_confirmation_fits_in_this_issue':0,
            'solver_probes_report_separately':True,'after_second_failure':'Unresolved; stop this path and preserve evidence. A new attempt requires independently justified mechanism, not a renamed repeat.'},
        'retention':{'registry_path':'training/review_policy/v137_verified_repair_registry.json',
            'closed_issue_registry_has_performance_success':False,'frozen_baseline_kept':True,
            'accepted_repair_original_rows_and_truth_immutable':True,'accepted_repair_new_errors_max':0,
            'unaccepted_experiment_gains_are_not_closed_repairs':True,
            'A0_correct_regressions_all_reported':True,'aggregate_counts_cannot_hide_repair_regression':True,
            'HELD_repair_rows_readonly_never_gradient_or_threshold_targets':True,
            'conflicting_guard_population_must_block_adoption_not_drop_rows':True,
            'training_repair_without_transfer_is_scoped_branch_only':True,
            'adoption_requires_original_full_quality':True},
        'task_adoption_quality':original['quality'],'scope_acceptance_is_not_task_promotion':True,
        'runtime_readiness':{'trainer_implemented':False,'run_seal_created':False,
            'requires_guard_replay_before_fit_and_adoption':True},
        'evidence_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in evidence}}
    save(PLAN,plan)
    registry={'version':'V137-verified-repair-registry','classified_case_repair_sets':[],
        'interpretation':'No current classification candidate has passed full task adoption. Verified engineering contracts are separate.',
        'engineering_contracts':[
            {'id':'DATA-ORIGINAL-MASS','status':'verified_in_V135','scope':'Official 2056871-row identity/truth and all TRAIN-role original label multiplicity',
             'evidence':'artifacts/v135_stable_learning_trial_20260930/verification.json',
             'on_change':'Recompute independent official truth, role row identity and per-class mass; no fitting or adoption on failure'},
            {'id':'NUM-ACTIVE-DENSE','status':'verified_in_V135_observed_scope','scope':'Actual input repeated forward/backward panels and matched first80 epoch trajectories on registered host/configuration',
             'evidence':'artifacts/v135_stable_learning_trial_20260930/postflight_audit/audit.json',
             'on_change':'Freeze existing dense kernel; changed imported compute code/configuration requires actual repeated forward and gradient/prefix replay'},
            {'id':'HEADER-CLOCK-INVARIANCE','status':'input_contract_verified_classification_unresolved','scope':'Registered ASA header transformations and 682 affected original rows',
             'evidence':'artifacts/v124_header_trial_20260929/input_preflight.json',
             'on_change':'Replay actual original rows and registered clock-format variants; preserve event body/facts and all score rows; no clock or identity shortcut'}],
        'open_issues':['TRAIN-PURE-READOUT','CROSS-SOURCE-M-S','RETENTION-INPUT-COLLISION'],
        'future_repair_closeout':{'requires':'Actual model replay of full issue population, old correct controls, held/reference regression audit and scoped acceptance',
            'store':['model_checkpoint','complete_source_data_role_bindings','original_row_ids','independent_truth','predictions','counterexamples','valid_scope','rollback_baseline'],
            'immutable_accepted_guard_rows':True,'later_new_errors_on_accepted_rows_max':0,
            'held_labels_in_future_training':False,'failing_candidate_may_replace_baseline':False,
            'scope_failure_is_not_hidden_by_global_metric':True}}
    for e in registry['engineering_contracts']:e['evidence_sha256']=sha(ROOT/e['evidence'])
    registry_path=ROOT/'training/review_policy/v137_verified_repair_registry.json';save(registry_path,registry)
    document='''# V137：一次只解决训练侧分类缺口，最多两轮，修复成果分范围保护

2026-09-30。方案优化与只读真实数据审查，新增拟合0次、参数更新0次。最新实际训练仍V135：12次拟合、71,200步，训练与来源外质量失败，无模型晋升。V136、V135原方案和证据均保留，本方案取代当前推进顺序，没有启动训练。

## 结论及范围

下一轮只处理一个问题：**当前表示下，全部无混标TRAIN输入仍未被稳定正确分类**。现有缺口为80次训练角色原行错误（4M/76S），涉及56条不同官方原行、27种实际数值输入。不能把“修这27种”变成只训练/只计分这27种；全部合法TRAIN和原正确对照必须继续参与。

本问题最多两轮：第一轮完成读出求解对照并给出确定结论；第二轮仅在第一轮取得具体可表达性证据时修同一个求解缺口。不能保证未知问题在两轮内必然解决，但可以保证不通过换版本名称、再加轮数、再加因素无限重复。两轮仍失败就明确记录未解决，停止此路径，而非降低验收条件。

这两轮不同时处理来源泛化、扩大网络、改端口编码、补新数据或修所有格式。跨来源仍做完整审查，是风险检测与晋升条件，不是同轮拟合目标。

## 必须修正“保护成果”的含义

重新核查发现一个真实障碍：当前完全相同的数值输入对应78条官方原行，标签72M/6S，原A0判对74条（72M/2S）。同输入的确定性分类器最多判对72条，最少6错；若强行要求保留A0所有74条正确判断，至少两条无法同时保持。原始标准化正文和既有解析事实各只有一种。

这不是已经证明官方所有可用字段或上下文完全相同；也不能直接说标签错误。原文仍含日期、IP和脱敏形式差异，其真实行为意义尚未证实。不得重新注入时间外观、身份编码或行ID来取得漂亮分数；不得按留出标签路由；全部78条继续完整计分。此审查仅用于排除不可能的保护承诺，不用于调整训练权重。

因此，**历史上一次判对不等于已验收的可靠修复**。目前没有通过完整任务晋升的分类修复集，不能虚构已有成功。但官方数据/原行质量核查、实际确定性计算路径、已登记的时间头编码不变性，可以作为各自范围内已验证的工程成果保护。时间头分类仍有错误，不能因为输入不变性通过就关闭分类问题。

## 第一轮：只隔离末层求解方式

保留V136的最小六拟合设计：三折，各有H_A全量Adam与H_L全量L-BFGS，唯一候选为H_L。缓存完整16成员的128维表示和495个事实，冻结first/second，只更新原head及facts_direct。保留66,287维输入、官方标签和全部原始行质量；共同使用原行成员CE，不增加类别辅助或重复降权。

每臂每折最多200次全角色梯度评估；Adam固定200次更新，L-BFGS实际闭包与已接受更新分别计数。最高6次末层拟合、1200次全角色梯度评估。缓存/精度转换零步判决须逐原行与基座相同，概率差≤2e-6；否则暂停拟合先处理实现问题。不能按结果增加预算或挑早期状态。

训练侧验收：三折全部纯M/S原行零错；完整M零错、S只剩表示混标最低22/6/28；全部旧正确纯输入不退化。末五个实际不同的已接受状态均通过，重复评估同一状态不充作五次学习。面板、损失或总体正确率不能代替全人口分类验收。

第一轮如果H_L仍错，运行原读出margin可行性诊断，最多三折各一次监督求解，保留V136的50次约束生成/每折600秒上限。必须实际重放全人口约束；超时或约束过强导致失败都不证明模型不可解。这项诊断只服务当前训练分类问题，不作为第二个优化目标。

## 第二轮：只有具体证据才执行

仅当第一轮得到“固定表示与原读出存在满足分类约束的解，但实际风险/求解没有达到”的真实证据，第二轮才允许加入**合法TRAIN分类margin约束**，其余表示、参数范围、人口和标签保持不变。

约束覆盖每个纯输入的真值类别，以及混标输入的严格原行多数类别；每成员与两个竞争类margin至少1。混标S/M原始标签仍全部参与原行CE，不改成多数标签，不删除。第二轮前必须验证包含混标多数决策的完整训练约束可行；只证明纯输入可行不足以自动启动第二轮。

第二轮最多6次主拟合（同范围无约束匹配参照和C_margin候选各三折），每拟合最多200次全角色梯度评估，每折监督求解累计不超过600秒；具体有界凸求解器、实际闭包/更新限制和候选终点须在新入口运行前登记并封存。不得根据HELD分数选求解器、权重、阈值或种子。若无法实现可靠约束求解、可行性未知或需要换表示/扩网络，不执行第二轮，直接报告当前路径的证据缺口。

即使约束强制训练分类全对，也可能只是在记忆当前人口，不说明学会真实安全语义。第二轮候选仍必须接受来源外完整审查。

## 修复的留存与后续保护

修复记录保留模型、完整原行ID与独立真值、预测、原正确对照、计算/输入/折依赖、反例、适用范围与回滚基线。分类保护集以验收时的原始行身份固定，不能在下一轮按新编码重新挑一批容易样本，不能删除后来出现的冲突行来保绿。

每次改动先重放适用的已验证工程契约；失败即不进入拟合。每次候选替换前重放已验收分类修复集，新错必须为0；哪怕修复了更多别的错误或总F1上升，也不能掩盖已解决问题的退化。训练角色修复集只能证明训练范围；来源留出修复集只能用于只读验收，不加入梯度、阈值拟合或按行纠错。

如果仅训练侧通过、来源外失败，状态是“训练侧修复已验证，迁移未通过”：隔离保存该分支和证据，现有参照不替换，后续方法必须保留这项训练能力。完整任务门槛继续按原A0逐类及来源条件，不能把诊断成功升级为比赛模型成功。

不以“冻结了模块”代替回归验证：head改变仍可能改变全人口判决。未通过验收的历史实验收益继续作为比较证据，不自动写进已解决注册表。保护集若在当前可观察输入上产生矛盾，阻断候选晋升，核查信息来源；不删样本、不承诺永久未知数据正确。

## 一轮必须交付的账本与决定

只交付当前问题的修复/未修复、全部原正确记录新增错误、各类分母与判对/漏判/误报、来源外逐类变化、实际拟合/更新/闭包成本、适用范围和下一步判定。注册“未解决、部分改善、范围内已解决、全任务可替换”四种不同状态；不能把低损失或初次判对称为完全解决。

原任务晋升门槛保持：全部2,056,871行，三类precision/recall/F1保护；ASA M≤318、S≤2074、总错≤2170；至少两折改善及来源/未知参数/头部/2868保护。任何新错都分修复集回归和未验收参照回归报告；后者不能直接当可靠修复成果，但全部保留真实计分和原质量门槛。

本轮只更新计划、保护登记和真实反例审查；新训练器与运行封存尚未实现。下一轮不需要先扩充平台资源，先做可在本地验证的单问题末层试验。
'''
    DOC.write_text(document,encoding='utf-8')
    save(DEST/'delivery.json',{'status':'zero_fit_single_issue_design','new_fits':0,'new_updates':0,'quality_acceptance':False,
        'latest_actual_training':'V135','model_promoted':False,'runtime_ready':False,
        'single_issue':'TRAIN-PURE-READOUT','maximum_rounds':2,'currently_accepted_classification_repairs':0,
        'retention_counterexample':summaries[0],'plan_sha256':sha(PLAN),'direction_sha256':sha(DOC),
        'registry_sha256':sha(registry_path),'source_sha256':sha(Path(__file__))})
    rp=ROOT/'README.md';text=rp.read_text(encoding='utf-8');prefix='当前方向（V137，未训练）：**下一轮只解决80次纯TRAIN残错（56条原行、27种输入），三折末层Adam/L-BFGS六拟合；同问题最多两轮，第二轮须有明确可表达性证据。已验收修复集以后新增错为0；未验收实验收益不冒充已解决。真实78行同输入72M/6S冲突说明不能承诺保留所有历史判对。最新实际仍V135质量失败，本轮0拟合0更新。** [单问题训练、两轮预算及成果回归保护](docs/V137_SINGLE_ISSUE_TRAINING_AND_RETENTION.md)。\n\n'
    rp.write_text(text.replace('# SOC 日志威胁检测项目\n','# SOC 日志威胁检测项目\n\n'+prefix,1),encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';catalog=read(cp)
    if catalog['project']['authoritative_delivery_id']!='v135-delivery':raise ValueError('Actual authority changed')
    catalog['project']['authoritative_direction_id']='v137-review'
    catalog['project']['current_direction']=['V137本轮0拟合0更新；下一轮只处理TRAIN-PURE-READOUT，最多两轮。',
        '第一轮沿用V136六个末层对照；不同时改输入、类别权重或骨干。',
        '第二轮须有完整合法训练分类约束可行证据，否则停止此路径。',
        '已验收工程/分类修复分范围保护，新错拒绝晋升；现无通过全任务验收的分类修复集。',
        '78行同输入含72M/6S，历史74条判对不能同时保持；保留全部计分，不恢复时间/身份捷径。']
    for id,title,path,category in [('v137-review','V137单问题训练及成果保持方案',DOC,'current_direction'),
        ('v137-design-evidence','V137零拟合方案与真实保护冲突证据',DEST/'delivery.json','diagnostic_evidence')]:
        if any(e['id']==id for e in catalog['documents']):raise ValueError('Duplicate authority entry')
        catalog['documents'].append({'id':id,'title':title,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
            'category':category,'summary':'最多两轮解决单一训练分类问题，成果分范围回归保护；0拟合0更新，最新实际仍V135失败。',
            'keywords':['V137','当前','下一轮','单问题','训练','成果','回归','两轮','ASA','M/S']})
    save(cp,catalog);require_run_seal(ROOT/'training/v135_train.py')
    print(json.dumps({'single_issue':'TRAIN-PURE-READOUT','rounds_max':2,'retention_collision_rows':len(rows),'new_fits':0,'new_updates':0}),flush=True)

if __name__=='__main__':main()
