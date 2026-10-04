"""Create immutable result/report/case records after actual six-model replay."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings

TRIAL=ROOT/'artifacts/v158_fusion_trial_20261001'
BASE=ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001'
AUDIT=ROOT/'artifacts/v158_saved_result_transfer_audit_20261001'
OUT=ROOT/'artifacts/v158_full_result_records_20261001'
CASE=ROOT/'training/review_policy/v158_observed_fusion_cases.json'
REPORT=ROOT/'docs/V158_COMPLETE_TRAINING_RESULTS_AND_DECISION.md'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists() and not CASE.exists() and not REPORT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve(),ROOT/'training/v158_replay_observed_cases.py',
        TRIAL/'final_delivery.json',TRIAL/'quality.json',TRIAL/'verification.json',TRIAL/'learning_qualification.json',
        TRIAL/'full_prediction_ledger.parquet',TRIAL/'ASA_prediction_ledger.parquet',TRIAL/'all_fixed_cohort_quality.json',
        BASE/'base_phase_completion.json',BASE/'legacy_phase_completion.json',BASE/'execution_failure.json',
        ROOT/'artifacts/v158_legal_fusion_bank_20261001/failure.json',
        ROOT/'artifacts/v158_zero_update_stage_identity_audit_20261001/audit.json',
        AUDIT/'audit.json',AUDIT/'all_ASA_origin_and_actual_endpoints.parquet']
    paths.extend(TRIAL/f'fold{f}_{a}/fit.json' for f in range(3) for a in ['A','B'])
    paths.extend(ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{f}/deployment_probabilities.npy' for f in range(3))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(OUT/'pre_result_record_bindings.json',dict(status='actual_sources_bound_before_report_and_case_recount',source_sha256=bindings))
    d=read(TRIAL/'final_delivery.json');q=read(TRIAL/'quality.json');v=read(TRIAL/'verification.json');diagnosis=read(AUDIT/'audit.json')
    assert d['classifier_fits']==60 and not d['quality_acceptance'] and not d['model_promoted']
    a=pd.read_parquet(AUDIT/'all_ASA_origin_and_actual_endpoints.parquet');assert len(a)==112807
    failed=[k for k,b in q['B']['gates'].items() if not b]
    save(CASE,dict(version='V158_actual_failed_probability_fusion_cases',failed_task_gates=failed,
        cases=[dict(name='lower_OOF_CE_and_TRAIN_retention_do_not_close_outer_quality',action='Reject task adoption and automatic confirmation.'),
            dict(name='equal_total_error_hides_10_M_repairs_and_10_S_regressions',action='Enforce both registered class protections and multiple folds.'),
            dict(name='strict_correct_control_regresses_at_origin_and_during_learning',action='Keep all original controls and separate origin from learning.'),
            dict(name='all_17_negative_S_minus_M_cannot_be_repaired_by_convex_weights',action='No claim that new conditions or additional iterations solve the proven 188 cases.'),
            dict(name='four_zero_accepted_stages_have_no_progress_file',action='Require registered stop and exact state identity; do not rerun completed fits.')],
        source_sha256=bindings,automatic_extra_fits=False,quality_acceptance=False))
    fits=[read(TRIAL/f'fold{f}_{arm}/fit.json') for f in range(3) for arm in ['A','B']]
    base=read(BASE/'base_phase_completion.json');legacy=read(BASE/'legacy_phase_completion.json')
    fusion_calls=sum(r['classifier_forward_calls'] for r in fits)
    total_calls=base['classifier_forward_calls']+legacy['ASA_classifier_forward_chunks']+fusion_calls+240+v['actual_evaluation_classifier_calls']
    costs=dict(classifier_fits=60,current_pipeline_fits=45,legacy_full_population_fits=9,fusion_fits=6,
        batch_gradients=base['batch_gradients'],current_dense_full_gradients=base['full_gradients'],legacy_full_gradients=legacy['full_gradients'],
        fusion_full_gradients=sum(r['full_gradients'] for r in fits),fusion_proposals=sum(r['proposal_evaluations'] for r in fits),
        current_accepted_updates=base['accepted_updates'],legacy_accepted_iterations=legacy['accepted_iterations'],fusion_accepted_updates=sum(r['accepted_updates'] for r in fits),
        classifier_forward_calls=dict(current_pipeline=base['classifier_forward_calls'],legacy_ASA_chunks=legacy['ASA_classifier_forward_chunks'],
            fusion_preflight=240,fusion_fit=fusion_calls,fusion_evaluation=v['actual_evaluation_classifier_calls'],total=total_calls),
        internal_features_calls=base['internal_features_calls'],setup_dummy_forwards=3,setup_dummy_gradients=3,setup_dummy_updates=0,
        failed_first_completed_fit_already_in_current_pipeline_totals=True,failed_assembly_extra_classifier_calls=0,
        post_result_diagnosis_classifier_calls=0,post_result_diagnosis_gradients=0,automatic_extra_fits=False)
    assert costs['batch_gradients']==35600 and costs['current_dense_full_gradients']==5893 and total_calls==185522
    save(OUT/'complete_actual_costs.json',costs)
    origin={str(c):int((a.truth.eq(c)&a.pred_registered_origin.ne(c)).sum()) for c in [1,2]}
    assert origin=={'1':1720,'2':1658}
    source_flip=a[a.pred_A.ne(a.pred_B)].groupby(['fold','root','truth']).size().reset_index(name='original_rows')
    oof_lines=[]
    for row in diagnosis['OOF']:
        z=row['OOF'];oof_lines.append(f"|{row['training_role']}|{z['registered_origin']['CE']:.9f}|{z['A']['CE']:.9f}|{z['B']['CE']:.9f}|{z['registered_origin']['classes']['1']['errors']}/{z['registered_origin']['classes']['2']['errors']}|{z['A']['classes']['1']['errors']}/{z['A']['classes']['2']['errors']}|{z['B']['classes']['1']['errors']}/{z['B']['classes']['2']['errors']}|")
    failed_text='、'.join(failed)
    report=f'''# V158 完整训练、逐行验收与停止决定

2026-10-01。**60 次登记拟合已完成，六个融合终点及最后五步已真实重放；完整 2,056,871 行质量失败，停止本配置，不晋升、不追加确认。** B 的 ASA M/S 错为 414/1,872；相对 A 修复 10M、新增 10S，违反 S 保护且没有两折改善。旧联合 TRAIN 保护通过，只证明其登记训练范围保持；第二、第三及完整目标仍 active。

## 1. 真实拟合和监督来源

三外折、每折三次内层留根，每次从合法 FIT 重建 V135 全网、V138 读出、V140 C 读出、V142 既有第二层、V146 A 第二层，共 45 次；另外九次全任务 N1 稀疏线性三分类拟合，保留 N/M/S 原频次。所有五个当前监督阶段和 N1 均排除同一查询根与外层根，没有复用旧监督表示作为新 OOF。54 次实际终点、模型/分数散列、原类质量和逐次调用已核验后才装配 225,614 条合法 OOF 角色原行，并登记六个融合拟合。

17 专家为当前 16 个完整三类概率和旧 A0 N1 的完整三类归一分数。A 是共享概率评分器（208 参数）；B 增加全部 495 事实和固定 32 维全正文 CountSketch（8,640 参数）。投影有损、两臂容量不同；结果不单独证明某字段有因果作用。旧 OVA 分数归一保持 argmax，不宣称校准。三个初始旧模型份额在新 OOF 拟合前由合法 FIT 间隔固定为 0.095702018/0.122452508/0.112641298；没有比例、种子、维数或外层阈值扫描。

所有六次融合均使用合法 OOF 原行平均概率 CE 梯度，到固定 200 次接受更新，共 1,200 完整梯度/1,200 有限 proposal/1,200 更新。每次接受状态保留实际部署 FIT 的所有当前正确原行；外层真值没有进入梯度或终点选择。

## 2. 完整成本与实际重放

|阶段|真实拟合|梯度|接受更新/迭代|分类器分块 forward|
|---|---:|---|---:|---:|
|当前完整五阶段链|45|35,600 batch + 5,893 dense 完整梯度|39,381|155,759|
|全任务 N1|9|976 完整风险梯度|885 L-BFGS-B 迭代|27 ASA 查询分块|
|融合零步|0|0|0|240|
|融合训练|6|1,200 完整梯度|1,200|29,016|
|六终点及各末五状态验收|0|0|0|480|

实际分类器分块 forward 总计 **185,522**；当前网络内部 features 调用 59,639 单列，不与上述调用相加。三次运行依赖预热另有 3 dummy forward/gradient、0 更新。不同粒度的 batch 梯度、完整梯度及线性接受迭代不混作同一种步数。全部成本见 `artifacts/v158_full_result_records_20261001/complete_actual_costs.json`。

原第一项全网 100 轮/6,300 更新完成后才发生日志重复 `stage` 错误；原模型与成本保留，新版本只接续剩余项目，没有重跑第一项或重置 60 次预算。装配首版在列举缺失零更新轨迹时失败，发生在读取数组重组前，额外模型调用/拟合/梯度/更新均零；另版仅允许登记 `numerical_no_change` 且接受更新为零的四项缺失 progress。四项初始/终点张量与概率身份已实际核对，没有补造轨迹。旧源码、seal、故障和首项成本均保留。

六个终点、各最后五个不同参数状态和概率均与已保存结果重放一致；模型、轨迹、真实梯度/试探调用和 OOF 类质量相符。六融合的终点及末五状态都保持纯 M/S 错零，完整 FIT M0/S22、6、28，225,558 条正确 TRAIN 角色原行联合旧 V138/V140 范围通过。22/6/28 是当前数值表示契约，追加旧 N1 分数后经验下限为 22/4/26，不能宣称融合已到完整专家输入的可观测下限或官方原文不可分。

N1 逐梯度日志保留原类质量、总 objective 与梯度大小，并记录接受迭代；没有每接受状态的逐类/来源错误、CE/Brier 和参数身份。完整神经轨迹不能扩称为 N1 全轨迹。该缺口保留，不回填历史或重训已完成项。

## 3. 来源外完整任务和 A/B

|ASA 原行开发人口|M 错 /78,748|S 错 /34,059|总错|
|---|---:|---:|---:|
|旧 A0 项目参考|318|2,074|2,392|
|前轮 V146 A|1,904|1,629|3,533|
|V158 登记零步先验|1,720|1,658|3,378|
|V158 A|424|1,862|2,286|
|V158 B，登记候选|414|1,872|2,286|

**初始化与学习分开。** 零步相对 V146 A 净少 184M、净多 29S；B 从零步继续实际修复 1,306M，新增 214S，无 M 新错或 S 修复。不能将零步发生的退化全部归给后续学习，也不能把大量 M 修复包装成两类共同改善。B 相对 A0 修复/新增为 M118/214、S368/166；相对 V146 A 为 M1,490/0、S0/243。

匹配 B 对 A 的 10M 修复分布于外1根29的2条、根18100的4条、外2根21999的4条；10S新增错全部在外1根29。三折总错误 A 为 282/1,006/998，B 为 282/1,010/994：第一折持平、第二折恶化、第三折改善。总错持平不替代 S 保护和多折要求。

全 2,056,871 行的独立 gold、身份与路由完整重算，类支持 N=1,899,723、M=111,728、S=45,420。B 的 N 漏判0、误报105；M 漏判446、误报1,874；S 漏判1,947、误报414。完整 P/R/F1 分别为 N 0.999944732/1/0.999972365，M 0.983438792/0.996008163/0.989683570，S 0.990566683/0.957133421/0.973563103。M 召回和 S 精确率低于 A0，即使两个威胁类 F1 略升也不能通过逐项保护。非 ASA 的登记冻结107错仍全部计分；正常类巨大分母不替代 ASA 与来源保护。

B 的 ASA 总错2,286大于2,170，M414大于318；头部682记录错18，根2868恶意错192，大三组外 S 错1,387（上限1,372）。S 来源平均召回 0.111559997 低于 A0 的 0.115125271，零召回根203大于198。B 失败项为：{failed_text}。逐项布尔和全 N/M/S 指标见 `quality.json`，不能用损失或总错净收益覆盖。

## 4. 合法 OOF 学习与困难/正确对照

|训练角色|初始 OOF CE|A终点 CE|B终点 CE|初始 M/S错|A M/S错|B M/S错|
|---|---:|---:|---:|---|---|---|
{chr(10).join(oof_lines)}

以上保留每角色全部原频次。六次 CE 都实际下降；外0及外1从零步到 B 的 S 分类错误却增加50/30。这是平均损失改善不保护逐类分类的实际反例，不能改类权重来追认本轮。OOF 拟合侧错误与部署 FIT 守卫不是同一人口；登记分类守卫只覆盖合法部署 FIT，不能宣称所有 OOF 记录分类已学会。

578 困难 S：V146 A 错576，零步即错578，A/B仍错578；188条全部17专家 S-M 间隔严格为负的共同失败原行留在该578分母内，非负凸融合数学上不能将其判 S。51严格正确 S 对照：V146 A 错0，零步错21，A/B错49，因此其中28条净退化发生于后续学习。ICMP2373原行全部保留；B恶意错148/1,654、可疑错719/719。未知/缺端口49,599原行，B恶意错330/35,816、可疑错1,223/13,783；未知哨兵不能冒充已观测同值，ICMP不需要传输端口。当前数值混标206原行全部保留，B为 M0/S28；V155新增16M反例本轮全部正确，但不足以抵销新S错。

完整固定切片和全部变化原行/来源保存在 `all_fixed_cohort_quality.json` 及 `artifacts/v158_saved_result_transfer_audit_20261001/`。后者只读取已封存的输出、做三次初始数组重组，0新增模型/特征求值、梯度、拟合、更新，属于结果诊断，不是新试验或因果识别。

## 5. 决定、恢复和未完成目标

本配置已结束，60 次预算耗尽；不追加相同凸融合的种子、份额、阈值、条件维数或确认。A 不是预登记候选且也失败，不事后晋升 A。当前最新实际训练和完整任务交付都应指向 V158，旧 V155/V157/V158部分执行快照作为历史保留。

后续只能根据新的合法机制资格另立版本；188共同失败需能改变判别证据或分类器可表达范围，不能继续只改这17专家的凸权重。合法 OOF支持、输入可观测性、训练分类与来源外分类分别核验；不将已查看外层真值反写专家路由。当前没有新拟合登记，也没有新外部环境验收。第二、第三及完整目标仍 active；本轮分类质量失败不等于完整目标完成或应自动暂停。

恢复入口：`artifacts/v158_fusion_trial_20261001/final_delivery.json`、`verification.json`、`quality.json` 和完整逐行 parquet；来源监督 `artifacts/v158_current_pipeline_OOF_trial_20261001/`；合法专家库 `artifacts/v158_legal_fusion_bank_v2_20261001/`；正式计划 `training/review_policy/v158_complete_nested_trial_v2_plan.json` 与融合契约 `v158_fusion_execution_contract.json`。旧 seal 和入口不可改写或重跑。所有本轮训练/evaluation session 已 terminal，无活动训练进程。

新增真实反例入口为 `training/review_policy/v158_observed_fusion_cases.json`，回放 `training/v158_replay_observed_cases.py`；回放独立验证全人口、真实类别交换、末五身份、188凸组合限制及分类门槛，0新训练。只读 MCP 发布与测试凭据另存，不改上述正式训练交付哈希。
'''
    REPORT.write_text(report,encoding='utf-8');check_bindings(bindings)
    save(OUT/'records.json',dict(status='V158_full_real_results_and_observed_cases_written',quality_acceptance=False,model_promoted=False,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [CASE,REPORT,OUT/'complete_actual_costs.json',OUT/'pre_result_record_bindings.json']}))
    print(dict(status='records_written',classifier_fits=60,task_acceptance=False,total_classifier_calls=total_calls))

if __name__=='__main__':main()
