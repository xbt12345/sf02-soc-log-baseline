"""Publish incremental predicate and sealed TRAIN transfer evidence, no fits."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import read,sha,check_bindings
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v153_raw_residual_delta_20261001'
GAP=ROOT/'artifacts/v153_independent_training_transfer_gap_20261001'
DOC=ROOT/'docs/V153_INCREMENTAL_PREDICATES_AND_TRANSFER_MECHANISM_REVIEW.md'
CASE=ROOT/'training/review_policy/v153_observed_mechanism_cases.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not DOC.exists() and not CASE.exists()
    a=read(BASE/'audit.json');g=read(GAP/'audit.json')
    check_bindings(a['source_sha256']);check_bindings(g['source_sha256'])
    assert a['new_original_rows_parsed']==5854 and a['V53_original_captures_reused_and_raw_gold_verified']==106953
    assert a['new_rows_in_known_578']==0 and a['explicit_access_group_binding_rows']==0
    assert a['new_non_grammar_reason_flags_hitcount_rows']==0
    hard=[x for x in g['cohorts'] if x['cohort']=='known_578'];assert len(hard)==2
    for x in hard:
        assert x['TRAIN_role_rows']==x['pure_TRAIN_role_rows']==1156
        assert x['pure_TRAIN_errors']==x['all_TRAIN_role_errors']==0 and x['historical_outer_unique_errors']==576
    assert g['same_family_and_outer_fold_correct_control_S']==51
    evidence=[BASE/'audit.json',BASE/'all_original_capture_provenance.parquet',BASE/'new_5854_complete_raw_spans.parquet',GAP/'audit.json',
        ROOT/'docs/V53_ASA_EXECUTION.md',ROOT/'docs/V95_FOUR_ARM_TRAINING_RESULTS_AND_FAILURE_ANALYSIS.md',
        ROOT/'docs/V97_MATCHED_TRAINING_RESULTS_AND_DECISION.md',ROOT/'docs/V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md']
    cases=dict(status='TRAIN_mastery_not_transfer_new_raw_predicate_not_qualified',latest_actual_classifier='V146',
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,issue_solved=False,
        next_classifier_training_registered=False,cases=[
            dict(id='V153-MASTERED-NOT-TRANSFERRED',hard_original_rows=578,hard_TRAIN_role_rows_per_arm=1156,
                hard_TRAIN_errors_per_arm=0,hard_outer_errors_per_arm=576,
                minimum_TRAIN_pS_A=hard[0]['TRAIN_S_probability']['0.0'],minimum_TRAIN_pS_B=hard[1]['TRAIN_S_probability']['0.0'],
                action='Do not reclassify these rows as TRAIN optimizer failure or repeat the existing mastery fit. A new mechanism must explain held-source transfer and retain current TRAIN behavior; no hard-cohort fit whitelist.'),
            dict(id='V153-CONTROL-EXCHANGEABILITY',other_correct_both_port_S=19698,same_protocol_direction_ranges_and_fold_controls=51,
                action='Broad rare-correct S are not interchangeable with outside-to-dmz hard examples. Use narrow controls for development falsification only, never as new weights, labels, a selected-field rule or a blind score.'),
            dict(id='V153-PURE-VS-PROTECTED',pure_TRAIN_role_rows=225202,pure_independent_original_rows=112601,
                protected_correct_TRAIN_role_rows=225558,protected_independent_original_rows=112779,all_TRAIN_errors=56,
                action='Keep purity, protection and class conflict populations separate. All 56 errors lie in mixed numeric inputs; joint V142/V140/V138 protection still covers majority-correct mixed rows.'),
            dict(id='V153-INCREMENT-NOT-NEW-LABEL-SUPPORT',V53_original_rows_reused=106953,new_rows=5854,new_roots=2,new_fold=1,
                new_M_rows=2,new_S_rows=5852,known_hard_new_rows=0,
                action='Reuse verified legacy grammar and validate only the actual delta. Do not count the old review or repeated new rows as independent hard-case support or a new feature-discovery mechanism.'),
            dict(id='V153-ACL-NAME-NOT-BINDING',quoted_ACL_labels=7,explicit_access_group_binding_rows=0,
                action='ACL label suffix in/out is not the configured access-group direction. Preserve labels but do not infer policy bindings, rule intent or class targets without actual context.'),
            dict(id='V153-OLD-METHOD-NOT-RESET',historical_methods=['V94-97 MLDG/MAG','V40-53 exact-port interactions and complete combinations'],
                action='A newly searched paper, second-order approximation change or renamed interaction alone is not a new mechanism qualification. Bind prior actual results and identify a new evidenced factor before any new fit.')],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence})
    save(CASE,cases)
    DOC.write_text('''# V153 增量原文谓词与已掌握分类的迁移边界

2026-10-01。**原 578 条困难 S 在 V146 A/B 的两个合法 TRAIN 角色均全部分类正确，最低 S 概率约 0.991；各自来源外仍错 576 条。当前缺口是已掌握类别的迁移，不能继续称这些记录“训练侧没学会”。原文增量检查没有发现可据以启动新训练的漏提取谓词。** 最新实际分类器仍 V146；本轮新前向、梯度、拟合、更新均 0。已验收 TRAIN 能力受保护，细支持、跨来源稳定性及完整任务仍开放。

## 实际 TRAIN 与来源外必须用对应模型分开

父入口 `training/v153_independent_training_transfer_gap.py` 实际核对六终点的 checkpoint/概率文件哈希；每臂 225614 TRAIN 角色账本的概率逐元素与该角色模型已保存 `sealed_all_prob.npy[local]` 相同，自身 outer 折决策与全部 112807 原行相同。没有新运行分类器，父本轮没有独立前向重放；模型执行身份继续依赖绑定的 V146 全终点重放凭据，不将概率文件一致偷换成新模型计算。

|原 578 群体|A|B|
|---|---:|---:|
|合法 TRAIN 角色曝光|1156|1156|
|其中纯数值输入曝光|1156|1156|
|TRAIN 错误|0|0|
|TRAIN 最低 p(S)|0.990660|0.990348|
|自身 outer 原行错误|576|576|

两次 TRAIN 曝光不是 1156 条新独立数据。这个差距拒绝“只继续提高现有训练拟合强度就能解决 578”的归因，也不证明一切算法不可能。新机制需对合法 TRAIN 的类别条件、推理可得信息与来源外迁移有具体假说，而非进一步记忆已掌握输入或重建已读对字段。

必须澄清分母：`pure_TRAIN_input` 为 225202 角色行、112601 独立原行；已正确 TRAIN 保护为 225558 角色行、112779 独立原行。保护包含混标输入的多数正确记录，不等于纯输入。每臂 56 个全 TRAIN 错误都在混标数值输入中，纯 TRAIN 错误 0；不借冲突删除难例或缩小旧联合保护。

其他双端口可读且来源外正确的 S 有 19698 原行；与困难记录匹配协议、源/目的角色、源/目的端口范围且同 outer 折，仅 51 条。大量出站 UDP 正确样本与 outside→dmz 困难样本不能无条件比较。51 条只用于已看开发数据的对照审查，不能构造训练名单、权重、标签或按此选字段，也不能计作盲测。

实际概率/角色来源见 `artifacts/v153_independent_training_transfer_gap_20261001/audit.json`；父文件归父维护，执行方未改写。

## 只补查新增原文，不重复旧语法发现

[V53](V53_ASA_EXECUTION.md)已解析 106953 原行并核查基础行为字段，没有找到新明确行为字段。本轮按真实 row_position 核对这 106953 行原文 SHA256 与标签，全部一致；仅解析当前多出的 **5854 原行**。它们只有 **两个隔离根、同一第 1 折，M 2/S 5852**，原 578 在增量中为 0。

入口 `training/v153_raw_residual_delta_review.py` 沿用 V53 已执行 whole-message 语法，对新增每行保存完整不重叠字符跨度，原文拼回与长度完全一致，未匹配 0。新增原行仍只有 Deny、协议、源/目的端点、可选 ICMP type/code、ACL 标签及哈希，不出现额外原因、TCP flags、命中次数或策略绑定命令。这个结论是这套观察语法的增量验证，不是假定所有可能安全上下文都已穷尽。

合并已核验旧捕获与新捕获，全部 112807 原行均保留，包括全部 ICMP、未知端口和原 578。ACL 子句词恒为 `ORG-1738-group`，哈希对恒为 `0x0, 0x0`，当前 V124 时间头之后、Deny 之前的标识恒为 `USER-0010-0324`。这些常量没有新增类别条件；标识已脱敏，不据它宣布确定的原厂事件号。7 个 ACL 标签及完整原文保留，不变成名单路由。

## ACL 标签与真实谓词的区别

`outside_ORG-1738_in`、`DMZ_ORG-1738_out` 等是日志中的 ACL 标签。观察到 `_in/_out` 文字不等于已观察真实配置方向。[Cisco access-group 官方命令](https://www.cisco.com/c/en/us/td/docs/security/asa/asa-cli-reference/A-H/asa-command-ref-A-H/aa-ac-commands.html)把 ACL 名称、`in/out` 和绑定 interface 作为不同参数；例如名称 `acl-1` 也可以绑定 inbound。本题这些日志没有明确的 access-group 绑定命令或策略条目。推论：仅凭名称尾缀生成 ingress/egress 教师会把未验证的命名约定当真值，本轮不授予该机制训练资格。名称在推理时可读取，也不足以证明其跨环境含义。

本轮只读取官方现有数据及一手命令资料，没有外部训练集、标签补全、原文删除或历史输入改写。语法剩余可区分不证明攻击可区分，常量/匿名词也不提供新的威胁监督。

## 不用已有方法重新起算资格

检索交互与元学习时发现项目已执行精确端口/协议交互、EBM 与完整组合对照（V40–V53），以及 MLDG/MAG 风格更新（[V95](V95_FOUR_ARM_TRAINING_RESULTS_AND_FAILURE_ANALYSIS.md)、[V97](V97_MATCHED_TRAINING_RESULTS_AND_DECISION.md)）。它们的局部收益和整体退化均保留。不能把新搜到同一论文、重命名方法或仅改变梯度近似当作独立新机制，自动重置拟合预算。本轮没有运行这些旧入口。

暂无新正式分类拟合登记，不重复现有 TRAIN mastery、字段解码或旧辅助配置。[V152 条件式下一轮](V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md)仍规定：新因素有可证伪依据后，单因素两臂三折固定有界拟合、训练前完整绑定、旧正确 TRAIN 联合保护、完整 2056871 行逐类和来源外质量验收；本报告没有降低这些要求或关闭目标。

产物 `artifacts/v153_raw_residual_delta_20261001/` 保存新 5854 捕获/完整跨度、全部旧新来源账本、原文成分清单及 audit。机器反例 `training/review_policy/v153_observed_mechanism_cases.json` 把已掌握未迁移、严格对照范围、纯输入与保护范围、旧语法增量、ACL 名非配置、旧方法非新预算落实为动作。当前审查完成不等于训练收益或全任务完成。
''',encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project']
    assert pr['authoritative_delivery_id']=='v146-delivery'
    pr['authoritative_direction_id']='v153-mechanism-review'
    pr['current_summary']+=' V153绑定真实TRAIN/outer概率：578在两合法TRAIN角色0错、outer仍576错；仅补查5854新原文，无新增合格谓词，0新模型/梯度/拟合/更新。'
    pr['current_direction']=[
        '最新实际分类器仍V146，V153概率/原文证据不替代训练交付；无质量晋升。',
        '578已在两个合法TRAIN角色掌握，当前是迁移缺口，不重复现有mastery或事实读出拟合。',
        '19698广泛正确双端口S中同协议/方向/范围/折仅51；只作开发对照，不据名单造目标或权重。',
        '复用核验V53的106953原行，仅解析5854增量；ACL名称/常量hash不提供可信新条件，旧MLDG/MAG和交互不重置资格。',
        '新机制有独立依据才按V152条件式计划登记两臂三折；旧联合保护与全2056871逐类质量保持，第二第三及完整目标active。']
    pr['known_limits']+=['V153父本轮只绑定保存概率，不新前向；TRAIN与historical outer概率角色分开。',
        '225202 pure TRAIN角色记录与225558正确保护记录不同；ACL标签不是实际策略绑定。']
    entries=[('v153-mechanism-review','V153已掌握分类的迁移与增量谓词决定',DOC,'current_review'),
        ('v153-raw-delta-results','V153官方新增5854语法增量实测',BASE/'audit.json','review_evidence'),
        ('v153-training-gap-results','V153父六终点TRAIN与来源外概率绑定',GAP/'audit.json','review_evidence'),
        ('v153-observed-cases','V153真实迁移与谓词边界动作',CASE,'review_evidence')]
    assert not {d['id'] for d in cat['documents']}.intersection(e[0] for e in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=pr['current_summary'],sha256=sha(p),keywords=['V153','当前方向','迁移','TRAIN','谓词','原文']) for i,t,p,c in entries]+cat['documents'];save(cp,cat)
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8').replace('当前方向（V152）','当前方向（V153）',1)
    s=s.replace('[V152当前条件式决定]','[V153迁移与增量谓词审查](docs/V153_INCREMENTAL_PREDICATES_AND_TRANSFER_MECHANISM_REVIEW.md)；[V152当前条件式决定]',1)
    s=s.replace('V152新增实际：','V153新增实际：578在两合法TRAIN角色均0错、outer仍576错；正确双端口S的协议/方向/范围/折严格对照仅51。复用核验旧106953原文，仅补查5854增量，无新合格谓词；0模型前向/梯度/拟合/更新，暂无新正式训练登记。pure TRAIN角色225202不等于保护225558。\n\nV152新增实际：',1);rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V153真实TRAIN/outer差距与原文增量

当前docs/V153_INCREMENTAL_PREDICATES_AND_TRANSFER_MECHANISM_REVIEW.md，下一条件式计划仍V152，actual classifier V146。父v153_independent_training_transfer_gap.py绑定六终点checkpoint/probs，两个TRAIN角色578均pure/0错，1156角色曝光每臂，最低pS A.990660/B.990348，outer仍576错；本轮父0新前向不冒充独立模型重放。广泛正确双端口S19698，同协议/方向/范围/折仅51，只开发诊断不能造名单权重。pure TRAIN225202角色/112601原行，保护225558/112779包括混标多数正确，56 TRAIN错全混标。

执行方v153_raw_residual_delta_review.py验证旧V53 106953 raw/gold，只新解析5854（M2/S5852、两个根、fold1、原578增量0）完整跨度精确，无新reason/flags/hitcount/真实policy绑定。全112807旧新账本保留；hash0x0,0x0、ACL词、事件匿名标识常量；7个ACL标签在推理可得但不等于access-group in/out配置。没有新语义教师/特征/伪标签/模型。旧V40-53交互及V94-97 MLDG/MAG已执行，不新搜改名重置预算。

产物artifacts/v153_raw_residual_delta_20261001/及父gap audit，机器案例v153_observed_mechanism_cases.json已绑定动作。当前暂无合格新正式拟合，不重训已掌握TRAIN或已读字段，不缩小/关闭第二第三/full；新机制须解释合法条件、严格correct controls与迁移收益，再按V152固定预算、旧联合保护及全2056871逐类验收。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v152-training-decision','v153-mechanism-review').replace('self.assertIn("V152", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V153", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    save(BASE/'publication.json',dict(status='V153_executed_evidence_published_no_classifier_fit',latest_actual_classifier='V146',new_fits=0,new_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),DOC,CASE,cp,rp,hp,tp]}))
    print(json.dumps(dict(published=True,current_direction='V153',latest_actual_classifier='V146',new_fits=0,quality_acceptance=False),ensure_ascii=False))

if __name__=='__main__':main()
