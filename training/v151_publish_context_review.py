"""Publish executed context evidence and enforce its observed scope; zero fits."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import read, sha, check_bindings

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v151_context_coherence_20261001'
CURRENT=ROOT/'artifacts/v151_current_context_view_20261001'
INDEPENDENT=ROOT/'artifacts/v151_independent_context_qualification_20261001'
DOC=ROOT/'docs/V151_CONTEXT_COHERENCE_AND_CLASS_CONDITION_REVIEW.md'
CASE=ROOT/'training/review_policy/v151_observed_context_cases.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not DOC.exists() and not CASE.exists(),'Preserve observed publication'
    audits=[read(BASE/'audit.json'),read(CURRENT/'audit.json'),read(BASE/'verification.json'),read(INDEPENDENT/'audit.json')]
    for audit in audits:
        check_bindings(audit['source_sha256'])
        assert all(audit[k]==0 for k in ['new_model_forwards','new_gradients','new_fits','new_updates'])
        assert not audit['quality_acceptance'] and not audit['issue_solved']
    a,b,v,ind=audits
    ledger=pd.read_parquet(BASE/'all_ASA_context_ledger.parquet')
    assert len(ledger)==112807 and ledger.row_position.nunique()==112807
    assert ledger[ledger.truth.eq(1)].product_name_state.eq('empty').all()
    assert ledger[ledger.truth.eq(2)].product_name_state.eq('observed').all()
    assert ledger.syslog_severity.eq(4).all()
    assert v['exact_available_attribute_groups']==50863 and v['all_ICMP_retained']
    assert b['current_text_byte_input_replayed_local_entries']==22546 and b['current_text_byte_input_max_abs_gap']==0
    assert ind['old_v75_date_sensitive_original_rows']==682 and ind['current_date_sensitive_original_rows']==0
    projections={r['key']:r for r in b['projections'] if r['scope']=='whole_population'}
    floors=[projections[k]['deterministic_observed_projection_floor'] for k in ['v1_old_v75_partial_clock_view_key','old_normalized_partial_clock_view_key','current_V124_normalized_ordered_view_key']]
    assert floors==[50,114,116]
    evidence=[BASE/'audit.json',BASE/'verification.json',BASE/'all_ASA_context_ledger.parquet',
        BASE/'all_outer_legal_condition_support.parquet',CURRENT/'audit.json',CURRENT/'all_ASA_current_ordered_view_ledger.parquet',
        INDEPENDENT/'audit.json',ROOT/'docs/V151_INDEPENDENT_CONTEXT_QUALIFICATION_AND_NEXT_ACTION.md',
        ROOT/'artifacts/v150_independent_result_review_20261001/audit.json',ROOT/'docs/V150_INDEPENDENT_REVIEW_AND_NEXT_TRAINING_DECISION.md']
    cases=dict(status='observed_context_not_qualified_threat_teacher',latest_actual_classifier='V146',
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,issue_solved=False,
        next_classifier_training_registered=False,cases=[
            dict(id='V151-PRODUCT-PROXY',M_empty=78748,S_observed=34059,projection_floor=0,
                action='Perfect product-presence separation is provenance evidence only. Do not create a threat teacher, routing answer, lookup prediction, pseudo-label or training weight from this association.'),
            dict(id='V151-FACILITY-NOT-SEVERITY',severity4_original_rows=112807,facts_floor=2510,facts_severity_floor=2510,facts_facility_floor=1330,
                action='Separate configurable facility from severity. Lower projection conflict is not new classifier gain or qualified M/S semantics.'),
            dict(id='V151-OLD-CLOCK-SCOPE',old_date_sensitive_rows=682,current_date_sensitive_rows=0,
                text_input_entries_replayed=22546,byte_max_gap=0,old_unclustered_floor=50,old_normalized_floor=114,current_normalized_floor=116,
                action='Keep v1 and qualify its old partial-clock key name. Use the current V124 path for current-input claims; do not attribute 50->116 entirely to date removal or equate text projection with whole CSR identity.'),
            dict(id='V151-AVAILABLE-ATTRIBUTES-NOT-PACKETS',original_rows=112807,unique_event_ids=112807,distinct_available_attributes=50863,duplicate_excess_rows=61944,
                action='Retain original classification frequency. Equal available attributes and distinct IDs do not establish packet identity, original event rate, generated duplication or causal temporal support.'),
            dict(id='V151-CROSS-FIELD-NAMESPACE',source_IP_different_rows=111407,destination_IP_different_rows=112553,
                source_port_not_comparable_rows=37793,ICMP_rows=2373,
                action='Keep equality/difference/not-comparable separately. Without mapping evidence, do not declare data wrong, verified same entity, attack chain or a new graph mechanism; retain all ICMP and unknown states.'),
            dict(id='V151-OBSERVED-PROJECTION-NOT-QUALITY',facts_mixed_groups=215,facts_mixed_rows=23220,facts_floor=2510,current_text_floor=116,actual_CSR_floor=28,
                action='State each observed projection scope. Neither floor is classifier quality, official-data impossibility, a blind score or authority to discard unresolved raw context. Continue legal TRAIN leave-root class-condition evidence before registering a new classifier trial.')],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence})
    # Actual counterexample quantities remain tied to the executed ledger.
    c=cases['cases'][4]
    assert int(ledger.src_ip_record_body_relation.eq('different').sum())==c['source_IP_different_rows']
    assert int(ledger.dst_ip_record_body_relation.eq('different').sum())==c['destination_IP_different_rows']
    assert int(ledger.src_port_record_body_relation.eq('not_comparable').sum())==c['source_port_not_comparable_rows']
    save(CASE,cases)
    DOC.write_text('''# V151 原始上下文、当前输入视图与类别条件资格

2026-10-01。**产品缺失状态能完全区分当前 ASA 的 M/S 标签，但没有独立依据将其当威胁判据；本轮未找到合格的新上下文教师。首版“时钟已移除”的过宽命名已通过另版实际纠正，历史证据保留。** 最新实际分类器仍 V146，未新增模型前向、梯度、拟合或参数更新。第一问题仅登记 TRAIN 范围通过并受保护，细同类支持、跨来源稳定性、完整 N/M/S 与泛化目标仍开放。

本轮三个执行文件为 `v151_context_coherence_audit.py`、`v151_current_context_view_audit.py`、`v151_verify_context_evidence.py`。从官方 2056871 行重新读取全部 112807 ASA 原行，原文、真值及原行身份与追踪表一致。所有 ICMP、未知、混标、原 578 群体和正确对照均保留；没有生成查表预测、分类权重或复制标签。V150 的六事实解码拟合及 72 特征前向仍单独计账，本轮零拟合不抹去历史计算。

## 原列状态与语义资格

|原始 ASA 类别|产品/厂商为空|产品/厂商有值|原行数|
|---|---:|---:|---:|
|M|78748|0|78748|
|S|0|34059|34059|

执行方逐行核对，父会话独立重读官方列得到同一结果。完整 facts 有 215 混标组、23220 混标原行，该投影观察最低错误 2510；加入未获资格的上下文状态后为 0。父独立审查确认，仅加入产品有无状态就已为 0。因此“零混标”只说明当前列与标签有关，不证明新增攻击条件或跨环境可靠性。产品、NULL/空串、采集包装保留作来源风险证据，当前不授予其威胁教师、路由答案或伪标签职责。

|facility|severity|M 原行|S 原行|
|---:|---:|---:|---:|
|20|4|14432|34059|
|22|4|64316|0|

全部 112807 条 severity=4。facts 加 severity 的观察最低错误仍 2510，加 facility 才变为 1330。[RFC 5424](https://www.rfc-editor.org/info/rfc5424/)分别定义 facility 和 severity；[Cisco logging facility 命令](https://www.cisco.com/c/en/us/td/docs/security/asa/asa-cli-reference/I-R/asa-command-ref-I-R/m_log-lz.html)允许配置 facility。当前相关性应按日志来源/配置线索审查，不能解释成安全严重度改善。

[Cisco 同类 Deny 日志说明](https://www.cisco.com/c/en/us/td/docs/security/asa/syslog/asa-syslog/syslog-messages-101001-to-199021.html)说明 ACL 拒绝包，持续同源消息可能提示扫描。它没有提供本数据的 M/S 标注细则；原文消息号已脱敏，也不能断言全部原行就是未脱敏的 106023。策略规则、事件关联和持续性仍需实际记录依据，不从一条 Deny 自动产生攻击标签。

## 结构化列与正文实体、频次

结构化源 IP 与正文可比较源 IP 不同：M 77348、S 34059，共 111407；相同 1400。目的 IP 不同 112553，相同 254。源端口可比较且相同 70144、不同 4870、不可比较 37793；其中全部 2373 ICMP 不适用 TCP/UDP 端口比较。原 578 群体中源端口相同 510、不同 68，全部结构化/正文地址不同。

这是字段之间的实际相等性，不是实体关联证书。没有命名空间映射时，不能把不同解释成错误数据、同一攻击主体或新攻击链；亦不能因这些差异将旧正文构图路线包装成新机制。官方实际 13 列没有独立 ACL 策略、包标识、事件链或标注理由列；协议/action/severity 可以由正文解析，“无独立列”不等于这些观察值已丢失。本轮仅使用题目与现有数据，不等待额外标注。

112807 个 event_id 全不同；去掉 event_id 与 label 后，全可用属性精确相同分组 50863，重复属性组 16826，包含 78770 原行，超出每组第一行的记录共 61944。复核 exact drop_duplicates 与指纹分组一致，未发现哈希碰撞。全部时间值有限，43554 个不同时间值。相同可用属性无法证明重复生成、包身份或原始速率；分类原频次全部保持，未以去重后的数量替换质量分母。

## 当前规范视图：版本纠正与实际缓存一致

首版 `ordered_identity_clock_removed_key` 实际来自旧 `v75_views.view`，应解释为“旧部分时钟清洗视图”。旧审查/账本/源码不覆写；另版采用当前 `v124_header.transform`，包含当前包装规范。父独立仅改变日期复核：旧视图 682 原行变化，当前视图 0 变化；682 补丁的正文、头部跨度和规范文本哈希与已执行 V124 凭据一致。执行方还重建 **22546 个实际文本输入条目**，与当前 `B_header_ASA.npz` 的 65792 文本坐标最大差 **0**。这些是输入路径验证，本轮没有运行分类模型。

|观察投影|混标组|混标原行|本投影观察最低错误|
|---|---:|---:|---:|
|旧 v75 部分时钟清洗、未做当前包装规范|11|962|50|
|旧头部、已做当前包装规范|24|1392|114|
|当前 V124 规范有序文本|24|1394|116|
|当前规范有序文本 + 完整 facts|24|1394|116|

**50→116 混有包装规范变化，不能把 66 全归于日期；同一包装规范下头部差为 114→116。** 日期不变性只覆盖已登记语法，不是任意日志语法保证。完整数值输入还含记录级端口与事实坐标，已审计经验最低错误仍 28；单独文本投影的 116 不替代完整输入身份。ACL/接口名称、大小写与包装残余没有新语义资格，不删除它们后宣布全部不可解，也不因为字符串可区分就当威胁真值。

完整 facts 的三合法 TRAIN 角色保留原频次 92337/40957/92320，混标组 194/27/196，本投影观察最低错误 2286/318/2116。当前有序文本三角色最低错误 82/40/92。它们是描述性条件证据，不是新拟合成绩或分类器可达质量证书。

## 决定与可执行边界

没有新分类器方案登记；不重开已失败的 V146 辅助训练，不把 V150 已读对的 578 字段作为重建缺口。下一实际动作是在每外折合法 TRAIN 内留出来源根，核对完整 facts+观察状态、省略精确源端口的单独检索键、同类/反类原频次与跨根集中度，并导出真实原文反例；省略检索字段不改变分类输入。全部未知、ICMP、混标和正确对照继续保留。已看的外折仅作开发诊断，不能据此构造条件名单、权重或标签。来源根只是隔离代理，不代表真实企业或攻击主体。

父会话独立 V152 条件统计另行执行、另有产物；本报告不预记其未完成结果。若出现具体合格条件或表示缺口，再按独立登记、固定预算与三折两臂进入有界试验；旧 V142/V140/V138 正确 TRAIN 联合保护、全 2056871 原行逐类质量仍是后续门槛。没有新的真实判据时，不用来源包装保证修复或将全目标标记完成。

实际凭据：`artifacts/v151_context_coherence_20261001/` 的 audit、verification、全原行 context/support 账本；`artifacts/v151_current_context_view_20261001/` 的另版 audit/ledger；父独立 `artifacts/v151_independent_context_qualification_20261001/audit.json`。机器边界 `training/review_policy/v151_observed_context_cases.json` 绑定实际来源并规定反例动作；发布入口对全人口、类别状态、时钟反证和投影范围作真实断言。

独立决定见 [V151 上下文资格审查](V151_INDEPENDENT_CONTEXT_QUALIFICATION_AND_NEXT_ACTION.md)；V150 全原行解码与独立复核范围见 [V150 独立复核和下一训练决定](V150_INDEPENDENT_REVIEW_AND_NEXT_TRAINING_DECISION.md)。本轮输入/状态同步不证明分类质量提升，没有晋升模型，三个问题和最终任务继续 active。
''',encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project']
    pr['current_summary']+=' V151实际全112807 ASA上下文与另版当前视图审查，22546文本输入重建差0，0新模型前向/梯度/拟合/更新；产品状态完美分离未获威胁资格。'
    pr['authoritative_direction_id']='v151-context-review'
    pr['current_direction']=[
        '最新实际分类器V146质量失败；V150六解码拟合与V151零拟合上下文审查不替代分类交付。',
        'V151产品存在状态完美分离标签但未获威胁资格，facility与severity分开，不生成查表/伪标签/训练权重。',
        '首版旧部分时钟视图保留另版纠正；当前V124文本22546条重建差0，投影116不等于完整CSR最低28。',
        '下一实际动作合法TRAIN逐根排除类别条件与原文反例，完整facts/掩码与省源端口检索分别计数，保留全部状态和正确对照。',
        '无新正式分类拟合登记，继续旧正确TRAIN联合保护及全2056871逐类质量；第二第三和完整目标active。']
    pr['known_limits']+=['V151产品/facility/匿名实体/属性重复只诊断来源，尚无新可信M/S上下文。',
        'V151首版key名称过宽，另版复核当前路径；50->116含包装规范变化，不能全部归因日期。']
    entries=[('v151-context-review','V151实际上下文审查及当前视图纠正',DOC,'current_review'),
        ('v151-context-results','V151全原行上下文描述',BASE/'audit.json','review_evidence'),
        ('v151-current-view-results','V151另版当前文本重建',CURRENT/'audit.json','review_evidence'),
        ('v151-context-verification','V151官方源与原频次复核',BASE/'verification.json','review_evidence'),
        ('v151-context-cases','V151实际上下文反例动作',CASE,'review_evidence'),
        ('v151-independent-review','V151独立上下文资格决定',ROOT/'docs/V151_INDEPENDENT_CONTEXT_QUALIFICATION_AND_NEXT_ACTION.md','review_evidence'),
        ('v151-independent-results','V151独立原列与日期反证',INDEPENDENT/'audit.json','review_evidence'),
        ('v150-independent-review','V150独立读取复核和下一决定',ROOT/'docs/V150_INDEPENDENT_REVIEW_AND_NEXT_TRAINING_DECISION.md','review_evidence'),
        ('v150-independent-results','V150独立全字段结果复核',ROOT/'artifacts/v150_independent_result_review_20261001/audit.json','review_evidence')]
    assert not {d['id'] for d in cat['documents']}.intersection(e[0] for e in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=pr['current_summary'],sha256=sha(p),keywords=['V151','上下文','当前方向','类别条件','时钟','来源']) for i,t,p,c in entries]+cat['documents']
    save(cp,cat)
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8').replace('当前方向（V150）','当前方向（V151）',1)
    s=s.replace('[实际六拟合与全量验收]','[V151实际上下文与当前视图审查](docs/V151_CONTEXT_COHERENCE_AND_CLASS_CONDITION_REVIEW.md)；[实际六拟合与全量验收]',1)
    s=s.replace('V150新增实际：','V151新增实际：全112807 ASA上下文审查、22546当前文本条目重建差0，0模型前向/梯度/拟合/更新；产品状态完全区分类别但无威胁资格。首版旧时钟视图保留并另版纠正；继续合法TRAIN逐根类别条件证据，暂无新分类拟合登记，完整目标active。\n\nV150新增实际：',1);rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V151上下文及当前视图实际审查

全112807 ASA官方原文/真值/13列重读；0模型前向/梯度/拟合/更新。M78748产品/厂商全empty，S34059全observed，加入产品状态观察投影最低0但未获威胁教师资格。facility20 M14432/S34059，facility22 M64316/S0，全部severity4；2510->1330不能解释成严重度。结构化/正文地址差异不授予实体关联，50863精确属性组和61944重复超额不授予包身份/速率，保持全部原频次与2373 ICMP。

首版ordered_identity_clock_removed_key实为旧v75部分时钟视图，保留全部已执行来源；另版当前V124处理682旧头缺口，全部22546文本条目字节重建差0。旧未包装规范投影50、旧头同规范114、当前116，不能将66全归日期，也不替代完整CSR最低28。报告docs/V151_CONTEXT_COHERENCE_AND_CLASS_CONDITION_REVIEW.md、机器案例v151_observed_context_cases.json、BASE verification及当前视图audit/ledger；父V151独立审查/V150独立复核已接入MCP，父源不覆写。

最新分类交付仍V146；V150实际6诊断拟合/72特征前向保持单列。无新分类拟合登记、无质量晋升；第一已登记TRAIN能力保护，第二第三/full active。父会话执行V152 legal TRAIN逐根排除类别条件统计（其独立文件归父），不预记结果，不重复请求额外官方说明；只用题目+现有数据。后续新机制有真实依据才独立登记试验，旧保护与全2056871质量保持。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v150-readability-review','v151-context-review').replace('self.assertIn("V150", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V151", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    receipt=dict(status='context_review_published_without_classifier_promotion',latest_actual_classifier='V146',current_direction='V151',
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,issue_solved=False,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),DOC,CASE,cp,rp,hp,tp]})
    save(BASE/'publication.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'},ensure_ascii=False))

if __name__=='__main__':main()
