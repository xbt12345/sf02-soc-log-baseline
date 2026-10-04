"""Publish observed scope; never mutate sealed numerical evidence."""
import json
from pathlib import Path
from experiment_review import sha, check_bindings

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v148_factual_relation_qualification_20261001'
DOC=ROOT/'docs/V148_FACT_RELATION_QUALIFICATION.md'
CASE=ROOT/'training/review_policy/v148_observed_qualification_cases.json'

def main():
    assert not DOC.exists() and not CASE.exists()
    sources=[OUT/'capacity.json',OUT/'gradient_probe.json',OUT/'mask_scope_audit.json',
        ROOT/'artifacts/v148_independent_relation_relevance_20261001/audit.json']
    for p in sources:
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('source_sha256'):check_bindings(d['source_sha256'])
    g=json.loads((OUT/'gradient_probe.json').read_text(encoding='utf-8'))
    assert g['qualification_passed'] and g['new_fits']==0 and g['functional_parameter_evaluations']==12
    cases=dict(status='whole_hidden_fact_distance_not_ready_for_classifier_trial',latest_actual_training='V146',
        local_optimizer_qualification=True,semantic_training_qualification=False,next_classifier_training_registered=False,
        new_classifier_fits=0,new_optimizer_updates=0,new_persistent_classifier_updates=0,
        actual_CE_gradients=6,actual_fact_gradients=6,actual_functional_parameter_points=12,
        issue_solved=False,quality_acceptance=False,model_promoted=False,
        cases=[dict(id='V148-MASK-INTERSECTION',observed_zero_cross_class_edges=6893,
            different_observation_masks=5917,same_complete_parsed_facts=976,
            action='Do not interpret common observed-field equality as whole representation or threat-label equality.'),
            dict(id='V148-LOCAL-DESCENT',local_points_passed=6,
            action='Retain numerical probe pass; require actual field recoverability gap before a new component auxiliary trial.'),
            dict(id='V148-NO-NEW-CLASS-SUPPORT',
            action='Fact graph edges are not new independent S labels. Preserve all original mass and same/opposite-class support controls.')],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    CASE.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    DOC.write_text('''# V148 事实关系的实际资格与停止范围

2026-10-01。最新实际分类器仍为 V146：6 次拟合、1200 次完整梯度、2408 次有限试探、1200 次接受更新，匹配作用和完整质量失败。本轮 V148 **分类器拟合 0、优化器更新 0、持久分类参数更新 0**；实际执行了 6 次 CE 梯度、6 次事实梯度和 12 个函数参数试探点，不能误称零计算。

三折仅合法 TRAIN 的事实图共 313936 条有向边、213430 原频次锚点质量；全部分类 TRAIN 角色原行 225614 条仍保留。标签置换图完全不变，未知哨兵不当具体值。三折监督覆盖 S 原行 32757/2033/31800；这不是新增独立 S 标签。

六个 A/B 局部点实际总目标下降且通过 V142/V140/V138 联合已正确 TRAIN 保护，模型状态未改变。`gradient_probe.json` 的局部数值资格通过保持原样；它没有授予整个分类表示的手工事实距离正式训练资格。

独立图复算发现 6893 条目标事实距离为零但两端纯 M/S 类别不同的合法 TRAIN 边。其中 5917 条观察掩码不同，976 条完整解析 facts 相同。前者只证明共同已知字段相等；后者也不证明完整模型输入或威胁判据相同。将这两种零距离约束施于整个成员隐藏方向，尚无任务相关性依据。范数和事实直连仍可能保留分类，因此也不能据此断言所有事实辅助必定有害。

**停止的是当前 whole-hidden 距离候选的直接正式拟合，保留其数值探针和反例；下一步定位原文→坐标→固定第一层/当前第二层的具体字段可恢复性。** 新解码诊断若求解读出参数，必须如实登记拟合与预算；已见这些 TRAIN 标签的冻结基座，其内层主体留出读取结果不是独立盲测。只有具体成分缺口成立才考虑成分辅助，不能借覆盖或梯度默认再加重建任务。

当前执行方向以 [独立决定与条件式方案](V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md) 为唯一下一分类训练设计。输入端新增证据见 [V149 输入保真审查](V149_INDEPENDENT_INPUT_FIDELITY_AND_NEXT_CHECK.md)：578 困难 S 的基础可观察字段未丢在坐标；502 条大小写源角色缺口均为分类正确 M，不能追认为分类收益，历史输入不改。

原 A0 门槛属于项目预登记参照与保护要求，不是组委会评分公式。第二、第三及完整任务仍开放，无模型晋升。来源开发折已查看；事实读取进步不能代替全 2056871 行三分类及来源外修复/退化验收。

实际凭据：`artifacts/v148_factual_relation_qualification_20261001/capacity.json`、`gradient_probe.json`、`mask_scope_audit.json`；独立关系复算 `artifacts/v148_independent_relation_relevance_20261001/audit.json`；可执行边界 `training/review_policy/v148_observed_qualification_cases.json`。所有旧执行源码、封存和日志保持。
''',encoding='utf-8')
    catpath=ROOT/'mcp_readonly/catalog.json';cat=json.loads(catpath.read_text(encoding='utf-8'))
    project=cat['project'];project['current_summary']+=' V148完成6次CE和6次事实梯度、12函数参数试探，0新分类拟合/更新；局部资格通过但whole-hidden事实关系语义资格未成立。'
    project['authoritative_direction_id']='v148-direction-review'
    project['current_direction']=[
        '最新实际V146质量失败，无晋升；第一问题仅注册TRAIN范围闭合，继续V142/V140/V138联合保护。',
        'V146受限配对路径停止；V148整段隐藏表示事实距离仅局部数值资格通过，尚不正式拟合。',
        '下一步核实具体行为字段在现有表示中的读取缺口，诊断拟合另登记，不冒充新盲测或新增同类标签。',
        'V149输入审计显示578困难S的基础字段坐标保留；502源角色大小写缺口均分类正确，历史输入不改。',
        '第二、第三及全任务开放；新分类器拟合尚未登记。']
    project['known_limits']=[x.replace('正式 A0 质量门槛保持','项目预登记 A0 质量门槛保持') for x in project['known_limits']]+[
        'V148共同观察字段零距离不等于全输入或威胁类别相同。',
        '解码参数拟合属于实际拟合；冻结基座已见内层TRAIN标签，不能作为独立泛化证据。']
    entries=[('v148-direction-review','V148下一训练决定与针对性方案',ROOT/'docs/V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md','current_review'),
        ('v148-qualification-review','V148实际事实关系资格与停止范围',DOC,'review_evidence'),
        ('v148-capacity','V148合法事实关系图',OUT/'capacity.json','review_evidence'),
        ('v148-gradient-probe','V148真实梯度及函数参数探针',OUT/'gradient_probe.json','review_evidence'),
        ('v148-mask-scope','V148零距离观察范围反例',OUT/'mask_scope_audit.json','review_evidence'),
        ('v148-independent-relevance','V148独立任务相关性复算',sources[-1],'review_evidence'),
        ('v148-observed-cases','V148实际资格限制',CASE,'review_evidence'),
        ('v149-input-fidelity','V149独立原文与坐标保真审查',ROOT/'docs/V149_INDEPENDENT_INPUT_FIDELITY_AND_NEXT_CHECK.md','review_evidence')]
    old_ids={z['id'] for z in cat['documents']};assert not old_ids.intersection(z[0] for z in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=project['current_summary'],sha256=sha(p),keywords=['V148','V149','事实','字段','资格','当前方向']) for i,t,p,c in entries]+cat['documents']
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8');s=s.replace('最新实际与当前方向（V146）','最新实际（V146）与当前方向（V148/V149）',1).replace('[当前V147零拟合细支持审查](docs/V147_LEGAL_FINE_SUPPORT_REVIEW.md)','[V148实际资格与停止范围](docs/V148_FACT_RELATION_QUALIFICATION.md)；[当前针对性方案](docs/V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md)；[V149输入保真](docs/V149_INDEPENDENT_INPUT_FIDELITY_AND_NEXT_CHECK.md)',1)
    rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V148实际资格、V149输入保真与下一读取诊断

V148实际6 CE + 6事实梯度、12函数参数点，0分类拟合/优化器/持久更新；局部总目标及225558旧正确TRAIN角色保护通过。6893目标零距离跨纯类别边中5917观察掩码不同、976完整facts同值；停止whole-hidden事实距离直接正式训练，不覆写已通过的数值探针。当前方向docs/V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md，实际docs/V148_FACT_RELATION_QUALIFICATION.md，机器案例v148_observed_qualification_cases.json。

父维护V149原文/495事实坐标解码：13字段全112807与trace一致，578困难S原文可观察基础字段均保留；502 Outside-3620源角色缺口为105roots/139locals全部M，各历史模型均正确。不得改历史输入或追认为502修复。下一执行方独占模型读取诊断；如需解码器求解，先新独立诊断登记预算/全源码依赖/合法角色，实际计入诊断拟合；基座已见内层TRAIN标签，不冒充新盲测。最新实际分类模型V146保持，完整目标active。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v147-support-review','v148-direction-review').replace('self.assertIn("2408", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V148", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    print(json.dumps(dict(published=True,latest_actual='V146',direction='V148',new_classifier_fits=0),ensure_ascii=False))

if __name__=='__main__':main()
