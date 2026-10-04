"""Publish source-bound leave-root findings; preserve classifier identity."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import read,sha,check_bindings
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v152_worker_support_review_20261001'
PARENT=ROOT/'artifacts/v152_independent_leave_root_conditions_20261001'
DOC=ROOT/'docs/V152_WORKER_SUPPORT_CONCENTRATION_REVIEW.md'
DIRECTION=ROOT/'docs/V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md'
CASE=ROOT/'training/review_policy/v152_observed_support_cases.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not DOC.exists() and not CASE.exists()
    a=read(BASE/'audit.json');original=read(PARENT/'original_support_and_domain_qualification.json')
    for x in [a,original,read(PARENT/'audit.json')]:check_bindings(x['source_sha256'])
    q=pd.read_parquet(BASE/'all_query_root_mass_concentration.parquet')
    good=q[q.truth.eq(2)&q.support_state.eq('multiple_roots_same_class_only')]
    assert set(good.outer_training_role)=={2} and good.actual_V146_B_correct.all()
    assert good.groupby('key_scope').size().to_dict()=={'complete_facts_mask_key':330,'omit_source_port_key':1004}
    for x in a['original_support_summaries']:
        if x['truth']==2:
            parent=next(v for v in original['summaries'] if v['scope']=='all_S' and v['key_scope']==x['key_scope'])
            assert x['no_same_support_either_role']==parent['no_same_support_either_role']
            assert x['historic_outer_correct_without_same_support']==parent['no_same_support_but_historical_outer_correct']
            assert x['multi_same_no_opposite_both_roles']==parent['multi_root_same_only_both_roles']==0
    evidence=[BASE/'audit.json',BASE/'all_query_root_mass_concentration.parquet',BASE/'two_role_unique_original_support.parquet',
        BASE/'original_counterexample_scope_review.json',PARENT/'audit.json',PARENT/'original_support_and_domain_qualification.json',DIRECTION,
        ROOT/'training/v152_worker_support_review.py',ROOT/'artifacts/v152_worker_support_review_attempt1_20261001/failure.json']
    cases=dict(status='leave_root_condition_evidence_not_training_qualification',latest_actual_classifier='V146',
        new_model_forwards=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,issue_solved=False,
        next_classifier_training_registered=False,cases=[
            dict(id='V152-ROLE-EXPOSURE',independent_original_rows=112807,legal_TRAIN_role_rows=225614,two_key_query_rows=451228,
                action='Restore original-row summaries across the two legal roles. Do not count role exposure, duplicated queries or concentration as new independent labels.'),
            dict(id='V152-SUPPORTED-CONTROL-NOT-STABILITY',complete_key_multi_unopposed_S_role_rows=330,omit_SRC_multi_unopposed_S_role_rows=1004,
                only_training_role=2,all_historic_outer_correct=True,unique_S_multi_unopposed_in_both_roles=0,
                action='Keep favorable support and root concentration scoped to this one role and existing correct controls. Do not claim difficult S coverage, stable correction or new training gain.'),
            dict(id='V152-ABSENT-EXACT-SUPPORT-CORRECT',S_without_complete_same_support_both_roles=32781,historic_outer_correct=31246,
                action='Exact-key support absence is not a sufficient error cause or universal veto on compositional methods. New mechanisms still need lawful conditions, hard-versus-correct controls and matched classification evidence.'),
            dict(id='V152-HARD-ROLE-NOT-OUTER',known_hard_original_rows=578,complete_same_support_any_role=0,
                omit_SRC_same_support_any_role=70,omit_SRC_same_support_both_roles=22,omit_SRC_multi_same_roots_any_role=0,
                action='These legal TRAIN queries differ from the old outer query cohort with 48 matching S rows. Do not merge denominators, transfer single-root labels, erase port values or treat retrieval omission as label-preserving augmentation.'),
            dict(id='V152-RAW-FACT-SCOPE',legal_TRAIN_raw_pairs_verified=9,
                action='Equal parsed facts/masks with cross-root M/S quotes are not whole-input equivalence, label noise proof or a new context teacher. Product/facility/anonymous/time packaging still lacks threat qualification.'),
            dict(id='V152-CORRECTNESS-ROLE',prediction_source='historical_outer_V146_B',
                action='Historical outer correctness is descriptive control only. Do not use it as corresponding TRAIN-role retention proof, fit selection or weights.')],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence})
    save(CASE,cases)
    lines=['# V152 执行方独立复核：原频次集中与角色范围','',
        '2026-10-01。**451228 个逐根查询的双方类别原频次与根数独立重算一致；多根同类且无反类的 S 只出现在合法 TRAIN 角色 2，未形成跨角色稳定依据。** 最新分类器仍 V146，新增模型前向、梯度、拟合、更新均 0；质量和完整目标未通过。当前下一训练决定以 [父 V152 条件式决定](V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md) 为准。','',
        '实际入口 `training/v152_worker_support_review_v2.py`。首版在反例引用处误用已经设为索引的 row_position 属性，输出前失败；旧源码与 failure.json 保留，另版改用真实索引后完成。此前均无模型调用或拟合。父 V152 原文件未修改。','',
        '按全部合法 TRAIN 的 key×类别×root 原行质量独立构造根频次表，排除查询自身根。与父的 451228 查询逐行比较两类原频次与根数，全部一致；重新构造完整 facts+掩码与省精确源端口键，也逐行一致。全部 112807 原行保留，每行两次合法 TRAIN 角色，不能将 225614 曝光或 451228 查询当新独立数据。','',
        '除根数外，另算每类其他根的最大原频次占比、HHI=sum(root_mass²)/total_mass²、有效根数=1/HHI。没有改变频次、增加等根权重或产生预测。下表只列“多根同类且无反类”的 S 角色原行；零支持的集中度保持未定义，不填成满分。','',
        '|检索键|合法 TRAIN 角色|S 角色原行|最大根占比范围|有效根数范围|',
        '|---|---:|---:|---:|---:|']
    for x in a['concentration_metrics']:
        if x['truth']==2 and x['support_state']=='multiple_roots_same_class_only':
            lines.append(f"|{x['key_scope']}|{x['role']}|{x['original_role_rows']}|{x['max_root_share_min']:.4f}–{x['max_root_share_max']:.4f}|{x['effective_root_min']:.4f}–{x['effective_root_max']:.4f}|")
    lines+=['',
        '这些原行历史 outer V146 B 均已正确；完整键 330、省源端口键 1004 都仅在角色 2。两合法角色均具有多根同类且无反类的独立 S 原行数为 0。这个结果限制“当前支持池已经稳定覆盖难例”的主张，不禁止其他有依据的组成机制，也不说明数据普遍无解。','',
        '两个合法角色都无完整键同类支持的 S 原行 32781，其中历史 outer 正确 31246；省源端口键对应 27292、25945。与父去角色重复摘要一致。缺支持不能独自解释类别错误，更不能设成所有新方法的必要门槛。历史 outer 正确标记不等于这些 TRAIN 角色模型预测，旧 V142/V140/V138 TRAIN 验收仍须实际联合入口证明。','',
        '原 578：完整键任一合法角色同类支持为 0；省精确源端口后任一角色有支持 70、两个角色都有 22、任一多同类根 0。它不同于旧外折检索的 48；保留具体参数和未知状态，省略仅用于检索对照，不复制 S 标签。','',
        '父按合法 TRAIN 原频次固定选出的 9 对原文已逐对核对：原行身份/原文精确、M/S 真值相反、root 不同、外折排除成立、facts/观察状态相等。另保留当前规范文本相等性与产品状态差异，不解释成整个输入/威胁相同或错误标注。正文未确定差异仍保留，不生成上下文规则。','',
        '全人口根集中与所有类别/支持状态结果保存在 `artifacts/v152_worker_support_review_20261001/`：all_query_root_mass_concentration、all_class_state_concentration_metrics、two_role_unique_original_support、original_counterexample_scope_review 和 audit。`training/review_policy/v152_observed_support_cases.json` 将角色重复、正确无支持、单角色收益、外折人口差异落实为审查动作。','',
        '暂无合格新分类拟合登记，不追加旧 V146 或字段重建路线。新机制需解释困难与正确稀缺对照、合法 TRAIN 判定条件及推理可获得性，再按父条件式方案注册单因素两臂三折的有界分类试验，保持旧正确 TRAIN 链和完整 2056871 行逐类保护。第一问题只在登记 TRAIN 范围通过；细支持、来源外稳定性、完整 N/M/S 及泛化仍 active。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project']
    assert pr['authoritative_delivery_id']=='v146-delivery'
    pr['authoritative_direction_id']='v152-training-decision'
    pr['current_summary']+=' V152合法TRAIN逐根451228查询及独立原频次集中复核完成，0新模型/梯度/拟合/更新；缺精确支持的32781 S中31246历史outer正确，多根同类无反类S仅单角色，暂无新正式训练登记。'
    pr['current_direction']=[
        '当前方向按V152条件式决定；最新实际分类器仍V146，V151/V152审查无模型晋升。',
        '不追加旧V146、字段重建、端口抹平、重复过采样或产品/日期/匿名身份答案。',
        '缺精确同类支持不是失败充分原因或所有方法必要门槛；多根同类无反类S仅角色2且均历史outer正确。',
        '下一独立机制须解释hard与正确稀缺对照、合法TRAIN真实类别条件及推理可得性，有依据才登记两臂三折固定有界拟合。',
        '旧V142/V140/V138联合保护和完整2056871逐类验收保持；第二第三及完整泛化目标active。']
    pr['known_limits']+=['V152两次合法TRAIN角色不是新增独立样本，historical outer正确标记不能认证对应TRAIN角色。',
        'V152根是隔离代理，多根计数/集中度不证明真实攻击条件；精确缺支持不排除组成泛化。']
    entries=[('v152-training-decision','V152当前训练决定和条件式下一轮',DIRECTION,'current_review'),
        ('v152-worker-support-review','V152独立原频次集中和角色复核',DOC,'review_evidence'),
        ('v152-legal-conditions','V152父合法TRAIN逐根原行条件',PARENT/'audit.json','review_evidence'),
        ('v152-original-summary','V152父去角色重复原行摘要',PARENT/'original_support_and_domain_qualification.json','review_evidence'),
        ('v152-worker-results','V152独立查询身份与集中实测',BASE/'audit.json','review_evidence'),
        ('v152-observed-cases','V152真实支持反例审查动作',CASE,'review_evidence')]
    assert not {d['id'] for d in cat['documents']}.intersection(e[0] for e in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=pr['current_summary'],sha256=sha(p),keywords=['V152','当前方向','支持','训练决定','角色','来源']) for i,t,p,c in entries]+cat['documents'];save(cp,cat)
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8').replace('当前方向（V151）','当前方向（V152）',1)
    s=s.replace('[V151实际上下文与当前视图审查]','[V152当前条件式决定](docs/V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md)；[V152独立原频次集中复核](docs/V152_WORKER_SUPPORT_CONCENTRATION_REVIEW.md)；[V151实际上下文与当前视图审查]',1)
    s=s.replace('V151新增实际：','V152新增实际：合法TRAIN逐根451228查询及独立原频次集中复核，0模型前向/梯度/拟合/更新。多根同类无反类S仅单角色，32781无精确支持S中31246历史outer正确；精确缺支持不是万能根因，暂无合格新分类拟合登记。完整目标active。\n\nV151新增实际：',1);rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V152实际逐根条件与独立原频次集中

当前方向docs/V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md（父维护勿覆写），latest actual V146保持，0模型前向/梯度/拟合/更新。父完整facts+mask和省精确SRC检索合法TRAIN排自身root451228查询；执行方v152_worker_support_review_v2.py独立重构键/双方原频次/根数全一致，另实际计算HHI/最大根占比/有效根数。首版row_position已设索引后误用属性，输出前失败，旧源码与attempt1/failure保留；v2完成。

S多根同类无反类330完整/1004省SRC角色行均仅role2且历史outer正确，两角色均成立的独立S为0；不能当稳定难例支持。32781原S两个角色均无完整键同类，31246仍historical outer正确，支持缺失不是充分根因/所有方法必要门槛。578完整键同类0；省SRC任一角色70、两个22、任一多根0，不与历史外折48混用。所有未知/ICMP/206混标/正确控制保留。historical outer正确不能认证对应TRAIN角色，root非真实domain。

报告docs/V152_WORKER_SUPPORT_CONCENTRATION_REVIEW.md、artifacts/v152_worker_support_review_20261001/审查与全集中账本；9对父合法TRAIN原文M/S反例核对，不等于全输入/威胁同一。机器反例v152_observed_support_cases.json。当前无新正式训练资格，不重复同一解释或字段读出；独立机制须解释hard与正确controls、合法TRAIN条件、推理可得性，有依据才注册单因素两臂三折固定预算及封存，旧链保护与全2056871质量保持，第二第三/full active。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v151-context-review','v152-training-decision').replace('self.assertIn("V151", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V152", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    save(BASE/'publication.json',dict(status='V152_evidence_published_no_classifier_promotion',latest_actual_classifier='V146',new_fits=0,new_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),DOC,CASE,cp,rp,hp,tp,DIRECTION]}))
    print(json.dumps(dict(published=True,current_direction='V152',latest_actual_classifier='V146',new_fits=0,quality_acceptance=False),ensure_ascii=False))

if __name__=='__main__':main()
