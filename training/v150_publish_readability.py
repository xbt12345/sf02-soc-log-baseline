"""Publish all fields, controls and diagnostic compute without model promotion."""
import json
from pathlib import Path
import pandas as pd
from experiment_review import sha,read,check_bindings

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v150_field_readout_diagnostic_20261001'
DOC=ROOT/'docs/V150_FROZEN_FIELD_READABILITY_RESULTS.md'
CASE=ROOT/'training/review_policy/v150_observed_readability_cases.json'

def main():
    assert not DOC.exists() and not CASE.exists()
    audit=read(OUT/'result_audit.json');check_bindings(audit['source_sha256'])
    assert audit['actual_diagnostic_decoder_fits']==6 and audit['new_classifier_fits']==0
    fields=pd.read_parquet(OUT/'all_field_state_metrics.parquet')
    values=pd.read_parquet(OUT/'all_exact_value_metrics.parquet')
    controls=pd.read_parquet(OUT/'all_classifier_control_metrics.parquet')
    # Actual diagnostic exposure, not assumed representation destruction.
    icmp=fields[fields.layer.eq('H1')&fields.role.eq('inner_HELD')&fields.field.eq('icmp_type')]
    unseen=icmp[icmp.stratum.eq('observed_value_unseen_decoder_fit')]
    assert int(unseen.original_rows.sum())==1700 and int(unseen.errors.sum())==1700
    same=icmp[icmp.stratum.eq('observed_value_seen_decoder_fit')]
    assert int(same.original_rows.sum())==32 and int(same.errors.sum())==0
    rare=controls[controls.layer.eq('V146_B_H2')&controls.role.eq('outer_HELD')&controls.stratum.eq('correct_rare_same_class_S')]
    assert int(rare.original_rows.sum())==18669
    evidence=[OUT/'result_audit.json',OUT/'all_field_state_metrics.parquet',OUT/'all_exact_value_metrics.parquet',
        OUT/'all_classifier_control_metrics.parquet',OUT/'registration.json',ROOT/'training/review_policy/v150_field_readout_plan.json']
    cases=dict(status='factual_readability_not_threat_criterion',latest_actual_classifier='V146',
        diagnostic_decoder_fits=6,new_classifier_fits=0,new_classifier_updates=0,quality_acceptance=False,
        issue_solved=False,next_classifier_training_registered=False,
        cases=[dict(id='V150-READABLE-STILL-WRONG',original_rows=578,all_observed_fields_exact_both_layers=578,
            all_field_states_exact_both_layers=578,classifier_errors=576,
            action='Do not justify a field-reconstruction classifier trial as repair for this cohort. Check category-conditioned support and missing context; historical HELD is diagnostic only.'),
            dict(id='V150-DECODER-UNSEEN-TYPE',inner_observed_type_rows=1732,unseen_type8_rows=1700,unseen_type8_errors=1700,
            seen_type_rows=32,H1_seen_errors=0,H2_seen_errors=0,
            action='Separate decoder training exposure from backbone information. Linear-readout failure on unseen target values is not representation destruction or a new blind score.'),
            dict(id='V150-CORRECT-FIELD-ERROR-CONTROL',outer_correct_rare_same_class_S=18669,
            H2_complete_ports_exact=int(rare.complete_port_values_exact.sum()),
            action='Keep correct rare controls and all ICMP/unknown/zero states; outside-field readout errors alone do not cause threat errors.'),
            dict(id='V150-FIT-COUNT',diagnostic_fits=6,normal_equation_solves=6,
            learned_decoder_coefficients=audit['fitted_decoder_parameters_total'],
            action='Count factual decoder fitting as six actual diagnostic fits. Classifier fits/gradients/updates remain zero; no automatic budget reset or classifier promotion.')],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence})
    CASE.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# V150 冻结表示的字段读取：实际六拟合与下一分流','',
        '2026-10-01。**原 578 条困难 S 的全部已观察字段、全部字段状态，在固定第一层和 V146 B 第二层均准确读出；其中 576 条仍分类错误。继续重建这些已能读出的字段，不能据此解释或关闭这个细支持缺口。** 第二、第三和完整任务仍开放，最新实际分类器仍 V146，质量失败、无晋升。','',
        '本轮事前登记并执行 **6 次事实解码拟合、6 次固定 ridge 正规方程求解**；保存 4053504 个解码系数。分类器拟合、分类梯度、分类参数更新均 0，优化器更新 0；3 次登记检查和 69 次已冻结特征批前向，共 72 次特征模型前向。不能把正规方程求解称零拟合，也不能称 6 次新分类器训练。','',
        '两层 × 三固定来源折；每层展平完整 16×128 特征，不做成员均值。各折合法 TRAIN 内，主体 SHA256 的模 5 等于 0 作读出留出，其他作 FIT；三角色 FIT 原频次 83968/31984/82980，内层留出 8369/8973/9340。所有 112807 ASA 原行每折均保留。固定 ridge=0.0001，无系数、字段、层或检查点选择。348 个原事实 bit/enum 坐标同时监督，包含未知状态；它不是 13 字段等权威胁损失。','',
        '为避免浮点缓存微变体伪造信息，同 canonical CSR 输入固定最小 local 的特征代表，并恢复全部原频次。事实目标对 canonical 别名完全一致；分类器本身未改。3559 个实际物理依赖在拟合前封存，每次拟合先复核；OS/GPU 驱动没有做字节快照。基座已见内层留出分类标签，因此该留出只隔离读出拟合，不是独立盲测；外折已反复查看，只是来源外开发诊断。','',
        '下表为全部原行已观察字段的整值/类别错误，分母保持各字段的真实观察人口。端口必须完整整数读对，不用逐 bit 准确率代替。','',
        '|字段|内层主体留出 H1 错/观察行|内层主体留出 H2 错/观察行|完整外折 H1 错/观察行|完整外折 H2 错/观察行|外折同频次常量错/观察行|',
        '|---|---:|---:|---:|---:|---:|']
    metrics={(r['layer'],r['role'],r['field']):r for r in audit['all_observed_field_metrics']}
    for f in ['action','outcome','transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed','src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable']:
        cells=[]
        for role,layer in [('inner_HELD','H1'),('inner_HELD','V146_B_H2'),('outer_HELD','H1'),('outer_HELD','V146_B_H2')]:
            r=metrics[layer,role,f];cells.append(f"{r['errors']}/{r['original_rows']}")
        r=metrics['H1','outer_HELD',f];cells.append(f"{r['mode_errors']}/{r['original_rows']}")
        lines.append('|'+f+'|'+'|'.join(cells)+'|')
    lines+=['',
        '全部 FIT 原行的观察源/目的端口：H1 为 0/132430、0/166686；H2 为 6/132430、0/166686。协议、源/目的角色两层 FIT 均无观察值错误。这里是事实解码拟合成功范围，不能套作 M/S 分类指标。常量按各折 FIT 的原频次选择，未知状态可成为众数；deny/blocked 全人口常量，零错不算新增学习。每个协议/角色值和数值端口的分母、未知/不适用/观察零均在逐值与状态表中单列。','',
        '保留不利结果：全外折第二层目的端口错 8378 条；它不等同 8378 个威胁错误。同样稀缺但分类正确的双端口 S 对照共 18669 行，第二层完整端口仅 9842 行读对。字段读取错误不能单独当分类错误的因果证明；事实直连仍在原分类器中，坐标可以精确解码。','',
        'ICMP 不能用 TCP/UDP 的 578 群体掩盖。内层观察类型 1732 次，其中两角色共 1700 次 type=8 的值在各自解码 FIT 未出现，全部读错；已见类型 32 次两层均读对。代码/消息仍有其他真实错误，稀缺 FIT 类型 5/11 也存在错误。固定读出失败说明此受限读出与暴露范围不足，不证明第一层或第二层销毁信息；更强读出若需要拟合，必须独立登记，不能后验追加本轮预算。','',
        '原 578 群体分折 160/262/156，两层完整字段状态均读对；V146 B 分类错 158/262/156。该分组与同类/反类支持只在拟合后作控制，不曾构造目标、权重、阈值或层选择。没有新独立 S 标签，也没有新的分类修复。','',
        '**下一动作转向类别条件支持与真实缺失上下文，不再以“578 的字段读取不足”为依据添加事实重建。** 先核对合法 TRAIN 内相同可观察行为的 M/S 条件与跨主体证据，并保持全 ICMP、未知、混标和正确稀缺对照；不得删目的端口、复制单主体 S 标签或引入时间/产品/匿名身份字面捷径。外折字段错误与类别错误分列，不据已查看外折挑一个字段直接正式拟合。','',
        '新分类器方案尚未登记。后续条件式训练边界仍见 [V148 独立决定与针对性方案](V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md)：真正成分缺口才考虑成分辅助，分类残余不强制事实距离同化，旧正确 TRAIN 联合保护及完整 2056871 行质量保持。A0 保护要求属于项目预登记，不是组委会评分公式。','',
        '实际凭据位于 `artifacts/v150_field_readout_diagnostic_20261001/`：事前 registration/run_seal、六 decoder.pt/fit.json/逐原行账本、result_audit.json、all_field_state_metrics.parquet、all_exact_value_metrics.parquet、all_classifier_control_metrics.parquet 和两层 whole_outer_development_ledger。复算核对原坐标整值与逐行解码结果、全部分母和角色；没有声称额外独立重放六解码器或新的完整分类器质量验收。可执行边界：`training/review_policy/v150_observed_readability_cases.json`。']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project']
    pr['current_summary']+=' V150实际6次冻结事实解码诊断拟合、72特征前向，0新分类拟合/梯度/更新；578字段均读对但576分类错。'
    pr['authoritative_direction_id']='v150-readability-review'
    pr['current_direction']=[
        '最新实际分类器V146仍质量失败；V150六事实解码器不替代分类模型或重置旧预算。',
        'V148整段隐藏表示事实距离不直接正式训练；578字段两层全读对，重建不足不能解释576剩余错。',
        '下一步类别条件支持与真实缺失上下文，保留ICMP、未知/不适用、混标及正确稀缺对照；无新分类拟合登记。',
        '全外折H2字段解码仍有错误；ICMP内层未见type8值与读出错误分列，不能推断信息销毁或盲测通过。',
        '继续V142/V140/V138保护，第二第三和完整目标active，完整原行A0项目预登记门槛保持。']
    pr['known_limits']+=['V150基座已见内层留出标签；受限线性解码不等于独立泛化、威胁判据或信息不可恢复。',
        'V150六事实解码拟合保存4053504系数，但分类器参数均未更新。']
    entries=[('v150-readability-review','V150实际六事实解码与下一分流',DOC,'current_review'),
        ('v150-readout-plan','V150事前冻结解码方案',ROOT/'training/review_policy/v150_field_readout_plan.json','review_evidence'),
        ('v150-readout-results','V150全原行事实读取复算',OUT/'result_audit.json','review_evidence'),
        ('v150-readout-registration','V150事前封存登记',OUT/'registration.json','review_evidence'),
        ('v150-observed-cases','V150读取与威胁判据边界',CASE,'review_evidence')]
    ids={d['id'] for d in cat['documents']};assert not ids.intersection(x[0] for x in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=pr['current_summary'],sha256=sha(p),keywords=['V150','字段','解码','拟合','当前方向','类别支持']) for i,t,p,c in entries]+cat['documents']
    cp.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8').replace('当前方向（V148/V149）','当前方向（V150）',1)
    s=s.replace('[当前针对性方案](docs/V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md)','[V150实际六解码拟合及当前分流](docs/V150_FROZEN_FIELD_READABILITY_RESULTS.md)；[后续条件式方案](docs/V148_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md)',1)
    s=s.replace('历史实际（V140）','V150新增实际：6次冻结事实解码拟合、72次特征前向、0新分类拟合/梯度/更新；578困难S在两层全部观察字段及状态读对，仍576分类错。全人口ICMP/未知/正确稀缺对照均保留，第二第三及完整目标未通过。\n\n历史实际（V140）',1);rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V150六次冻结事实解码实际执行

已执行6解码拟合/6正规方程求解，4053504解码系数；72已冻结特征批前向，0分类器拟合/梯度/更新。registration/run_seal在拟合前，3559物理依赖；固定ridge1e-4、主体hash内层划分、两层16×128全展平、canonical最小local代表恢复原频次。禁止重新拟合、挑层/字段/阈值/系数；已见内层标签，不是盲测。

原578在H1/H2全部观察字段及状态均读对，V146仍576错；停止以该群体字段读取缺口解释新重建训练。全外折H2目的端口8378错与正确稀缺S对照18669保持，ICMP内层type8有1700未见值全错但已见type32条均读对，不宣称全表示完好或信息销毁。报告docs/V150_FROZEN_FIELD_READABILITY_RESULTS.md，artifacts/v150_field_readout_diagnostic_20261001/result_audit.json及全原行状态/逐值/分类控制表，机器反例v150_observed_readability_cases.json。最新分类模型V146，第二第三/full active；下一步类别条件支持与缺失上下文，不自动正式拟合。父V149缓存身份/微差报告保持不覆写。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v148-direction-review','v150-readability-review').replace('self.assertIn("V148", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V150", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    print(json.dumps(dict(published=True,latest_classifier='V146',current_direction='V150',decoder_fits=6,classifier_fits=0),ensure_ascii=False))

if __name__=='__main__':main()
