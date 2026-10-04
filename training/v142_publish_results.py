"""Publish executed V142 and subsequent zero-fit V143 evidence, not a new model."""
from pathlib import Path
from v142_runtime import ROOT,OUT,read,save,sha,require_run_seal


def main():
    require_run_seal(ROOT/'training/v142_train.py')
    if (OUT/'publication_receipt.json').exists():raise FileExistsError('Preserve publication')
    delivery=read(OUT/'final_delivery.json');quality=read(OUT/'quality.json');reg=read(OUT/'verified_TRAIN_mastery_registry.json')
    support=ROOT/'artifacts/v143_fine_support_evidence_20261001';audit=read(support/'audit.json');detail=read(support/'support_detail.json')
    assert delivery['issue_solved'] and not delivery['quality_acceptance'] and not delivery['model_promoted']
    assert reg['protected_correct_TRAIN_role_rows']==225558 and not audit['second_issue_solved']
    summary='V142完成3次既有第二层拟合、600次全角色梯度评估、295次接受更新；注册纯TRAIN错12→0，修复12、新增0，第一问题在注册M/S训练范围通过。冻结225558条正确TRAIN角色记录（112779独立原行），后续联合旧guard保护。ASA M/S错1910/1633，完整2056871行质量失败、无模型晋升。V143仅做细支持审查，0拟合0更新；第二和第三问题未通过。'
    metrics=quality['task']['full_task']['B'];lines=[]
    for c,name in [('0','N'),('1','M'),('2','S')]:
        a=metrics[c];lines.append(f"|{name}|{a['support']}|{a['missed']}|{a['precision']:.9f}|{a['recall']:.9f}|{a['f1']:.9f}|")
    report='''# V142：注册训练分类通过，完整质量仍失败

2026-10-01。'''+summary+'''

## 实际训练、输入与计算边界

V141先从全部合法TRAIN原行残错诊断获得既有第二层可改善分类间隔的梯度证据，0拟合；V142随后登记并执行三个来源角色的S2候选。所有角色从预登记的V140 C固定终点开始，C只用于训练保护初始化与匹配参照，不后验晋升为V140候选。输入、原行频次和成员CE保持；第一层、读出及facts_direct冻结，仅22528个既有第二层参数适应，新增容量0。

计算图明确为 `h2_original_float32 + relu(second_adapted_double(h1)) - relu(second_reference_double(h1))`，再接冻结读出。数值补偿保留零步原模型函数，不应描述为普通第二层直接替换。无缓存推理入口training/v142_infer.py接收任意行数、注册66287维数值输入，不查rowID或本次预测缓存；未声称新原始文本预处理或外部部署已验收。

三次拟合各200次完整原行频次梯度，接受更新98/98/99；预算终点，不证明收敛或全局最优。超预算试探回滚，未挑中间状态、种子或改变门槛。原末层链两轮12拟合已耗尽，本轮新增末层拟合0；新第二层范围三拟合已执行，不自动追加。

运行前封存3511项实际物理模块、第三方源码/二进制、Torch DLL、Python运行时、数据/划分及历史模型依赖。GPU驱动与OS未作字节快照，非物理动态元数据单列。旧模型与封存保持；封存不是自动依赖发现证明。

## 第一问题的实际验收与能力冻结

|训练角色|起点纯TRAIN错|终点纯TRAIN错|完整M错|完整S错|已冻结正确原行|
|---|---:|---:|---:|---:|---:|
|0|12|0|0|22|92315|
|1|0|0|0|6|40951|
|2|0|0|0|28|92292|

纯M/S TRAIN角色记录225202条，对应112601条独立官方原行，全部正确。完整S剩22/6/28条为当前数值输入异标签冲突的经验下限，混标原行仍参与损失和评分，没有删除或改标签。该结论只关闭注册TRAIN-PURE-READOUT问题，不关闭完整三分类或来源泛化。

终点与各末五个实际不同参数状态独立重放；每个模型22546种数值输入通过无缓存推理函数重放，概率最大差不超过3.34e-16，分类完全一致。第一层/固定读出/参考层重放与参数身份通过，旧V138/V140范围均无新增错。官方label_binary独立真值账本及父对话只读重算也通过；父对话没有重复模型前向。

verified_TRAIN_mastery_registry.json是新保护唯一注册正文：225558条正确角色记录含混标多数M，112779条独立原行。后续调用training/v142_retention_check.py，它先调用v140/v138保护再检查三新范围。实际S2通过；旧V140 C在旧范围通过却在新范围错12条，会被拒绝。冻结模型不等于冻结后续预测正确性，必须实际重放保护。

新验收测试曾错误地按local编号分组，得到22/4/26。input_identity_audit.json重新散列全部真实稀疏索引和值，验证不同编号确有完全相同数值输入，跨编号异标签冲突补足22/6/28；保留冲突原行证据并修正测试分组，不变更训练或验收门槛，新增拟合0。

## 来源外与完整三分类

|ASA参照|M错/78748|S错/34059|
|---|---:|---:|
|任务参照A0|318|2074|
|上一轮登记候选V140 E|2036|1623|
|本轮匹配零更新参照V140 C|1982|1635|
|V142 S2|1910|1633|

相对匹配C：M修复136、新增64；S修复14、新增12。相对上一候选E：M少错126、S多错10，不能把1635→1633写成较上一候选的S改善。匹配C的三折总错变化+2/-90/+14，仅一折改善；root2868少错76超过全体净少错74，仍错1040/1184。root27221仍错685/685。完整质量保持原A0参照与既有门槛。

|类|完整原行|漏判|precision|recall|F1|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

完整总错3650，A0为2499；非ASA冻结组件107错。M、总错误、全类precision/recall/F1保护、多折改善、时间头682与root2868门槛失败。只读账本重算、数值前向重放和TRAIN能力各自只证明覆盖范围。开发折已反复查看，不是盲测、外部验收或正式提交，无模型晋升。

## 第二问题推进与下一步

V143已实际完成全112807条ASA原行、正确对照和10620条VPC原行的零拟合支持资格审查，见docs/V143_FINE_SUPPORT_QUALIFICATION.md。第一问题已经闭合，不继续同配置延长训练。第二问题须先区分目的参数投影、完整可观察事实、掩码相等、同类/反类真实来源和正确对照，再审查限定行为条件的跨来源配对监督；当前没有新配对损失入口、辅助系数或执行封存，不发起额外分类器拟合。

所有合法原行、未知参数及真实标签频次保留，HELD标签/错误不进入目标、权重、阈值或选模。不能用粗行为/端口含义复制M/S标签，不能重采样伪装补齐独立来源。第二、第三问题及完整三分类仍开放，目标保持active。

实际证据artifacts/v142_second_layer_training_20261001/；独立真值审查artifacts/v142_independent_review_20261001/。前置机制研究与取舍见docs/V142_SECOND_LAYER_PRETRAIN_DECISION.md；本报告记录最终实际行为。
'''
    target=ROOT/'docs/V142_SECOND_LAYER_TRAINING_RESULTS.md';target.write_text(report,encoding='utf-8')
    support_doc=ROOT/'docs/V143_FINE_SUPPORT_QUALIFICATION.md'
    support_doc.write_text('''# V143：细行为支持资格审查，零新增拟合

2026-10-01。最新实际模型仍V142；本次0拟合、0更新、0新增梯度。不直接进入下一轮分类器训练。已覆盖全部112807条ASA、正确对照及10620条VPC官方原行，支持严格来自查询折之外，根来源隔离；未丢弃未知或解析失败行。

## 原始人口与正确对照

独立审查发现：S完整事实无同类TRAIN支持32831条，错1547（4.71%）；多来源支持1066条，错56（5.25%）；单来源162条，错30（18.52%）。无支持人口占96.40%，不能凭1547这个数判断缺支持是错误主因。

1633条来源外S错全部已有粗行为同类多来源支持；另5900条无这种支持的S全部判对。不能把粗支持数量作为稳定M/S边界。

本次显式排除未知参数作为具体共享值后，目的参数投影结果：

|S目的参数支持|原行|错|错率|
|---|---:|---:|---:|
|多同类来源|1673|38|2.27%|
|单同类来源|368|164|44.57%|
|无同类来源|31140|1257|4.04%|
|参数未观测|878|174|19.82%|

目的投影保留action/outcome、协议、接口角色、目标端口及ICMP type/code；省略源端口和其他事实，仅用于分析，不替代模型输入。多个同类来源组中1124条亦有反类支持。三TRAIN角色这个投影仍有289/69/285个异标签键，经验最小错2536/538/2304，不能用同一个目的键直接映射标签。

## 真正细配对还缺什么

完整body事实、已知目的参数、同类跨根来源且不同数值输入的合法正对组，M为255/167/253组，S仅13/4/13组；这些S组全部是TCP/UDP且body源端口被遮蔽。要求完整事实且body双端口已知后，M仅6/2/1组，S三个角色均0组。该严格粒度的零覆盖不能推广成所有可共享行为都没有正对；它直接反驳“掩码相同就是相同细参数”的用法。

记录src_port是独立观察字段，现有输入已保留，不能默认为body源端口替代。719条已知参数但无record端口的S属于ICMP，错697；ICMP本来没有传输端口，不能称为漏编码source-port。root27221的type3/code13缺对应TRAIN细组合，粗行为近邻的标签不能直接复制。

## 其他官方格式不是现成细支持

全部10620条VPC原行均S；各合法TRAIN补集7788/5428/8024条，保留未知/解析未观测记录。能解析REJECT且TCP/UDP目标端口的有效行5900/4484/6136。

按协议与目标端口查找ASA留出，匹配2176/2904/2680行，其中M为2168/2824/2676，S仅8/80/4。VPC缺少ASA src/dst接口角色，没有等价完整细语义与反类VPC证据，因此不把这些S记录当作ASA M/S标签规则或已补齐细支持。技术协议含义不授予比赛类别。

## 当前决定与后续资格

第二问题尚未通过；本次排除了盲补频次、粗行为标签映射、未知掩码合并和直接混入VPC的候选。下一阶段只审查“已观测目标参数条件的跨来源配对”是否成立：限定粒度，保留完整输入与未知掩码，核对同类跨来源正对、邻近反类及可观察冲突；实际检查V142表示间隔、来源集中与辅助梯度对已验收分类的影响。目的投影配对若成立也不能宣称补齐完整细事实。

如资格成立，再注册一次三折匹配CE/辅助配对受控对照及新入口/封存；当前尚无此训练器、辅助系数或新执行封存，不新增拟合。第一问题保护225558条角色正确记录与旧范围，新候选必须调用v142_retention_check，来源外验收与原任务A0门槛保持。无资格的切片保留为证据缺口；不无限延长既有第二层训练。

证据：artifacts/v143_fine_support_evidence_20261001/audit.json、fine_support_population.parquet、VPC_observed_originals.parquet、support_detail.json、legal_full_fact_positive_groups.parquet；来源外标签仅用于诊断，非训练权重/选模。独立对照见docs/V142_INDEPENDENT_REVIEW_AND_NEXT_DIRECTION.md。
''',encoding='utf-8')
    # Current rule adds the proven scope, without rewriting historical run seals.
    rules=ROOT/'docs/EXPERIMENT_REVIEW_RULES.md';old_rules_hash=sha(rules)
    rules.write_text(rules.read_text(encoding='utf-8')+'''

## V142已验收能力与V143支持审查边界

第一问题仅在注册M/S TRAIN范围通过：三角色纯错0，完整M错0、S22/6/28达到真实数值冲突经验下限。后续必须调用training/v142_retention_check.py，保护225558条全部正确TRAIN角色记录（112779独立原行），并联合旧v140/v138范围；旧C在旧范围通过但新范围新增12错的真实反例见test_v142_mastery_evidence.py。

同local编号不等于唯一实际输入。分类冲突必须由真实输入字节身份重算，不按索引号误计；v142_input_identity_audit.py已重放全部22546种数值输入。未知参数哨兵不能作为已观测细值相同；ICMP不需要传输端口。已支持切片、无支持切片均须保留正确对照，支持计数不作为错误因果证明；V143零拟合证据不替换最新实际V142，也不授予下一拟合入口或模型晋升。
''',encoding='utf-8')
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    catalog['project'].update(as_of='2026-10-01',current_summary=summary,
        current_direction=['第一问题在注册TRAIN范围闭合，不继续同配置延长训练；完整质量仍失败。',
        '后续用v142_retention_check保护225558正确TRAIN角色记录和旧V138/V140范围。',
        '当前进入第二问题的细支持/共享判据资格审查；V143已零拟合排除粗映射、掩码合并及直接VPC补标。',
        '限定已观测目的参数跨来源配对仍需表示/梯度资格、注册新入口与封存，当前不拟合；第三问题保持开放。'],
        authoritative_delivery_id='v142-delivery',authoritative_direction_id='v142-review',
        known_limits=['注册TRAIN通过不代表完整任务通过，来源开发折反复查看，不是盲测或外部部署。',
        'V142相对匹配C少错2S，相对上一候选E多错10S；只一折总错误改善。',
        '正常/其他格式冻结，非ASA仍107错，完整三分类质量失败，无模型晋升。',
        '原末层12拟合耗尽；本轮第二层3拟合也不自动重复。',
        '严格已知双端口完整事实的跨源S正对为0；目的参数投影支持不等于完整细事实支持。'])
    entries=[('v142-review',target,'V142实际训练通过范围、全量失败与当前方向','current_direction'),
             ('v142-delivery',OUT/'final_delivery.json','V142三拟合与完整质量实际交付','execution_delivery'),
             ('v142-train-mastery',OUT/'verified_TRAIN_mastery_registry.json','V142三角色完整正确TRAIN保护','review_evidence'),
             ('v143-support-review',support_doc,'V143细支持与可共享判据资格审查，零拟合','review_evidence'),
             ('v143-support-evidence',support/'audit.json','V143全量细参数支持及跨格式原始证据','review_evidence'),
             ('v143-support-detail',support/'support_detail.json','V143协议参数及完整事实正对覆盖','review_evidence')]
    for ident,path,title,category in entries:
        assert ident not in {v['id'] for v in catalog['documents']}
        catalog['documents'].append(dict(id=ident,path=path.relative_to(ROOT).as_posix(),title=title,category=category,summary=summary,
            sha256=sha(path),keywords=['V142','V143','实际训练','支持','TRAIN','保护']))
    for entry in catalog['documents']:
        if entry['path']==rules.relative_to(ROOT).as_posix():entry['sha256']=sha(rules)
    save(catalog_path,catalog)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8').replace('最新实际与当前方向（V140）','历史实际（V140）',1)
    text=text.replace('# SOC 日志威胁检测项目\n','# SOC 日志威胁检测项目\n\n最新实际与当前方向（V142）：**'+summary+'** [实际训练与全量验收](docs/V142_SECOND_LAYER_TRAINING_RESULTS.md)；[V143零拟合支持资格](docs/V143_FINE_SUPPORT_QUALIFICATION.md)。\n',1)
    readme.write_text(text,encoding='utf-8')
    handoff=ROOT/'HANDOFF.md';(OUT/'handoff_before_v142.md').write_bytes(handoff.read_bytes())
    handoff.write_text('''# SF02续接：V142实际训练已完成，V143支持资格审查

2026-10-01。完整三问题目标与授权保持active。'''+summary+'''

先读AGENTS.md、docs/EXPERIMENT_REVIEW_RULES.md、docs/V142_SECOND_LAYER_TRAINING_RESULTS.md、docs/V143_FINE_SUPPORT_QUALIFICATION.md及只读MCP最新实际。项目C:\\Users\\xiabutian\\Desktop\\人工智能算法挑战杯\\SF02；.venv-v61 CUDA可用。父对话只读，执行对话独立负责训练。不得自动委派Agent。

第一问题在注册M/S TRAIN范围已通过，不能继续当作未学会再训练。225202纯角色记录（112601独立原行）零错；完整S冲突22/6/28已由真实输入字节身份重算。三终点与各末五不同状态重放、全部22546数值输入无缓存函数重放、旧保护与完整官方真值通过。所有正确TRAIN新guard为225558角色记录、112779独立原行，唯一新注册verified_TRAIN_mastery_registry.json；后续必须用v142_retention_check.py联合旧v140/v138。旧C在新范围会新增12错，应拒绝。

最新实际V142：3既有第二层拟合/600完整梯度/295接受更新，第一层和读出冻结。真实函数含float32原表示与double参考差分补偿；运行前3511物理依赖封存。原末层12拟合耗尽，本第二层3拟合也不自动重复，未新增头拟合/容量。正式A0和质量门槛保持。

ASA1910/1633错：较匹配C少72M/2S；较上轮候选E少126M但多10S。C三折总错变化+2/-90/+14，单折改善；root2868仍1040错，root27221仍685错。完整2056871行总错3650，A02499，非ASA107；全类与来源门槛失败，V142不晋升。

V143已实际完成0拟合0梯度支持审查：全ASA正确/错误对照、known目的参数、both-label/TRAIN来源、全部VPC、完整事实正对可用性。S目的参数单来源错164/368，多来源38/1673；所有1633S错已有粗同类多源支持。完整事实known目的S多源正对13/4/13组全masked body srcport；严格双端口known的完整事实S正对0/0/0，不能把masked相同称相同细值。719已知参数无record源端口S均ICMP，不能称端口遗漏。VPC协议端口匹配ASA多数M，缺接口等价语义，不补标。

当前第二问题仍未解决；下一独立工作是限定已观测目标参数跨来源正负对的事实兼容、表示间隔/来源集中和辅助梯度资格，不从HELD错例改权重/选系数，不把粗/端口映射为M/S。新辅助训练须先完成真实细配对资格、预算/目标/终点检查、入口/封存与零步重放，当前无新入口，不直接拟合。缺严格细支持不能由重复抽样创造；保留正确对照、未知、冲突、全部频次。第三问题与完整三分类审查仍开放。

证据artifacts/v142_second_layer_training_20261001/、v142_independent_review_20261001/、v143_fine_support_evidence_20261001/。父对话创建training/v142_independent_ledger_review.py与docs/V142_INDEPENDENT_REVIEW_AND_NEXT_DIRECTION.md，执行不要覆写。新test_v142_mastery_evidence验证模型guard正例及旧C负例；local分组测试错误已记录并依据actual稀疏字节身份修正，0新拟合。模型和旧封存不改。
''',encoding='utf-8')
    save(OUT/'publication_receipt.json',dict(status='published_actual_V142_and_zero_fit_V143',report_sha256=sha(target),support_report_sha256=sha(support_doc),
        catalog_sha256=sha(catalog_path),rules_before_sha256=old_rules_hash,rules_after_sha256=sha(rules),source_sha256=sha(__file__),
        new_training_after_terminal=0,cloud_MCP_verified=False))
    print(summary)


if __name__=='__main__':main()
