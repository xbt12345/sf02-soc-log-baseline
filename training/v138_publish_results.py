"""Publish evidence-backed completed run without changing sealed training sources."""
from v138_runtime import ROOT,OUT,read,save,sha,require_run_seal


def main():
    require_run_seal(ROOT/'training/v138_train.py')
    d=read(OUT/'final_delivery.json');a=d['training_attribution'];q=read(OUT/'quality.json')
    report=ROOT/'docs/V138_SINGLE_ISSUE_TRAINING_RESULTS.md'
    if report.exists():raise FileExistsError(report)
    probes=d['probe_receipts'];cap=d['scoped_training_capabilities'];pair=d['original_row_pairs']['ASA_vs_V135_R_decay']
    lines=[
        '# V138：V137单问题第一轮实际训练、回归及停止记录',
        '',
        '2026-09-30。本机CUDA执行，第一轮6次末层拟合、1200次全角色梯度评估、891次实际接受的参数更新。另有6次零步诊断梯度与2折有界监督可行性诊断；诊断不充作分类器成功。最新实际训练为V138，质量未通过，无模型晋升。',
        '',
        '## 结果与第一性原理判断',
        '',
        '**单问题有实质改善但未完全解决。**冻结原输入、全部官方标签频次及主体网络，只更换末层求解方式，纯TRAIN错误80→16，其中恶意4→0，可疑76→16；候选修复64次训练角色错误，原正确纯输入新增错误0。剩余为16条独立官方原行、8种当前输入、8个来源。不能将80次角色计数与56条独立原行混为一谈。',
        '输入66,287维、隐藏128维、16成员与495事实保持不变；仅更新7,677个原末层参数。',
        '',
        '|训练角色|原R_decay纯错误|匹配Adam纯错误|候选L-BFGS纯错误|L-BFGS有效更新/梯度评估|',
        '|---|---:|---:|---:|---:|',
        '|0|42（4M/38S）|30|14|98/200|',
        '|1|6（0M/6S）|4|0|96/200|',
        '|2|32（0M/32S）|16|2|97/200|',
        '|合计|80|50|16|291/600|',
        '',
        '匹配Adam总计600次更新/600次全量梯度；两臂总计891次更新/1200次梯度。两者起点、损失、频次、参数范围一致。L-BFGS闭包次数与更新次数不同，不能将1200称为参数更新数。第一折及第三折仍有纯S错，不能注册整个TRAIN-PURE-READOUT问题已解决。',
        '',
        '第二个训练角色在最后五个实际不同的已接受状态上都满足纯M/S零错、全M零错、全S仅6条混标经验最低错。其纯TRAIN保护集40879条原行已固定，保存原行ID、独立真值、模型、运行绑定和适用范围。它只是该训练角色范围内能力，不是未知来源能力。后续同路径或其他候选必须调用`training/v138_retention_check.py`重放，新增错即拒绝替换；当前整个任务参照保持。',
        '',
        '## 来源外真实逐类结果：不允许净收益掩盖回退',
        '',
        '|ASA来源留出|原A0|V135 R_decay|本轮Adam|本轮L-BFGS|',
        '|---|---:|---:|---:|---:|',
        '|恶意漏判/78748条M|318|1748|1934|2006|',
        '|可疑漏判/34059条S|2074|1655|1643|1637|',
        '|ASA总错|2392|3403|3577|3643|',
        '',
        f'相对V135，M修复{pair["1"]["repairs"]}条、新增{pair["1"]["new_errors"]}条，净多258错；S修复{pair["2"]["repairs"]}条、新增{pair["2"]["new_errors"]}条，净少18错。匹配Adam比较中，候选M多72错，S少6错，类别保护失败。来源组2868 M错1040→1116，不能被总体S改善覆盖。S来源均值召回为18.0623%，零召回来源185个，这些局部改善不授予晋升。',
        '',
        '已从独立官方真值对全部2056871条原行计算三类precision/recall/F1、逐折和来源条件；非ASA继续为冻结组件107错。不是只计困难面板、不是全判正常总体准确率、没有删除未知参数或冲突记录。全部原正确判断的回退单列保存。',
        '',
        '**因果结论的边界：**输入和主体网络不变仍修复64次训练错误，证明这些错误不是必须换输入或扩网络才能修复；求解方式是本轮可操作的影响因素。但训练侧更接近其监督目标，同时换来源的M判决变差，说明更充分地拟合当前训练分布不自动产生跨来源稳定M/S判据。不能据此断言剩余16条一定是表达不足，或官方标签一定错误；本轮尚未取得完整可表达性证明。',
        '',
        '## 可行性诊断、第二轮门槛与问题记录',
        '',
    ]
    for p in probes:
        lines.append(f'- 第{p["fold"]}折：完整纯输入margin证书={p["certificate"]}；终止={p["termination"]}；实际耗时约{p.get("seconds",p.get("seconds_including_imports",0)):.3f}秒。只用合法TRAIN标签，诊断解不作为候选。')
    lines.extend([
        '',
        '尚未取得所有所需角色的全人口证书；第一折生成的局部LP解曾出现极端数值，最后触发求解时间限制。局部约束满足不等于全人口满足；单成员硬margin充分条件无解/超时，不证明平均概率分类器不可能正确。没有所需完整纯输入证书，更没有包含混标多数判决的完整训练可行证书，因此第二轮不符合启动条件，本次不执行。禁止追加同样无约束拟合、按来源结果改阈值或降低验收标准。',
        '',
        '前置技术故障：PyTorch动态模块的不存在`_classes.py`路径被误列为源码，封存前中止，0拟合0更新。筛选真实文件后对完整缓存封存，未恢复任何失败拟合、未覆盖旧证据。过程及失败日志保留。',
        '',
        '预算问题必须如实记录：求解器600秒限额属于调用内限制，实际进程/退出耗时存在开销，第一折核心记录600.421秒、包含启动约604.172秒。不能把它称为严格600秒墙钟成功；该超限不允许增加梯度预算或授权第二轮。后续诊断需在求解器限额中预留退出余量，Windows虚拟环境启动器进程树也需实测验证。',
        '',
        '第1折L-BFGS在最后一次线搜索达到200次梯度评估后，未接受的试探参数被回滚。实际完整预测与最后接受状态独立重放通过，终点未按分类分数选择。日志末次试探损失不冒充终点损失，终点成员CE已无梯度重新计算。',
        '',
        '## 冻结成果与后续决定',
        '',
        '保留三折全量缓存/起点、两臂六终点、末五个实际状态、全部合法训练原行预测、剩余16条及修复64次角色清单、2056871条完整留出账本、梯度与更新日志、来源错误、绑定及监督诊断失败。模型质量不合格，所以不替换任务参照，不将训练范围成功写成任务成功。',
        '',
        '当前状态：一轮已消耗、最多两轮约束不变；问题未关闭、第二轮资格未通过。后续若重新取得真实全人口可行性证据，才可按原计划实施有约束末层第二轮；否则停止这条求解路径，另立有独立证据的新机制问题，不通过换版本名称绕过预算。跨来源M/S仍是独立开放问题，本轮未将其混入拟合目标。',
        '',
        '证据目录：`artifacts/v138_single_issue_round1_20260930/`；实际交付`final_delivery.json`，验收`quality.json`，独立重放`verification.json`，训练/回退归因`attribution.json`，修复保护`scoped_training_capabilities.json`及`fold1_verified_TRAIN_guard.parquet`。',
    ])
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary='V138按V137单问题第一轮完成6次末层拟合、1200次全角色梯度评估、891次接受更新；纯TRAIN错误80→16（M4→0、S76→16），原正确纯输入新增错0。第1训练角色末五状态验收通过，40879条纯TRAIN保护。ASA候选M/S错2006/1637，较V135净多258M错、净少18S错；质量失败、无模型晋升。两折margin诊断未取得全人口证书，第二轮不执行。'
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    catalog['project'].update(current_summary=summary,current_direction=[
        'TRAIN-PURE-READOUT第一轮有改善但仍16条独立S原行、8输入未修复，整个问题保持开放。',
        '保护第1训练角色已验证范围；不以总体收益覆盖新增错，不把训练范围成功称为迁移成功。',
        '两折诊断无完整可行性证书，第二轮条件未满足，禁止追加同类无约束训练。',
        '保留完整原行及模型账本；任务参照未替换。'],authoritative_delivery_id='v138-delivery',authoritative_direction_id='v138-review')
    catalog['project']['known_limits']=[
        '开发来源已反复查看，无独立盲测、比赛提交或真实网络迁移验收。',
        '训练角色错误次数不等于独立原行数；范围内零错不证明安全语义。',
        '两折有界监督诊断未取得完整证书；未证实剩余错误必然由模型容量导致。',
        'Windows求解进程存在退出计时开销，实际超限已记录。',
        '只重训ASA末层，其他格式为冻结组件；无模型晋升。']
    for ident,path,title,category in [('v138-review',report,'V138单问题第一轮实际结果及停止决定','direction_review'),
        ('v138-delivery',OUT/'final_delivery.json','V138末层训练实际交付与全量验收','execution_delivery'),
        ('v138-scope-retention',OUT/'scoped_training_capabilities.json','V138已验证训练范围与后续原行保护','review_evidence')]:
        catalog['documents'].append({'id':ident,'title':title,'path':path.relative_to(ROOT).as_posix(),'category':category,
            'summary':summary,'sha256':sha(path),'keywords':['V138','单问题','实际训练','回归','M/S','停止决定']})
    save(catalog_path,catalog)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('# SOC 日志威胁检测项目\n','# SOC 日志威胁检测项目\n\n最新实际执行（V138）：**'+summary+'** [实际结果、单问题验收与停止记录](docs/V138_SINGLE_ISSUE_TRAINING_RESULTS.md)。\n',1)
    text=text.replace('当前方向（V137，未训练）','执行前方向（V137，现由V138执行）',1).replace('当前方向（V136，仅诊断与方案，未训练）','历史设计（V136，仅诊断与方案）',1)
    readme.write_text(text,encoding='utf-8')
    save(OUT/'publication_receipt.json',{'status':'actual_result_and_direction_published','report_sha256':sha(report),'catalog_sha256':sha(catalog_path),
        'new_training_after_closeout':0,'source_sha256':sha(__file__),'cloud_MCP_verified':False})
    print(summary,flush=True)


if __name__=='__main__':main()
