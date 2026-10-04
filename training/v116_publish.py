"""Publish executed experiment, metrics and explicit stop decision."""
import json
from v116_preflight import ROOT, DEST, save, sha


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    e=read(DEST/'evaluation.json');v=read(DEST/'independent_verification.json')
    drift=read(DEST/'baseline_drift_audit.json')
    assert drift['evaluation_sha256']==sha(DEST/'evaluation.json') and drift['optimizer_steps']==0
    assert v['all_checks_passed'] and v['evaluation_sha256']==sha(DEST/'evaluation.json')
    assert not e['all_gates_passed'], 'A passing result requires registered seed replication before publication.'
    assert len(list(DEST.glob('*_fold*/fit.json')))==6
    a=e['ASA']['A'];b=e['ASA']['B'];c=e['ASA']['class_comparison'];epochs=e['selected_epochs']
    report=ROOT/'docs/V116_NESTED_SELECTION_EXECUTION_AND_STOP.md'
    assert not report.exists()
    summary=(f"V116按V115完成6次新拟合（3次内层、3次外层），0次校准；质量未通过、无模型晋升。"
        f"内层锁定轮次为{epochs['0']}/{epochs['1']}/{epochs['2']}；ASA错误{a['errors']}→{b['errors']}，"
        f"M错误{c['M']['A_errors']}→{c['M']['B_errors']}，S错误{c['S']['A_errors']}→{c['S']['B_errors']}。"
        "全部关键检查点、K/U分层与逐行修复退化已记录；停止本选模分支，不继续排列epoch或权重。")
    class_table='\n'.join(f"| {name} | {q['support']:,} | {q['A_correct']:,} | {q['B_correct']:,} | {q['repaired']:,} | {q['regressed']:,} |" for name,q in c.items())
    fold_table='\n'.join(f"| {k} | {epochs[k]} | {q['A']['errors']} | {q['B']['errors']} |" for k,q in e['ASA']['per_fold'].items())
    task_rows=[]
    lock=read(DEST/'locked_selection.json')
    for k,q in lock['folds'].items():
        for task in ('K','U'):
            z=q['selected_tasks'][task]['class']
            task_rows.append(f"| {k} | {task} | {q['selected_epoch']} | {z['malicious']['correct']}/{z['malicious']['support']} | {z['suspicious']['correct']}/{z['suspicious']['support']} |")
    task_table='\n'.join(task_rows)
    gate_table='\n'.join(f"| {k} | {'通过' if val else '未通过'} |" for k,val in e['gates'].items())
    full_rows=[]
    for cls in ('benign','malicious','suspicious'):
        aa=e['full_task']['A']['class'][cls];bb=e['full_task']['B']['class'][cls]
        full_rows.append(f"| {cls} | {aa['support']:,} | {aa['correct']:,} | {bb['correct']:,} | {aa['f1']:.6f} | {bb['f1']:.6f} |")
    seconds=sum(q['seconds'] for q in e['fits'])
    text=f'''# V116：来源/参数分层选模训练结果与停止决定

日期：2026-09-29。执行范围：V115 第六节固定 N1、SparseTabM、原频次 CE、25 epochs 上限的内层 K/U 选模对照。**本机 GPU 已完成全部六条训练轨迹；未使用平台，未添加外部数据，未调用官方未知答案。**

{summary}

## 1. 执行及信息隔离

- 划分策略在统计候选人口之前写入 `split_policy.json`，只尝试一次。U 按协议参数的固定 SHA256 桶留出，并移走整个关联组；K 从剩余关联组固定留出，只评价拟合侧见过的参数。
- 源/相同输入完整隔离；U 真实参数不在拟合中。未知值不能充当已知参数。两类均至少两个来源的规则只保证最基本重复观察，不是统计把握度保证。
- 三折内层拟合原始行数分别为 55,944、23,504、55,591。连带移出的其他行全部保存在 manifest，不能混回内层拟合；外层重训恢复其完整合法训练人口。
- 固定 epoch 候选为 1/2/5/10/15/20/25。K/U 两项的 M/S 判对数均不得低于本项第 25 epoch，之后按较差任务的 M/S F1、来源宏平均召回、较早轮次顺序择优。
- **全部内层选轮先锁定，再开始任何外层重训。** 三次外层轨迹同时提供固定 25 的 A 与锁定轮次的 B；没有复用旧模型充当本轮 A。
- 原始记录频次、冲突标签计数、输入及网络均保留；没有添加权重、阈值校准、域对齐或额外重采样。六条轨迹累计记录耗时约 {seconds:.1f} 秒，不含准备、模型重放与独立验证。

## 2. 内层关键数据

| 外折 | 验证任务 | 选中 epoch | M 判对/总数 | S 判对/总数 |
|---|---|---:|---:|---:|
{task_table}

K 是已知参数的新来源，U 是训练侧未见具体参数。第 1 折 U 只有 34 条 S，来自 15 个关联组；第 15/20/25 epoch 类别判对数相同，按已锁定平手规则选择 15。它的 U-S 仍仅 2/34，不能把“选择通过”解释为“S 已学会”。第 0、2 折只有第 25 epoch 通过四项保护。

## 3. ASA 成对外层结果

| 类别 | 总数 | A 判对 | B 判对 | 旧错修复 | 原对变错 |
|---|---:|---:|---:|---:|---:|
{class_table}

| 外折 | B 轮次 | A 总错误 | B 总错误 |
|---|---:|---:|---:|
{fold_table}

全量 ASA A 错误 {a['errors']:,}，B 错误 {b['errors']:,}。重新训练的 A 与旧 V107 N1 相比，逐行预测变化 **{e['historical_baseline_check']['fresh_A_vs_V107_decision_changes']} 条**。这项历史一致性检查不能替代本轮成对比较，但能帮助排除基线漂移。

此处检查**实际发现了基线漂移，不能声称历史基线完全复现**。本轮 A 的 M/S 错误为 294/2,434，旧 N1 为 318/2,094；因此本轮 B 的因果比较使用同时训练的 A，历史绝对门槛另列。全部训练行、标签、频次和特征哈希已核对一致；模型与损失、种子、批次顺序设置相同。

零优化步数的数值复核发现：同一初始化模型、同一小批次，连续三次前向/反向，参数始终不变，却出现最大约 1.49e-8 的 logit 差异及最高约 1.91e-6 的未归一化梯度差异。当前运行时确有数值非确定性；**单批探针不能证明全部 364 条历史漂移均由此造成**。后续实验若要解释很小的收益，需先把确定性运行或数值重复性的边界纳入预注册；本轮不在六次拟合之外私自补训或挑更有利的重复。

完整 2,056,871 条官方训练行的开发回放，非 ASA 模型和正常路由保持冻结，得到：

| 类别 | 总数 | A 判对 | B 判对 | A F1 | B F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(full_rows)}

详细来源、参数支持类别和逐行负翻转分别见 `source_group_changes.csv`、`support_strata_results.csv`、`OOF_ASA_decisions.parquet`；所有检查点类别曲线见 `checkpoint_class_metrics.csv`。

## 4. 第一性原理与反证审查

**事实一：验证覆盖存在，不等于验证任务已可学好。** 第 1 折 K/U 都有 M/S 和多个来源，U-S 仍只对 2/34。内层选模机制能够选择已有检查点，却不能产生这些检查点从未学会的判别依据。

**事实二：内层平台期不保证外层重训仍处于同一学习阶段。** 内层数据量、S 占比、独立输入数与外层不同，原样迁移 epoch 数仅保证相同遍历次数，不保证相同优化步数或学习状态。V115 明确指定按 epoch 迁移，本轮没有事后改成步数或调权来追外折成绩；这项限制作为问题记录，不能事后声称已修好。

本轮少错的 24 条 M 全在关联组 2868；新增 2,542 条 S 错误中，2,528 条来自组 29，其余 14 条分布在另外六组。这说明第 15 epoch 并未保留第 25 epoch 对这些 S 的判断能力。K/U 第 1 折的弱 S 基线加上“平手选较早”，允许通过相对保护，但并没有提供足够的绝对 S 能力保证。该缺陷不能在查看外折后靠修改平手规则修饰成本轮成功。

**事实三：本轮最多只有一折能发生预测变化。** 第 0、2 折 B 与 A 都来自同一轨迹的第 25 epoch；因此在选轮锁定后就可推导，本候选不可能满足“至少两折改善”的晋升条件。仍完成已注册的三次外层匹配轨迹，用于实际测量第 1 折效果与复核基线；不启动追加种子。

**事实四：训练、选模、质量通过是不同结论。** 五项训练/选模单元检查验证逐类保护与频次损失；独立脚本用 sklearn 重算全部内层检查点指标和选轮，用模型重放核对外层预测。它们证明程序和记录一致，不证明候选达到了跨来源目标。

本轮未把已知外折错例做训练权重、未针对真实标签决定推理路由，也未把缺支持记录排除于主分数。

## 5. 验收与停止决定

| 原冻结门槛 | 结果 |
|---|---|
{gate_table}

**质量未通过、无模型晋升。** 保留旧 N1 开发参照，停止这条 epoch 选择分支；不继续排列新轮次、修改平手规则、改变 K/U 划分或增加权重。该结果只否定本次锁定的选模方案，没有证明所有早停或所有模型都无效。

仍待解决的问题是未见参数的 M/S 依据、同事实异标签的真实上下文，以及有支持区域的跨来源学习。后续只有新的可验证行为/上下文证据或具体机制对照，才应开启另一项实验；本轮不以修审计规则冒充分类提升。

## 6. 文件与核验边界

- `split_policy.json`、`inner_split_manifest.parquet`、`preflight.json`：一次性划分规则、全部行角色、支持覆盖。
- `training_registration.json`、`locked_selection.json`：训练身份绑定与外层开始前的固定选轮。
- `inner_fold*/`、`outer_fold*/`：每次拟合、逐轮损失、检查点模型/概率与实际耗时。
- `evaluation.json`、`verification.json`、`independent_verification.json`：逐类全量结果、模型重放、独立复算。
- `baseline_drift_audit.json`：原始训练人口逐项一致性、无参数更新的数值重复性探针、修复与退化来源。
- `training/v116_preflight.py`、`v116_train.py`、`v116_evaluate.py`、`v116_verify.py`、`test_v116.py`：执行与检查源码。

上述结果均来自反复使用的官方开发折；没有新的独立来源盲测，也没有训练或发布替换生产模型。模型和数据未删除。
'''
    report.write_text(text,encoding='utf-8')
    delivery=DEST/'delivery.json'
    assert not delivery.exists()
    paths=[report,DEST/'evaluation.json',DEST/'verification.json',DEST/'independent_verification.json',DEST/'baseline_drift_audit.json',
           DEST/'locked_selection.json',DEST/'training_registration.json',ROOT/'training/v116_publish.py']
    save(delivery,{'status':'v116_nested_selection_executed_failed_quality_gates',
        'quality_acceptance':False,'model_promoted':False,'classifier_fits_new':6,'calibration_fits':0,
        'baseline_fits_reused':0,'validation_scope':'Registered one-split nested K/U selection; six real fits, matched checkpoints, full developmental replay and independent recount. No new blind transfer validation.',
        'summary':summary,'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}})
    for name,link in [('README.md','docs/'+report.name),('docs/TRAINING_PLAN.md',report.name)]:
        p=ROOT/name;old=p.read_text(encoding='utf-8');assert report.name not in old
        old=old.replace('当前稳定判据审查与训练方案（V115，未新增训练）','历史训练方案（V115；已由V116执行）',1)
        head,body=old.split('\n',1)
        p.write_text(head+'\n\n当前训练结果与停止决定（V116）：**'+summary+'** [训练数据、问题与核验]('+link+')。\n'+body,encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp)
    cat['project'].update(as_of='2026-09-29',current_summary=summary,
        authoritative_delivery_id='v116-delivery',authoritative_direction_id='v116-stop-decision',
        current_direction=['保留N1开发参照；V116执行完成但质量未通过，无模型晋升。',
            '停止当前epoch选模分支，不事后搜索轮次、切分、平手规则或权重。',
            '先明确未见参数和同事实异标签所需的真实判别证据；本轮未开启新的模型方向。'],
        known_limits=reg_limits())
    for doc in cat['documents']:
        if doc['id'].startswith('v115-'):doc['category']='historical_review'
        if doc['path'] in ('README.md','docs/TRAINING_PLAN.md'):doc.update(summary=summary,sha256=sha(ROOT/doc['path']))
    additions=[('v116-stop-decision',report,'V116 内层选模训练结果与停止决定'),
               ('v116-delivery',delivery,'V116 六次训练执行凭据'),
               ('v116-evaluation',DEST/'evaluation.json','V116 全量逐类与来源评价')]
    cat['documents']=[{'id':i,'title':title,'path':p.relative_to(ROOT).as_posix(),
        'category':'current_review' if i=='v116-stop-decision' else 'current_evidence',
        'summary':summary,'sha256':sha(p),'keywords':['V116','当前','训练','结果','M/S','停止决定']}
        for i,p,title in additions]+cat['documents']
    save(cp,cat)
    p=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';txt=p.read_text(encoding='utf-8')
    txt=txt.replace('v115-direction-review','v116-stop-decision').replace('v113-delivery','v116-delivery')
    txt=txt.replace('v113_case_only_fits_executed_failed_quality_gates','v116_nested_selection_executed_failed_quality_gates')
    txt=txt.replace("'3次新拟合'","'6次新拟合'").replace("'共6次匹配拟合'","'3次内层、3次外层'")
    p.write_text(txt,encoding='utf-8')
    print(json.dumps({'published':str(report),'summary':summary},ensure_ascii=False),flush=True)


def reg_limits():
    return ['内层第1折U仅34条S，已记录低支持的不稳定性。',
            '第0/2折选择25，只有第1折有变化，不能形成两折改善。',
            '内层选轮迁移到全训练池改变优化步数和类别比例；不代表新M/S知识。',
            '全部旧外折已反复查看；检查通过不代表泛化质量通过。']


if __name__=='__main__':main()
