"""Publish verified local V75 results; no raw logs exposed through MCP."""
import json
from pathlib import Path
import pandas as pd
from run_v75 import ROOT,OUT,read,save,sha


def main():
    four=read(OUT/'four_arm/complete.json');follow=read(OUT/'corrective/complete.json')
    full=read(OUT/'official_replay/scoring.json');final=read(OUT/'F_diagnostic/scoring.json')
    audit=read(OUT/'analysis/summary.json');ca=read(OUT/'corrective/audit.json')
    verify=read(OUT/'release_verification.json');identity=read(OUT/'typed_identity_verification.json')
    assert verify['all_checks_passed'] and final['csv_ids_labels_and_original_order_match']
    scores={**four['curves'][-1]['validation'],**follow['curves'][-1]['validation']}
    def pct(x):return f'{100*x:.4f}%'
    rows=[]
    for a,z in scores.items():rows.append(f"| {a} | {z['errors']:,} | {pct(z['accuracy'])} | {pct(z['macro_f1'])} | {pct(z['recall'][1])} | {pct(z['recall'][2])} | {z['normal_false_alerts']} |")
    whole=[]
    for a,z in [('旧 v5.1',full['before']),('D 完整重训',full['after']),('F 稀缺优先诊断',final['totals']['F'])]:
        whole.append(f"| {a} | {z['errors']:,} | {pct(z['accuracy'])} | {pct(z['macro_f1'])} | {pct(z['recall'][1])} | {pct(z['recall'][2])} | {z['normal_false_alerts']} |")
    slices=[]
    for s in final['slices']:
        if s['D']['errors'] or s['F']['errors']:
            slices.append(f"| {s['route']} | {s['D']['rows']:,} | {s['D']['errors']:,} | {s['F']['errors']:,} | {s['fixed']} | {s['broken']} |")
    f=final['totals']['F'];d=full['after']
    report=f'''# v7.5：完整训练覆盖、四臂归因与稀缺切片实际执行

日期：2026-09-21。**本轮完成 8 次真实拟合：四臂 4 次、身份处理/稀缺加权 2 次、D/F 全量重训各 1 次；本地 CPU，只用官方数据。模型质量未通过，无新模型晋升。** F 是内部恶意召回保护失败后预先登记的完整任务诊断，不通过追加回放改写原验收门槛。

## 1. 对用户要求的执行结果

- **32,596 条 native_flow 恶意记录全部用上。** 这不是官方全部 M 的数量；全训练集 M 为 111,728。内部四臂先用 25,626 条拟合、6,970 条验证；D/F 最终均用全 32,596 条。每个全量模型包含全部 2,056,871 条官方记录、6 个完整 epoch，原事件曝光 12,341,226 次；未去重删行。相同文本只缓存编码，不合并原记录的监督。
- 全量记录、event_id、标签和原文件哈希已核对。17 个解析路由 × 3 类都出审计表，包括零支持项；四臂共 204 项，补充对照共 102 个“格式×类别×角色”单元。少数仅拟合、没有合法留出证据的单元明确标注。这里的“格式”是当前解析器路由，不宣称已穷尽现实日志语法。
- 2,056,871 条消息都有原文摘要、行引用和完整字符区间账本；原文件不改写，NULL/空串状态保留。359 条原始记录跨全部 17 路由重放，样本数不超过 50 的稀缺“格式×类别”单元全部纳入此原文重放。不是逐条人工审阅两百万条日志，也不是完整性等于模型语义理解。
- 新增独立 record-level src_port 输入：150,626 条在消息派生端口缺失时仍有记录级观测，7,856 条两者冲突。两路分开保留，不假设字段语义必然一致，不相互覆盖；16 位编码覆盖全部 65,536 种合法端口及缺失/冲突状态。
- 稀缺优先已真实训练：按拟合侧“格式×类别”计数，用 `max(1,min(32,1000/n))` 加权，包含稀缺正常类；不按验证错误挑样本，不使用私有答案计算权重，不改原标签。这是风险权重对照，不是制造新的独立监督。

## 2. 四臂和补充对照：收益来自哪里

共同验证 403,321 行，拟合/验证按来源符号与原正文组件隔离。A/B 拟合 949,749 行，C/D/E/F 拟合 1,653,550 行。A=旧输入旧池，B=新输入旧池，C=旧输入完整池，D=新输入完整池；E=D＋补充身份代码处理，F=E＋稀缺优先。

编码器、SGD 分类器、末轮选择规则、6 epoch 和 argmax 固定。A 是本轮同学习器控制，并不是历史 v5.1 的逐数值复刻。新输入同时包含全文残差和独立记录级端口，因此 B/D 收益不能全部归为单一解析补丁。因本机内存限制，使用固定 UTF-8 字节一/二元计数，不构建巨大字符词表；无 OOV、哈希碰撞和前缀截断，但**不保留任意长距离顺序，也不是无损语义编码**。

| 臂 | 错误数 | 准确率 | Macro-F1 | M 召回 | S 召回 | 正常误报 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

A→D 减少 2,406 错，其中 native_flow 减少 2,320 错，说明完整覆盖重要；但该训练路由只有 M 支持，不能以此证明学会同格式 B/M/S 区别。B 仍有 96 条正常误报，补足真实正常覆盖后 D 为 0。

E 比 D 多 229 错；显式身份代码处理去掉了部分有利相关性，不能因为分数下降恢复未经迁移证实的身份捷径。F 相对 E 少 386 错，但 M 多错 50、S 少错 436；F 比 D 总错更少而 M 召回下降，**两个补充候选均未通过全部内部保护条件**。

## 3. 完整任务回放：内部进步没有转化为整体通过

对 2,014,052 条 valid_input 先无标签推理，再由独立评分阶段读取已经历轮查看的 private 答案。完整回放是开发回归，不是新盲测。D/F 拟合只读取 train；F 方案及完整拟合合同在本轮首次查看完整评分前冻结，不在评分后调参。

| 模型 | 错误数 | 准确率 | Macro-F1 | M 召回 | S 召回 | 正常误报 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(whole)}

D 在内部恶意召回 98.47%，到完整开发回放只有 17.66%；这是本轮最重要的反证。最终 F 全量结果是 {f['errors']:,} 错、M 召回 {pct(f['recall'][1])}，不能用某个稀缺切片成功替代整体验收。

| 路由 | 完整回放行数 | D 错误 | F 错误 | D→F 修复 | D→F 新增错误 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(slices)}

完整回放的主要恶意来源与训练内分布严重不一致：unsupported M 6,226、VPC M 2,664、ASA M 5,098；训练内 native_flow 大量改善不能替代这些缺口。VPC 的 M 在官方 train 的当前路由内为 0；这是已观察到的监督支持缺口，不足以指控官方标注错误。删除/降权 32,596 条 native_flow 不能凭空补出 VPC-M，也不符合本轮全部保留要求。

## 4. 稀缺切片逐项结果

以下均是相同内部留出角色，D→F，不与完整回放混用：

| 切片 | 留出样本 | D 错误→F 错误 | 含义 |
|---|---:|---:|---|
| Windows S | 435 | 435→10 | 原训练目标压制稀缺监督是实质因素；并非只能加大模型 |
| authentication S | 4 | 4→0 | 真实小样本局部收益，不能据 4 条宣称普遍解决 |
| CEF M | 12 | 6→0 | 32 条拟合记录错误也从 28→0 |
| bounded_payload S | 10 | 10→5 | 拟合 25→0 错，留出仍错 5：剩余偏向迁移/支持问题 |
| unsupported M | 8 | 8→8 | 拟合 22→22 错，当前输入＋学习器＋优化组合仍未学会 |
| ASA M | 16,226 | 342→640 | 少数类保护失败，不能为其他切片收益放行 |
| ASA S | 10,304 | 239→224 | 局部小收益，远未解决 ASA 边界 |

Windows M 仅 2 条、CEF S 仅 2 条、ACL B 等部分单元没有本折合法留出；已经使用和核验拟合表现，但无法虚构独立泛化结论。所有无支持/无留出的单元见 CSV，不从报告分母中抹掉。

## 5. 是解析不足、模型能力还是官方数据不足

同一全部 train 的实际输入经验冲突下限：旧表示 2,540 错，新 D 为 26 错，修正 E/F 仍为 26；拟合侧下限为 24。原文哈希没有混标只说明这些原文字符串不相同，身份/日期本身就能区分字符串，不能推导出原始行为能可靠区分标签。

F 拟合仍错 {ca['fit_metrics']['F']['errors']:,} 条，远高于 24 的固定输入经验下限；因此不能把多数错误归为“现有信息绝对不可分”。这指向学习器、优化目标或表示泛化问题，但经验查表容量也不证明某个更强模型必然跨来源有效。D 的 1,044 个留出错误中 1,034 个是未见精确输入；“精确输入未见”不等于“行为无支持”。

- **已确认并修复部分工程问题**：旧池未用全 native_flow、未知消息静默压空、独立记录端口没进入新模型、稀缺单元不出表。
- **已确认训练目标问题且获得局部修复**：Windows/认证/CEF 稀缺监督，固定加权产生真实收益。
- **仍有学习缺口**：unsupported 22 个拟合 M 全错；ASA M/S 边界退化。不能判定只是模型小，也不能把它们宣布为标签不可识别。
- **确有当前数据支持缺口**：VPC-M 零支持、某些类别只有 1～2 个独立来源/极少记录。复制、生成伪标签、直接把 REJECT 设为 M 都不补充真实判据。
- **还不能保证信息全部用于模型**：原件完整保留，但设备判断仍在独立区间账本中，尚未完成独立预测分支；字节词袋丢长距离顺序。原始记录级其他字段仍保留溯源，不当作已经证明无用而销毁。

## 6. 逐项验收与对抗性审查

1. 原文件和所有角色/数量/模型哈希复核；全量拟合确实包含 32,596 条，每 epoch 每原事件一次，无重复降权或静默剔除。
2. 原文→区间账本→视图→实际稀疏向量逐层审查。全量账本覆盖；359 条全格式/稀缺原文重放，四主模型和两个补充模型重载复核，逐格式混淆矩阵由预测独立重算。
3. 来源/正文不跨角色；单一支持组合整体保留 fit-only。当前来源符号不保证代表真实企业，所有已查看数据仍为开发资料。
4. 稀缺加权收益按同角色计算，同时暴露 ASA M 退化；不以总错减少绕过类别保护。
5. **反证测试本身也要审查**：最初身份替换误改 ISO 月/日，第一次修正仍改变 UUID 长度。这两种探针不能用于纯身份归因，原始失败记录保留但其数字已被后续正确探针替代。保持日期后缀和 UUID 词法形状的最终 359 条测试：D 完整模型翻转 2 条，E/F/F 完整模型均 0；E/F 仍有 3 条向量微变，F 完整模型最大概率变化 {identity['models']['F_full']['max_probability_delta']:.8g}。仅证明这些实际样本上的有限一致性，不证明所有干扰不变。
6. 完整 201 万行 ID 顺序、三类输出、概率、CSV 与预测一致性通过。D 混淆矩阵独立重算，F 同样核对完整输出与逐路由成绩。**执行完整性通过不等于模型质量通过。**

## 7. 下一步只推进有明确判别结果的对照

1. **先修验证目标**：把格式×类别零支持模拟列为必测机制，而不是仅在同样格式俱全的来源折选模；按无标签目标输入的格式覆盖登记差异，配合保留来源折。模拟从训练中撤去某格式某类时，检验是否能由其他格式的共享行为识别它；不得借目标答案训练。不要把 native_flow 只有 M 的高分当跨格式恶意能力。
2. **优先突破 unsupported 的拟合缺口**：固定现有 30 个官方 M 与训练侧同载体正常反例，先比较包含顺序的字段/字符片段表示与现表示，再比较线性收敛参考与有交互能力的小模型。仅拟合侧定义样本和调参；必须同时报拟合错误、正文/来源留出、正常误报。若同输入少量可分记录都拟合不了，先修解析/优化，禁止直接扩大全量训练。
3. **保留稀缺优先，隔离它对 ASA 的代价**：固定已证明有局部收益的样本集合/信息合同，比较共享基础表示＋独立残差参数与当前全部共享的风险权重；路由只控制参数共享/审计，不直接给类别。只有 ASA M 不退化且稀缺留出收益保留，才进入完整回放。当前 32 倍上限只是本轮固定实验值，不宣称最优，不继续扫描它。
4. **显式测试设备判断分支和跨格式行为共享**：保留观测原文和来源，把设备结论单独消融，附缺失/置换/正常反例检查；不能一边删除合法观测一边宣称信息充分，也不能直接复制设备 verdict 当三分类真值。VPC-M 要通过机制对照证明迁移，失败则如实登记判据/标注不足，不编造 M 样本。
5. **停止用本轮 full 回放调参**：本轮私有答案仅历史回归。新的候选必须先在 train 内冻结规则，按原有完整任务准确率/Macro-F1/M/S/误报门槛验收；不降低门槛包装提升。旧 578 清单有 418 条进入新拟合，正式独立修复数仍不增加，不能换分母宣称修好了。

## 8. 复现与交付

所有实际结果位于 `artifacts/v75_four_arm_20260921_r2/`。`four_arm/` 是四臂，`full/` 是 D 全量模型，`corrective/` 是 E/F，`F_diagnostic/` 是 F 全量诊断；两个 `res_development.csv` 都只是开发回放，未提交比赛、未替换正式入口。

已执行脚本：`run_v75.py`、`prepare_v75_metadata.py`、`analyze_v75.py`、`v75_corrective.py`、`audit_v75_corrective.py`、`evaluate_v75.py`、`v75_f_candidate.py`、`verify_v75_release.py`、`verify_v75_typed_identity.py`。已有训练输出拒绝覆盖，不应原样再次训练。8 项输入测试通过；MCP 只发布本报告和汇总证据，不发布原始日志/数据/凭据。

本轮不需要平台操作；未安装新依赖、未采集外部数据、未对外提交。下一轮需要平台时应由顺序模型的实际资源试算决定，而不是因为本轮分数不好就直接换大模型。
'''
    report_path=ROOT/'docs/V75_FOUR_ARM_EXECUTION.md';report_path.write_text(report,encoding='utf-8')
    evidence=ROOT/'evidence/2026-09-21/v75_four_arm';evidence.mkdir(parents=True,exist_ok=True)
    tracked=[report_path,OUT/'configuration.json',OUT/'four_arm/complete.json',OUT/'analysis/summary.json',
        OUT/'analysis/all_formats_all_classes.csv',OUT/'metadata_audit.json',OUT/'corrective/contract.json',
        OUT/'corrective/complete.json',OUT/'corrective/audit.json',OUT/'corrective/all_format_class_roles.csv',
        OUT/'full/complete.json',OUT/'official_replay/scoring.json',OUT/'F_diagnostic/contract.json',
        OUT/'F_diagnostic/fit_complete.json',OUT/'F_diagnostic/scoring.json',OUT/'release_verification.json',
        OUT/'typed_identity_verification.json']
    delivery={'status':'four_arm_and_rare_followups_completed_quality_not_promoted','quality_acceptance':False,
        'all_issues_solved':False,'actual_new_fits':8,'official_training_rows':2056871,'native_flow_malicious_all_used':32596,
        'duplicates_removed':0,'official_replays':{'D':2014052,'F':2014052},'internal_four_and_followup':scores,
        'complete_development_results':{'v51':full['before'],'D':d,'F':f},
        'normalization_probe':identity,'accepted_repairs':0,'accepted_remaining':578,
        'validation_scope':'Inspected train development split plus previously inspected official private-answer full regression; no new blind/external acceptance.',
        'platform_used':False,'external_training_data_used':False,
        'next_direction':'Rare-slice fit gap, source/format-class missing-support validation, and ASA-protected parameter sharing. Do not repeat current full SGD/weight sweep.',
        'temporary_cleanup':'Automatic approval policy rejected removal of aborted artifacts/v75_four_arm_20260921 with blocked by policy; retained, r2 is authoritative.',
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in tracked}}
    delivery_path=evidence/'delivery.json';save(delivery_path,delivery)
    intro=f'''当前执行（2026-09-21）：**v7.5 已完成 8 次真实拟合，32,596 条 native_flow 恶意记录全部纳入最终重训，17 个解析格式逐类审查。** 四臂内部错误 3,450→1,044；追加稀缺对照至 887，但完整开发回放 F 错 {f['errors']:,} 条、M 召回 {pct(f['recall'][1])}，质量未通过，无新模型晋升。Windows/认证/CEF 稀缺切片有局部收益，unsupported 拟合缺口和 ASA 恶意退化仍存在。完整报告与下一步以 [V75_FOUR_ARM_EXECUTION.md](docs/V75_FOUR_ARM_EXECUTION.md) 为准。以下内容为历史阶段，不覆盖本条。\n\n'''
    for rel,prefix in [('README.md',''),('docs/TRAINING_PLAN.md','docs/'),('training/README.md','training/'),('evidence/README.md','evidence/')]:
        path=ROOT/rel;old=path.read_text(encoding='utf-8');block=intro
        if prefix=='docs/':block=block.replace('(docs/V75_FOUR_ARM_EXECUTION.md)','(V75_FOUR_ARM_EXECUTION.md)')
        elif prefix:block=block.replace('(docs/V75_FOUR_ARM_EXECUTION.md)','(../docs/V75_FOUR_ARM_EXECUTION.md)')
        assert '当前执行（2026-09-21）：**v7.5' not in old
        first,rest=old.split('\n',1);path.write_text(first+'\n\n'+block+rest.lstrip('\n'),encoding='utf-8')
    catalog=read(ROOT/'mcp_readonly/catalog.json')
    for entry in catalog['documents']:
        if entry['path'] in ['README.md','docs/TRAINING_PLAN.md','training/README.md','evidence/README.md']:
            entry['sha256']=sha(ROOT/entry['path'])
        if entry['category']=='current_review':entry['category']='historical_review'
        if entry['category']=='current_evidence':entry['category']='historical_evidence'
    entries=[{'id':'v75-direction-review','title':'v7.5 四臂与稀缺优先真实结果、反证和下一步','path':report_path.relative_to(ROOT).as_posix(),
        'category':'current_review','summary':'8次拟合，全量记录覆盖；稀缺切片局部收益与完整任务退化同时报告，质量未通过。','keywords':['当前','最新','方向','下一步','训练','稀缺','完整性','v7.5'],'sha256':sha(report_path)},
        {'id':'v75-delivery','title':'v7.5 真实训练和完整回放交付证据','path':delivery_path.relative_to(ROOT).as_posix(),
        'category':'current_evidence','summary':'32,596条全部使用，17路由，四臂和两补充，两次201万行回放，未晋升。','keywords':['当前','最新','结果','证据','delivery','v7.5'],'sha256':sha(delivery_path)}]
    catalog['documents']=entries+catalog['documents'];project=catalog['project']
    project.update({'as_of':'2026-09-21','current_summary':f'v7.5新增8次拟合，32,596条native_flow M全部使用，17格式逐类审查。完整F开发回放错{f["errors"]:,}条、M召回{pct(f["recall"][1])}；质量未通过，无新模型晋升。',
        'authoritative_delivery_id':'v75-delivery','authoritative_direction_id':'v75-direction-review',
        'current_direction':['先解决unsupported稀缺样本拟合缺口，分开验证输入表示、优化与学习器。','增加格式×类别缺支持迁移对照，内部得分不能替代完整任务。','保留稀缺收益并保护ASA恶意召回，设备判断分支独立消融，全部官方样本仍保留。'],
        'known_limits':['本轮完整回放是已查看答案的开发回归，不是新盲测。','稀缺单元独立来源不足，VPC-M训练零支持，不能伪造监督。','输入原件完整不等于字节词袋无损理解；设备判断分支仍未训练。','补充加权未通过内部M召回条件；未更新正式模型/提交入口；旧578清单不能换角色后宣称独立修复。']})
    save(ROOT/'mcp_readonly/catalog.json',catalog)
    print(json.dumps({'report':str(report_path),'delivery':str(delivery_path),'new_fits':8,'quality_acceptance':False},ensure_ascii=False))


if __name__=='__main__':main()
