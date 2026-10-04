"""Render the verified executed 2x2 trial, without reselecting failed models."""
from run_v75 import ROOT,read,sha

DEST=ROOT/'artifacts/v87_solver_supervision_r2_20260927'
DOC=ROOT/'docs/V87_SOLVER_SUPERVISION_TRAINING_RESULTS.md'
ARMS=['A0','A1','A2','A3']


def main():
    v=read(DEST/'verification.json');ledger=read(DEST/'problem_ledger.json');cont=read(DEST/'continuation.json');gates=read(DEST/'completion_gates.json')
    assert v['status']=='passed' and v['actual_classifier_fits']==4
    assert ledger['verification_sha256']==sha(DEST/'verification.json')
    folder=DEST/'fold1';reg=read(DEST/'registration.json');expo=read(folder/'exposure.json');pair=read(folder/'pair_audit.json')
    fit={a:read(folder/(a+'_fit.json')) for a in ARMS};traces={a:read(folder/(a+'_selection_trace.json')) for a in ARMS}
    diag=read(folder/'diagnosis.json');final={a:next(d for d in diag if d['name']==fit[a]['saved_states'][-1]) for a in ARMS}
    notes={a['arm']:a for a in ledger['arms']}
    first=read(ROOT/'artifacts/v87_solver_supervision_20260927/failure_receipt.json')
    rotated=(DEST/'rotation/verification.json').exists()
    rv=read(DEST/'rotation/verification.json') if rotated else None
    rc=read(DEST/'rotation/continuation.json') if rotated else None
    if rotated:assert rv['status']=='passed'
    additional=rv['actual_classifier_fits'] if rotated else 0
    total=first['actual_classifier_fits_started']+v['actual_classifier_fits']+additional
    completed=first['completed_classifier_fits']+v['actual_classifier_fits']+additional
    lines=['# v8.7：实际步长投影与跨组件监督四臂训练结果','',
        f'2026-09-27。**本轮实际启动{total}次分类器拟合：{completed}次完成、1次中止；0次校准拟合。包括首次四臂的3个完成和1个中止、统一数值修复后的4臂，以及条件轮换新增{additional}次。主折复用v8.5教师，轮换新教师{rv["new_teacher_fits"] if rotated else 0}个。**','',
        ('预登记主选结果：没有合格候选，停止扩折、支持撤除重拟合和最终全量训练。主模型质量未通过，无新模型晋升。' if cont['primary_selected'] is None else f'主选冻结为{cont["primary_selected"]["name"]}；固定轮换通过={gates["rotation_passed"]}，锁定来源回归通过={gates["locked_regression_passed"]}，最终全量拟合授权={gates["final_full_fit_authorized"]}。未通过的候选不晋升，后续追加全量训练停止。'),
        '结论：训练保护与继续更新机制已执行，但当前跨组件配对配置未获得稳定来源迁移收益，停止采用本配置。不能将隐藏空间分离、损失下降或局部净修复作为新模型有效的证据。',
        '最近完整开发回放仍为v7.9的5,947错；本轮固定状态的全输入诊断不替代完整任务工程回放。这些数据已反复观察，属于开发回归，不能称新盲测。','',
        '## 0. 本轮遇到的实现问题及恢复','',
        '首次组合臂A3在第49次更新后被完整保护复核断言拦截，未保存失败提议、未输出合格训练结论。先前模型与日志保留；未将中止拟合隐去。独立探针在已保存A3第40次状态上发现：同权重、同输入重复前向的类别间隔最大变化4.768e-7，大于原1e-7检查容差。失败49次状态不存在，因此该探针支持数值风险，不证明故障唯一成因。','',
        '统一修复后重新登记并重跑四臂：搜索时额外保留min(1e-5,旧正间隔/4)数值余量，正式epsilon、训练标签、主/辅助目标、配对及质量标准不变；接受后再独立执行正式门槛检查，不合格即恢复。原登记源码与失败日志没有覆盖，四臂同样使用安全余量，避免不公平比较。','',
        '## 1. 按方案完成的改动','',
        f'- 保留R0、66287→256→64→3容量、初始权重、原频次OVR分类主损失。当前训练损失仍为{expo["selected_original_rows"]:,}条：B/M/S={expo["selected_per_class"]}，每个当前训练角色M/S全部纳入。',
        f'- 将硬保护扩展到全部训练人口中原本正确的{expo["protected_original_rows"]:,}条，按同输入聚合为{expo["protected_unique_inputs"]:,}种；保护每个真实类别与另外两类的间隔，逐条检查真实最终判定。未入样正常只进入保护，不能称其已经进入主损失。',
        f'- 仅从当前训练角色原件构造{pair["reviewed_pairs"]:,}组三元配对；每组检查ASA拒绝语法、动作/协议/方向、原始端口与解析一致、M/S真标签、不同组件、不同输入、完整训练输入无混标。M与S都作为锚点。',
        '- 优先同目的端口细行为配对，缺少合格正负例时才用兼容粗行为的真实记录；端口没有改写或删除。ACL策略和事件意图仍未知，行为兼容不是同事件/同意图保证。未配对及混标原记录全部保留在主分类损失。',
        f'- 配对温度{reg["pair_temperature"]}、辅助系数{reg["pair_weight"]}、Adam学习率{reg["lr"]}和最大60次更新在训练前固定，不根据H搜索。',
        '- 活跃约束投影作用于Adam之后的实际参数位移；小规模对偶QP记录原始/对偶残差和KKT检查，完整非线性前向验证再接受。因子化雅可比避免建立“样本×全部参数”巨大矩阵；已与独立自动求导及真实GPU核验。',
        '- 每臂连续3次没有有效更新就停止；达到求解可靠性/资源上限也停止，不能删保护样本、丢约束或放宽标准继续。记录模型、优化器和CPU/CUDA随机状态。','',
        '## 2. 原记录上的修复与退化','',
        '以下是预登记最终或停机状态，修复/退化为旧错新对/旧对新错。它们不一定合格，不能将净收益替代逐条保护。','',
        '| 臂 | 更新方向 / 配对监督 | 尝试 / 接受 | 训练错误（基线315） | 训练修复 / 退化 | 内层错误（基线194） | 内层修复 / 退化 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        s=traces[a][-1];f=s['selected_fit'];i=s['inner'];cfg=reg['arms'][a]
        lines.append(f'| {a} | {cfg["solver"]} / {"有" if cfg["contrastive"] else "无"} | {fit[a]["attempts"]} / {fit[a]["accepted_updates"]} | {f["errors"]} | {f["positive_flips"]} / {f["negative_flips"]} | {i["errors"]} | {i["positive_flips"]} / {i["negative_flips"]} |')
    lines.extend(['',f'共保存并独立重放{v["all_saved_states_replayed"]}个主折状态，含4个零修正初态。各状态按训练全P、真实分类修复、内层零负翻转和各类召回/精确率/F1/正常误报检验。主选在C/H诊断之前冻结。','',
        '## 3. 外层诊断与恶意召回','',
        'C/H只诊断固定状态，不用于选择配对、系数、检查点或保护记录。','',
        '| 臂 | 全训练错误 | C错误：修复 / 退化 | H错误：修复 / 退化 |',
        '|---|---:|---:|---:|'])
    for a in ARMS:
        d={v['role']:v for v in final[a]['roles']}
        lines.append(f'| {a} | {d["fit_full"]["errors"]} | {d["C"]["errors"]}：{d["C"]["positive_flips"]} / {d["C"]["negative_flips"]} | {d["H"]["errors"]}：{d["H"]["positive_flips"]} / {d["H"]["negative_flips"]} |')
    lines.extend(['','基线完整训练/C/H错误分别为315/3425/290。以下M/S正确数均以真实支持数为分母；完整精确率、F1、混淆矩阵保存在诊断JSON和全部格式逐类CSV中。','',
        '| 臂 / 角色 | 恶意正确 / 总数 | 可疑正确 / 总数 | 新增错误B / M / S |','|---|---:|---:|---|'])
    for a in ARMS:
        for d in final[a]['roles']:
            if d['role']=='fit_full':continue
            cm=d['cm'];sup=d['support']
            lines.append(f'| {a} / {d["role"]} | {cm[1][1]:,} / {sup[1]:,} | {cm[2][2]:,} / {sup[2]:,} | {d["negative_flips_by_class"]} |')
    lines.extend(['','## 4. 两种机制分别有没有作用','',
        '| 臂 | 第32次后接受更新 | QP次数 | 最多活跃约束 | 最小接受提议比例 | 主分类损失：首轮→末轮之前 | 配对损失：首轮→末轮之前 |','|---|---:|---:|---:|---:|---|---|'])
    for a in ARMS:
        n=notes[a];loss=n['first_and_last_losses']
        lines.append(f'| {a} | {n["accepted_after_epoch32"]} | {n["QP_solves"]} | {n["max_active_constraints"]} | {n["min_nonzero_proposal_fraction"]:.8g} | {loss["main_before"][0]:.8g}→{loss["main_before"][1]:.8g} | {loss["contrastive_before"][0]:.8g}→{loss["contrastive_before"][1]:.8g} |')
    lines.extend(['','新增配对监督是否触及旧错误：','',
        '| 臂 | 类别 | 是否获得配对 | 原错误数 | 其中修复 | 仍错 |',
        '|---|---|---|---:|---:|---:|'])
    for a in ARMS:
        for c in notes[a]['cohorts']:
            if c['old_wrong'] and c['rows']:
                lines.append(f'| {a} | {"M" if c["class"]==1 else "S"} | {"是" if c["pair_eligible"] else "否"} | {c["rows"]} | {c["repairs"]} | {c["rows"]-c["repairs"]} |')
    lines.extend(['','四条ASA训练恶意错误没有合格配对；309条ASA可疑错误中188条有配对、121条没有。不能把总配对19,603组当成同样数量的困难错误监督；大量配对锚点原来已经判对。'])
    lines.extend(['','投影方向是否执行、损失是否降低与实际分类收益分别判断。配对后隐藏空间的同/异类余弦区分统计也保留，但不能用表示变好代替留出记录判对。','',
        '输出分数的独立CPU分解：','',
        '| 臂 | ASA原错类别 | 有配对 | 原错数 | 向真实类别补偿的条数 | 真正修复 | 旧错误分数差中位数 | 修正补偿中位数 |',
        '|---|---|---|---:|---:|---:|---:|---:|'])
    for d in ledger['output_margin_decomposition']:
        lines.append(f'| {d["arm"]} | {"M" if d["class"]==1 else "S"} | {"是" if d["pair_eligible"] else "否"} | {d["rows"]} | {d["positive_true_vs_old_winner_gain_rows"]} | {d["total_correct_rows"]} | {d["median_old_wrong_gap"]:.4f} | {d["median_true_vs_old_winner_gain"]:.4f} |')
    lines.extend(['','这里分解真实总输出=固定教师分数+残差修正。正补偿只表示方向朝真实类别移动，不等于残差独立分类准确率。A3有配对S的188条错误中156条朝正确方向移动，但只修复2条：错误教师分数差中位数3.6143，修正补偿中位数0.1267。四条无配对M错误的补偿全部为负，中位数-0.0564。当前读出修正的力度与方向都存在缺口；不能据此排除表示、共享参数约束或真实语义支持不足。','',
        '此前两个M退化输入及训练阻挡组的M/S间隔：','',
        '| 臂 | 输入组 | 旧M/S间隔 | 最终M/S间隔 | 最终类索引（B=0,M=1,S=2） |','|---|---:|---:|---:|'])
    for c in ledger['targeted_margin_cases']:
        lines.append(f'| {c["arm"]} | {c["fid"]} | {c["old_M_S_margin"]:.6f} | {c["final_M_S_margin"]:.6f} | {c["final_prediction"]} |')
    lines.extend(['','中途修复是否保持：','',
        '| 臂 | 轨迹中最多修复 | 对应最早轮次 | 最终修复 | 解释边界 |','|---|---:|---:|---:|---|'])
    for a in ARMS:
        t=notes[a]['trajectory']
        lines.append(f'| {a} | {t["max_observed_train_repairs"]} | {t["earlier_best_epoch"]} | {t["final_train_repairs"]} | 中途统计不是新增选模检查点 |')
    lines.extend(['','原教师正确的P是静态保护集合，后来被修复的E并不会自动加入P。若轨迹出现中途修复又丢失，说明当前保护无法保证新增收益逐轮保持；不能把中途峰值当最终成绩，也不能事后补选未登记检查点。平均损失优化和实际判对仍可能错位，是否增加训练内新增修复保护需后续受控对照。'])
    if cont['primary_selected'] is not None:
        selected=cont['primary_selected'];chosen=next(d for d in diag if d['name']==selected['name'])
        lines.extend(['','## 4.1 冻结候选及固定轮换','',
            f'主选冻结为{selected["name"]}，内层错误{selected["inner_errors"]}，其他保存状态不再替换它。','',
            '| 主折固定候选角色 | 错误 | 修复 / 退化 | 新增B/M/S错误 |','|---|---:|---:|---|'])
        for d in chosen['roles']:
            lines.append(f'| {d["role"]} | {d["errors"]} | {d["positive_flips"]} / {d["negative_flips"]} | {d["negative_flips_by_class"]} |')
        if rotated:
            lines.extend(['',f'保持主选第{selected["epoch"]}次状态、温度/系数/结构和全部门槛，在折3/4重新生成训练角色、教师和配对，同时训练A0控制与冻结候选臂。不重新选轮次。','',
                '| 轮换折 / 臂 | 训练修复 / 退化 | 内层错误 | 内层修复 / 退化 | 固定状态通过 |','|---|---:|---:|---:|---|'])
            for h in [3,4]:
                for a in sorted(set(['A0',selected['arm']])):
                    st=read(DEST/'rotation'/f'fold{h}'/(a+'_selection_trace.json'))[-1]
                    lines.append(f'| {h} / {a} | {st["selected_fit"]["positive_flips"]} / {st["selected_fit"]["negative_flips"]} | {st["inner"]["errors"]} | {st["inner"]["positive_flips"]} / {st["inner"]["negative_flips"]} | {"是" if st["eligible"] else "否"} |')
            lines.extend(['','轮换固定状态的锁定来源逐类结果：','',
                '| 折 / 臂 / 来源 | 恶意正确 / 总数 | 可疑正确 / 总数 | 修复 / 退化 | 新增B/M/S错误 |',
                '|---|---:|---:|---:|---|'])
            for h in [3,4]:
                for item in read(DEST/'rotation'/f'fold{h}'/'diagnosis.json'):
                    for d in item['roles']:
                        if d['role'] not in ['C','H']:continue
                        lines.append(f'| {h} / {item["name"]} / {d["role"]} | {d["cm"][1][1]} / {d["support"][1]} | {d["cm"][2][2]} / {d["support"][2]} | {d["positive_flips"]} / {d["negative_flips"]} | {d["negative_flips_by_class"]} |')
            lines.extend(['','冻结配对臂在两折内部留出都修复0条；第四折控制臂却修复4条。本次候选未复现新增配对收益。第三折配对臂的H还新增2条恶意错误，不能称训练旧正确保护已经解决未知来源退化。'])
            lines.extend(['',f'轮换保存并独立重放{rv["all_saved_states_replayed"]}个状态，另有{rv["all_input_final_models_replayed"]}个最终全输入分支重放；轮换整体通过：{rc["rotation_passed"]}。局部通过不能自动晋升；轮换或锁定来源门槛失败就停止支持撤除追加拟合和最终全量重训。'])
    lines.extend(['','这些旧输入只用于诊断，没有加入训练保护、蒸馏或配对答案。它们已被观察过，不作为新盲测。','',
        '## 5. 覆盖与尚未解决的问题','',
        f'训练基线315条错误中，{ledger["same_input_feasibility"]["same_encoded_input_has_protected_correct_rows"]}条与受保护正确记录拥有相同固定编码输入，当前单输入判定与零退化要求使其无法同时修好。另{ledger["same_input_feasibility"]["no_same_input_protected_correct_rows"]}条（B/M/S={ledger["same_input_feasibility"]["unblocked_per_class"]}、{ledger["same_input_feasibility"]["unblocked_unique_inputs"]}种输入）没有这种直接矛盾。因此不能把几乎不修错全部归因于输入冲突；也不能据此保证共享参数存在可行修复方向。固定编码一致不等于原始信息完全一致。','',
        '配对覆盖细表：','',
        '| ASA训练锚点类 | 细级（同目的端口） | 粗级 |','|---|---:|---:|'])
    for cls in [1,2]:
        grade={r['grade']:r['rows'] for r in pair['by_class_grade'] if r['anchor_class']==cls}
        lines.append(f'| {cls} | {grade.get("fine",0)} | {grade.get("coarse",0)} |')
    lines.extend(['','没有配对的原因包括语法未处于本次已审核模板、解析事实缺失、完整编码混标以及缺少兼容的跨组件正负例。不能称所有格式、所有ASA困难样本的额外监督已经补齐。原件全保留，误差按配对/未配对分别计数。','',
        '内部留出另按训练内细行为同类支持为多组件、单组件、零支持和事实缺失分桶；每臂都有真实正确/修复/退化结果。它是当前自然支持条件的验证；支持撤除重拟合需另过轮换门槛，不把自然支持分桶称作已完成因果控制实验。','',
        'VPC在训练中缺M、未见格式缺监督、长序列/上下文的语义充分性仍未解决。所有格式逐类记录，不用大正常类或若干有利切片掩盖。','',
        '## 6. 第一性原理与多轮对抗性审查','',
        '1. **是否仍只保护了小样本？** 全部922009条训练正确进入P，独立原记录重放所有保存状态；未入样正常进入保护但不是主分类损失。未知输入仍不保证零退化。',
        '2. **是否投影梯度后又被Adam改变？** 先取得Adam真实位移再投影；每个活跃边界由真实类别与具体竞争类组成，最后完整P非线性核验。局部QP不能代替结果验收。',
        '3. **是否靠删混标/造端口反事实？** 混标只禁止参与矛盾配对，仍在原频次主损失；所有三元组来自原始真实记录，独立原标签与端口核对。',
        '4. **是否用H选方法？** 四臂参数提前登记，主选先冻结；失败状态C/H仅用于解释，不替代失败门槛。全部数据是开发记录。',
        '5. **是否把损失和机制改进当成绩？** 每类原记录正确数、修复和退化决定继续；无合格候选就停止扩折，不输出最终提交。','',
        '## 7. 下一步必须改变的地方（尚未验证收益）','',
        '1. 保留本轮全训练P保护与真实逐类验收，改变学习投入：优先诊断没有同输入直接矛盾的293条训练错误，直接监督真实类别对竞争类的判定间隔，并设置冻结隐藏表示重新训练读出头的受控对照，区分已学表示的读出不足与表示本身不足；保护正确判定，不能把固定教师的错误高分当必须保留的答案。不能只让辅助隐藏空间更分离、却让输出边界几乎不动；困难M目前无配对，不能继续只靠同一ASA拒绝模板的辅助损失。配对或梯度重分配仍需独立对照，不能宣称已经有效。',
        '2. 对训练内新增修复做逐条账和持续保持控制：静态P仅保护旧教师正确，未自动保护新修好的记录。下一对照可比较静态P与训练内新增正确保护，来源留出不得进入保护答案。该控制可能进一步压缩可行空间，需要记录实际修复/停滞；不承诺它能解决跨来源退化。',
        '3. 支持不足仍需类别支持撤除重拟合对照。本轮自然支持分桶已完成；因固定来源轮换/锁定门槛失败，按登记停止追加支持撤除和最终全量拟合。当前不能把自然缺支持统计变成数据问题或模型问题的唯一因果结论。22条直接编码矛盾另查原始信息，保留原标签，不能用删行制造改善。',
        '4. 下一轮只能以训练内生成的规则固定候选，再按来源隔离检验。C/H仍属于反复观察的开发回归，不能作为未知泛化证据；当前跨来源M/S边界未被本轮修复。','',
        '技术来源只作为实现参考，不是项目有效性证明：[OSQP官方Python接口](https://osqp.org/docs/interfaces/python.html)；[Correct-n-Contrast作者的监督对比实现](https://github.com/HazyResearch/correct-n-contrast/blob/main/contrastive_supervised_loader.py)。本轮效果由本项目原记录重放决定。','',
        '## 8. 证据与执行边界','',
        '- [预登记](../artifacts/v87_solver_supervision_r2_20260927/registration.json)',
        '- [保存状态与原记录独立核验](../artifacts/v87_solver_supervision_r2_20260927/verification.json)',
        '- [问题账、学习轨迹与配对诊断](../artifacts/v87_solver_supervision_r2_20260927/problem_ledger.json)',
        '- [固定来源轮换独立核验](../artifacts/v87_solver_supervision_r2_20260927/rotation/verification.json)',
        '- [最终继续/停止门槛](../artifacts/v87_solver_supervision_r2_20260927/completion_gates.json)',
        '- [全部格式逐类及完整分支诊断](../artifacts/v87_solver_supervision_r2_20260927/fold1/diagnosis.json)','',
        f'本机RTX 4060执行，未使用平台、外部训练集、伪标签或正式提交。{v["old_bound_files_rehashed"]}个历史绑定文件复核未变；新增源文件、原数据与注册输入已绑定哈希。独立显式GELU重放全部保存状态及最终全输入分支，CPU每状态512输入最大分数差{v["CPU_max_difference"]:.3g}。未保存逐轮状态的约束只能由实时检查日志证明其检查范围，未虚称每轮权重都独立重放。'])
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(DOC))


if __name__=='__main__':main()
