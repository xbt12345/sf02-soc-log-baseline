"""Produce the factual experiment report only after sealed evaluation completes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v61_common import read,save,sha
from v61_runtime import score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/v61_source_factorial_20260914_r2'
OUT=ROOT/'artifacts/v61_models_20260914'


def main():
    inner=read(OUT/'inner_comparison.json')['arms'];outer=read(OUT/'outer/summary.json')['runs']
    view=read(DATA/'model_view_ambiguity.json');audit=read(DATA/'full_input_replay_audit.json')
    data_audit=read(DATA/'data_audit.json');gate=read(OUT/'ceiling_gate_amendment.json')
    checks={}
    for run,v in outer.items():
        p=pd.read_parquet(OUT/'outer'/(run+'.parquet'));prob=p[['p_B','p_M','p_S']].to_numpy()
        assert np.isfinite(prob).all() and (prob>=0).all()
        actual=score(p.label,prob)
        assert actual==v['evaluation'];checks[run]={'reported_counts_and_metrics_recomputed':True,
            'maximum_probability_sum_error':float(np.abs(prob.astype(float).sum(1)-1).max())}
    models={arm:read(OUT/f'{arm}_20260915/selection.json') for arm in 'ABCD'}
    rows=[]
    for arm in 'ABCD':
        i=inner[arm]['selection']['ASA'];o=outer[f'{arm}_20260915']['evaluation']['ASA']
        rows.append(f"| {arm} | {i['macro_f1_M_S']*100:.2f}% | {o['macro_f1_M_S']*100:.2f}% | {o['recall_B_M_S'][1]*100:.2f}% | {o['recall_B_M_S'][2]*100:.2f}% | {o['errors']} |")
    comparison='\n'.join(rows)
    learning=[]
    for arm in 'BD':
        curve=read(OUT/f'{arm}_20260915/learning_curve.json')
        learning.append(f"{arm} 的三轮内层 Macro-F1："+' → '.join(f"{x['selection']['ASA']['macro_f1_M_S']*100:.2f}%" for x in curve)+f"；按预先规则选第 {models[arm]['selected_epoch']} 轮。")
    intervals='\n'.join(f"- {arm} 相对 A 的外层 Macro-F1 差值 95% 主体配对区间：[{outer[arm+'_20260915']['paired_vs_A_first_seed']['delta_95_percentile_CI']['macro_f1_M_S'][0]*100:.3f}, {outer[arm+'_20260915']['paired_vs_A_first_seed']['delta_95_percentile_CI']['macro_f1_M_S'][1]*100:.3f}] 个百分点。" for arm in 'BCD')
    eligible=[k for k,v in inner.items() if v['continue_seed_replication']]
    assert not eligible,'If a candidate qualifies, complete replication and rewrite the conclusion before delivery.'
    total=sum(m['elapsed_seconds'] for m in models.values())
    text=f'''# v6.1：四组真实训练与主体隔离评价

结论：当前四组配置均未产生可靠的新主体收益。外层 ASA Macro-F1 仅 66.75%–67.14%，suspicious 召回约 30%；D 相比 A 只少错 6 条，主体配对区间跨零。B 的 98.54% 内层分数没有转化为外层能力。应停止扩训这组配置，不能晋升或替换现有正式模型。

本轮已实际完成 A/C 两个简单分类器、B/D 两个完整微调编码器；B/D 各训练 3 个 epoch。没有使用主办方平台，没有扩大成全量 SOC 模型，也没有替换历史正式模型。核心模型拟合共 4 次、固定一个种子；另有两个批量配置的环境检查和两次仅拟合侧 21 条无冲突输入的小样本学习检查，均不计入模型质量结果。

**执行范围与结果。** A 是单条最小规范化正文＋13 项事实的词/字符 TF-IDF、事实独热编码及逻辑回归；B 使用同一条事件信息，完整微调 SecureBERT 2.0 后与事实融合；C 在 A 上加入 16 项匿名共现统计；D 在 B 上加入同样统计和最多 32 个可学习的邻居事实/相等关系状态。D 的完整统计覆盖全部合格状态，邻居明细有数量上限。没有同时更改类别权重、重复权重、损失和阈值；统一原行交叉熵、三类 argmax。

下表中的 Macro-F1 是全部 ASA 行上 malicious/suspicious 两个 F1 的均值，误判为 benign 仍计错。不是全体 SOC 三类成绩，也不是准确率。

| 对照 | 内层 Macro-F1 | 外层 Macro-F1 | 外层 malicious 召回 | 外层 suspicious 召回 | 外层 ASA 错误数 |
|---|---:|---:|---:|---:|---:|
{comparison}

外层 ASA 共 19,877 条。额外只有 2 条正常 ACL、来自 1 个源主体，不能用于宣称万分位误报可控。四组的 benign 回流和正常控制详情均保留在 [完整指标](../artifacts/v61_models_20260914/outer/summary.json)。

A/C 将这 2 条正常控制全部误报为 suspicious，B/D 判对；数量过少，不能据此判断实际正常流量误报水平。四组没有把 ASA 回流为 benign，本轮的主要错误是 suspicious 被判为 malicious。

**最大的失败已具体定位。** 外层一个源主体包含 3,650 条 suspicious，占该区 suspicious 的 66.42%。B/C/D 将其全部判错，A 判错 3,630 条。这批输入是外部到 DMZ 的多种 TCP/UDP 拒绝事件；与它们完全相同的单条可见输入，在拟合区共有 6,740 条 malicious、仅 86 条 suspicious。仅增加对训练标签的拟合，会强化既有多数类关联，无法自动知道这个新主体应该改判为 suspicious。

而且这 3,650 条的匿名关系规模全部超出拟合区范围：每条周边有约 1,824 种不同符号状态，拟合区最大只有 228；不同目的端口符号为 337，拟合区最大 188。D 面对的是未覆盖的关系规模外推，不只是“已有输入没背熟”。这些规模是去重后的语料符号状态，不能称为真实流量频率。[最大失败主体审计](../artifacts/v61_models_20260914/outer/largest_failure_group_audit.json)

打乱同协议记录的上下文后，C 外层 Macro-F1 从 66.75% 降到 57.38%，D 从 67.14% 降到 64.89%。说明模型确实使用了关系，不能说关系分支没有接上；但真实关系仍未让它们优于无关系对照。打乱会改变有意义的关系，属于依赖诊断，不是标签保持的反事实样本。

**不能用最高数字直接选赢家。** 内层 A 为 91.50%，B 一度达到 98.54%，但其中 1,184 条修正集中于同一源主体，主体重采样区间跨零，而且 suspicious 召回未同步提高。后续轮次的回落已被保留：

- {learning[0]}
- {learning[1]}

外层不参与训练参数或轮次选择。按 2,000 次源主体配对重采样：

{intervals}

本轮没有候选通过内层的完整继续条件，因此没有启动另外两个种子的扩训、目的主体协议、损失改造或更大模型；也不根据外层结果回头改变该决定。该停止只针对本次固定配置，不能推出所有关系模型或预训练模型都无效。

**输入损失与学习缺口已经分开。** 全量 99,398 行的一致身份重命名，关系统计、邻居索引和关系张量均完全一致；真实身份字符串、绝对时间、产品字段没有进入分类输入。正文最长 46 个 token，无未知 token、无截断，所有 15,664 种正文均通过分词—解码逐字回放。这个回放证明的是规范化后的正文完整进入编码器，不是证明被排除的身份或日志头从来没有语义价值。

| 当前可见输入 | 拟合区冲突错误下限 | 内层冲突错误下限 | 外层冲突错误下限 |
|---|---:|---:|---:|
| B 单事件模型入口 | 168 | 30 | 940 |
| C 加统计的信息视图，尚未计算 TF-IDF 额外碰撞 | 68 | 24 | 12 |
| D 实际类别编码、邻居上限后的模型入口 | 56 | 24 | 10 |

这些是每个分区在已知答案后对同输入异标签计数得到的经验下限，不是可达到的新主体成绩，更不是理论 Bayes 错误率。上下文增加区分度也可能增加语料结构指纹；不能把“冲突少”直接解释为“具有攻击语义”。但 D 的实际入口没有把 940 条级别的单事件歧义重新引入，因此后续若仍大量出错，不能把它们全部归因于入口删除信息。[精确输入视图审计](../artifacts/v61_source_factorial_20260914_r2/model_view_ambiguity.json)

具体冲突样例是 `outside → dmz`、UDP 514 拒绝、源端口隐藏：完整相同的本轮单条输入在拟合区是 4 条 suspicious，在内层是 256 条 malicious，在外层又是 10 条 suspicious。真实身份/上下文可能不同；这不证明官方标签错误，却直接反证“仅靠该单条可见输入就有唯一、跨主体的正确标签”。[样例计数](../artifacts/v61_models_20260914/udp514_domain_conflict_example.json)

**验证设计与修正披露。** 使用此前允许开发的 99,382 条 ASA＋16 条正常 ACL，旧压力区 7,571 行保持排除。固定源主体划分：拟合 59,640、选择 19,879、评价 19,879；同源主体和原正文组不跨区。源主体相等按全局脱敏地址保守隔离，上下文另按采集器命名空间隔离。关系构造只使用各自分区无标签共现；没有实体标签、邻居投票、真实访问频率或时间排序。

上下文仅代表离线批次中的符号共现，不能称为经确认的会话、攻击链或在线检测。该官方数据已在历史工作中反复研究，本轮只对当前模型做新的主体留出，不能宣称分析者从未见过的独立新测试。外层标签曾用于类别覆盖和输入冲突审计，未用于训练、轮次或门槛选择；外层预测在四组选择记录封存后才生成。[封存记录](../artifacts/v61_models_20260914/outer_seal.json)

原 v6.0 要求 suspicious 召回再提高 5 个百分点，但本轮 A 内层已是 {gate['A_inner_S_recall']*100:.4f}%，原条件在数学上不可能满足。在 B/D 拟合前单独记录了上限修正：高召回时至少减少一半剩余漏判，本轮等价于增加 {gate['replacement_delta_S_requirement']*100:.4f} 个百分点；Macro-F1 增加 2 个百分点、malicious 召回下降不超过 1 个百分点不变。旧条件的判断仍保留，不追认原方案通过。[修正记录](../artifacts/v61_models_20260914/ceiling_gate_amendment.json)

**连续反证与实际处理。** 以下是同一执行过程中的多轮检查，不是多位独立评审或多次盲测。

| 质疑 | 查证与处理 | 剩余边界 |
|---|---|---|
| 身份/时间会不会继续暗中入模 | 修复规则关键字中的组织编号；全量身份重命名后关系入口不变；时间不入模 | 关系结构和真实端口仍可能是环境代理 |
| 合并重复是否又删掉有用关系 | 训练前改为合并相同符号事件，保留同事实访问不同目标；原行损失权重未降低 | 共现不是已证明的真实访问次数 |
| 只是运行了模型外壳 | 真实前向、反向、编码器参数更新、保存重载、21 条拟合小样本可学会；两组均完成三轮 | 环境检查不等于识别质量 |
| 高分是否由一个大簇撑起 | 拆到主体发现 B 主要修正一个 1,184 行簇；采用配对主体区间 | 一个种子不足以证明稳定性，未过继续条件不扩训 |
| 差成绩是否都由入口丢失导致 | 对 B/C/D 入口分别计算冲突，下限差别已量化 | 唯一输入不代表其标签规则能从训练数据学出 |
| 更长训练是否一定更好 | 保存三轮曲线并只按内层选择；B/D 的更多训练并非持续改善 | 尚未证明训练目标对真实 M/S 语义充分 |
| 规范化重放是否可复现 | 统一冻结的字符串类型后全量关系回放一致；复用固定划分文件 | 跨 sklearn 版本重新生成 SGKF 会改变成员，禁止靠相同 seed 假定一致 |

**执行资源与保留结果。** 本机 RTX 4060 Laptop GPU，独立 `.venv-v61`，PyTorch 2.7.1+cu128、Transformers 4.52.4；原 `.venv` 未改。四组拟合阶段记录的用时合计约 {total/60:.1f} 分钟，不含下载、数据审计和最终外层评价。SecureBERT revision 固定为 `7f7c16d1b2316c5046759667ed97f527aa1b7709`，模型权重摘要与上游 LFS 一致。模型下载属于外部预训练先验，监督样本仍只有官方数据；没有把 SOC 日志发送给外部模型服务。

训练权重、预处理和逐行预测保存在 [模型与结果目录](../artifacts/v61_models_20260914)，本轮输入在 [冻结输入目录](../artifacts/v61_source_factorial_20260914_r2)。[环境凭据](../artifacts/v61_models_20260914/environment_receipt.json) 和 [旧证据复核](../artifacts/v61_models_20260914/prior_evidence_recheck.json) 记录了资源与来源：v5.7 的 128 个绑定文件和 v6.0 的 13 个绑定文件均未变化。

当前最稳妥的决定是：保留这四组作为能力对照和失败证据，不替换全量分类器，不以官方分布内高分宣称可迁移。若另开阶段，先依据本轮逐主体错误和拟合/外推差距，明确可验证的关系学习假设与 M/S 判定依据；没有这些证据时，不再默认通过加 epoch、换更大模型或调门槛解决问题。

需要补的首先是判定依据和训练覆盖：为何相同拒绝事件在不同主体属于不同类别，以及是否存在可验证的资产、会话、行为结果或同类大规模关系反例。当前还不能证明这些缺口能仅靠已有官方标签补齐；也不能证明所有其他算法都失败。下一阶段如仍限于官方数据，应先固定多个主体/行为覆盖分区和拟合范围外的风险标记，检验关系规律能否由已有样本支持；不得删除这个失败主体、重挑划分或直接把分布外记录改标 suspicious 来获得好看分数。
'''
    report=ROOT/'docs/V61_EXECUTION_REVIEW.md';report.write_text(text,encoding='utf-8')
    ev=ROOT/'evidence/2026-09-15/v61_execution';ev.mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'training').glob('*v61*.py'))
    files=[report]+sources+[p for p in DATA.iterdir() if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.pt','.joblib']]
    # Model bytes are bound directly as well; hashing is integrity, never quality acceptance.
    files += [OUT/f'{arm}_20260915'/('model.joblib' if arm in 'AC' else 'model.pt') for arm in 'ABCD']
    save(ev/'delivery.json',{'status':'four_arm_experiment_completed_no_model_promoted','core_model_fits':4,
        'source_seed_count':1,'neural_epochs_per_arm':3,'platform_used':False,'quality_acceptance':False,
        'metric_recomputation_checks':checks,'bindings':{p.relative_to(ROOT).as_posix():sha(p) for p in files}})
    print({'report':str(report),'evidence':str(ev/'delivery.json'),'model_promoted':False},flush=True)


if __name__=='__main__':main()
