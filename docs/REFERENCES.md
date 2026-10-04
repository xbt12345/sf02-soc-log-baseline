# 保留的参考依据

核查日期：2026-09-11。以下资料只承担明确的论证作用，不按论文数量或自评分决定方案。

| 一手来源 | 对本项目有用的内容 | 适用边界 |
|---|---|---|
| `official/` 内附件 2 与赛事通知原件 | 三分类任务、逐事件 CSV、解释与展示要求 | AUC/F1 汇总口径、预算和平台算力仍需官方确认；不能自行补成官方规则 |
| [scikit-learn：常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html) | 切分之后才拟合变换；预处理也会泄漏 | 仅做 Pipeline 不能修复数据本身的来源与标签混淆 |
| [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) | 分组互斥，同时尽可能保持类别比例 | 不保证每折都有每一类；实体组合键也不保证单个实体互斥 |
| [阈值选择](https://scikit-learn.org/1.5/modules/classification_threshold.html) | 分开概率估计、业务决策与独立阈值校准 | 页面示例主要是二分类；三分类需另外定义完整决策规则 |
| [AIT-LDS v2.1](https://zenodo.org/records/19483937) | 多个模拟环境、逐行攻击步骤标签、可下载不含 PCAP 的日志包 | 模拟环境且标签不等价于官方三分类；不是现成同分布训练集 |
| [USENIX Security 2024：True Attacks, Attack Attempts, or Benign Triggers?](https://www.usenix.org/conference/usenixsecurity24/presentation/yang-limin) | 安全告警、攻击尝试与实际攻击需要区分 | 不能据此擅自把所有告警映射为本题 malicious |
| [USENIX Security 2025：Sometimes Simpler is Better](https://www.usenix.org/conference/usenixsecurity25/presentation/bilot) | 对复杂系统做统一验证、检查数据与实验问题的价值 | 研究对象是溯源入侵检测；不证明 CatBoost 在本题必胜 |
| [USENIX Security 2025：AutoLabel](https://www.usenix.org/conference/usenixsecurity25/presentation/peng-yihao) | 新训练标签需有可追溯的行为与真值依据 | 不是让语言模型凭日志措辞自由生成真值标签 |

历史核验稿保留在 [reference/历史文献复核_2026-06-29.md](reference/历史文献复核_2026-06-29.md)，用于回查当时文献来源；其中路径、模型效果和路线判断以当前实际复核为准。

七份旧方案中仍有价值的内容——任务契约、简单模型基线、泄漏识别、类别失衡、文本行为证据、解释性和外部标签对齐——已汇入当前方案。相互重复的路线、未经验证的先进性判断、占位文档和错误的旧运行指引不再保留为并列入口。
