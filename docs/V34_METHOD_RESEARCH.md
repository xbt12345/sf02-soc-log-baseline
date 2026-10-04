# 面向 v3.3.1 实测失败的方法调研

调研日期：2026-09-12。结论：**当前最有依据的方向是共享事件事实表示、明确的小来源监督条件，以及分开验证零样本迁移和目标正常适配。** 没有公开实测可以证明某个现成模型在本项目上“市场最优”；下列论文和项目的任务、数据及评估单位不同，不能横向抄一个 F1 排名。除本项目返回模型外，这些项目本次均未安装、运行或在 SOC 数据上实测；未下载外部训练集或预训练权重。

## 候选与取舍

| 方法/项目 | 已阅读的一手材料和可借鉴部分 | 对本项目的决定 |
|---|---|---|
| **OCSF** | [官方 schema](https://github.com/ocsf/ocsf-schema)定义跨产品事件类别与属性；[认证类定义](https://github.com/ocsf/ocsf-schema/blob/main/events/iam/authentication.json)区分认证动作及相关状态/方法 | **最高优先级借鉴。** 做一个小型事实结构，把 AD、Duo 等不同表达对齐；OCSF 是表示标准，不是训练好就能检测攻击的算法。不建设完整 SIEM，也不把未知字段补成标准值 |
| **LogLead** | [官方项目](https://github.com/EvoTestOps/LogLead)分离加载、表示增强和检测，支持词/字符、模板、BERT、数值等组合；[作者论文](https://arxiv.org/abs/2311.11809)讨论统一比较与效率 | **借鉴比较方式与缓存设计。** 同协议逐项比较表示，保持简单分类器对照。官网演示的 HDFS 结果不能证明 SOC 迁移；[PyPI](https://pypi.org/project/LogLead/)声明 Python ≥3.9，不能原样塞入现有 3.8 环境 |
| **MaidLog，ICDE 2026** | [作者论文](https://dominatorx.github.io/files/26ICDE-p.pdf)将 LLM 放在训练辅助端，使用轻量、面向域泛化的检测器；[作者代码](https://github.com/hehepig4/maidlog)及[向量代码](https://github.com/hehepig4/maidlog/blob/main/data_preprocess/fasttext.py)使用预训练 fastText 表示 | **重要前沿参考，不整套替换。** 借鉴训练/推理分离和语义表示；其伪标签流程不适合直接覆盖本题已有标签。公开实验主要是系统日志异常，不是本题 benign/malicious/suspicious 的独立事实验证 |
| **fastText / Sentence Transformers** | [fastText 官方教程](https://fasttext.cc/docs/en/unsupervised-tutorial.html)提供子词与未登录词表示；[Sentence Transformers 官方项目](https://github.com/huggingface/sentence-transformers)提供预训练语义编码接口 | **第二阶段单因素候选。** 在已抽出的事实短文本上测试一个冻结编码器，避免让包装、无关长段和未知值吞没语义。不会假设相近向量等于相同威胁；预训练语料属于外部先验，需在实验说明中明确，不能声称模型只接触过官方语料 |
| **Drain3** | [官方项目](https://github.com/logpai/Drain3)支持流式模板提取、参数屏蔽、参数抽取和状态持久化 | **用于模板发现/覆盖诊断。** 不用模板编号直接当风险，不把端口/状态码等一律变成数字占位符；若模板是学习出来的，只能在允许的训练职责学习 |
| **LogBERT / LogAI** | [LogBERT 作者代码](https://github.com/HelenGuohx/logbert)提供 HDFS/BGL/Thunderbird 等日志异常实现；[LogAI 官方项目](https://github.com/salesforce/logai)提供统一日志分析与检测接口 | **保留对照参考。** 这些方案可解决系统日志异常问题，但高 HDFS 成绩不是跨厂商 SOC 三分类证据。当前先解决单事件事实与支持缺口，不直接套序列模型 |
| **PIDSMaker / ORTHRUS** | [USENIX 2025 统一审查](https://www.usenix.org/conference/usenixsecurity25/presentation/bilot)、[PIDSMaker 作者框架](https://github.com/ubc-provenance/PIDSMaker)与 [ORTHRUS 论文](https://www.usenix.org/conference/usenixsecurity25/presentation/jiang-baoxiang)/[代码](https://github.com/ubc-provenance/orthrus)强调溯源、归因与实际报警负担 | **借鉴评价原则，暂不采用图模型。** 本数据尚未证明有完整进程/文件交互、可靠事件顺序和实体关联，不能把几列 IP 拼成完整溯源图。以后链路信息真实可用再登记独立图候选 |
| **Cleanlab** | [官方交叉验证教程](https://docs.cleanlab.ai/stable/tutorials/pred_probs_cross_val.html)要求用于标签问题发现的概率来自样本外预测 | **只辅助排复核顺序。** 即使组外预测也可能系统性地不懂一个来源；Duo/AD 的失败正说明不能直接把模型不同意的原标签删除或改成正常 |
| **Sigma** | [官方规则库](https://github.com/SigmaHQ/sigma)用结构化条件表达安全检测逻辑 | **借鉴可审计条件、例外和关联要求。** 不把规则命中直接转成官方标签或“真实攻击”，也不下载一堆规则代替本轮证据审查 |

## 为什么本轮优先事实表示

Windows AD 实例保留了事件码 `4625`、`An account failed to log on`、`outcome=failure`、状态/子状态码等可观察内容；[微软事件定义](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4625)能核实它是登录失败。Duo 同样有认证结果和失败原因，可共享“认证/动作/结果/原因”表示。这种对齐没有把登录失败等同攻击，也不能从事件码推断所有失败都是可疑。

WAF 既有正常访问记录，又有设备报告的拒绝/协议违规。[Barracuda 原生日志格式说明](https://documentation.campus.barracuda.com/wiki/spaces/BWAFv76/pages/2721209/Exporting%2BLog%2BFormats)提供格式依据。CEF 容器中的等号、竖线应与请求载荷分开；设备攻击类型或风险意见单独标为上游判断。不能因为设备写了 `AttackGroup` 就当作独立的攻击事实，也不能把 HTTP 200 直接写成正常标签。

这是对本轮固定模型实测的推断：**共享语义缺失和结构符号混用比模型容量更值得先修**。本轮运算符消融能改变 WAF 误判，却同时丢掉可疑，说明必须按角色处理，而非全局删词。是否最终提高真实判断能力，仍要由匹配验证回答。

## 前沿方法的对抗性筛选

- **对 MaidLog 的质疑：**如果教师 LLM 也依据 denied、格式或缺失值造标签，只是换了一个偏差来源。可借鉴它的轻量推理和表示方式；暂不采纳自动伪标签替换监督真值。作者材料也讨论了 LLM 的偏差和提示敏感性。当前代码首页指向的 `code/README.md` 在本次读取中返回 404；运行入口、许可证和平台依赖还需在真正采用前核查，不把论文可读等同部署即用。
- **对图模型的质疑：**如果图边是脱敏 IP 的偶然相等而非可靠交互，更多图层会放大伪关联。PIDSMaker 对训练扰动与结果不稳定也有明确提示；不能挑出最好一次种子就与本轮固定批次比较。
- **对语义向量的质疑：**语言模型可能把否定、端口方向和“记录了被拦截请求”压成相似句；加一个编码器必须保留事实字段和反例，单独衡量表示贡献。先比较后组合，禁止同时改解析、加权、阈值与神经网络后归因。
- **对标签清洗的质疑：**在来源不支持的训练里，模型低置信度/不同意标签可能恰好指向最重要的困难样本。复核需要独立字段证据或人工确认，不能让模型自证正确。

公开网页和源码已阅读，GitHub API 的 HEAD 身份查询遭遇速率限制，未取得提交号；本报告采用明确 URL 与读取日期，不声称已锁定所有仓库版本。实际引入代码前须绑定确切版本、许可证、依赖与模型权重来源。查询记录：[只读身份请求结果](../evidence/2026-09-12/v331_result_review/open_source_identity.json)。
