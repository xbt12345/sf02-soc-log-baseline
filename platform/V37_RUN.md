# v3.7 平台重训

状态补记：该轮已完成，回传文件为 `v37_results_20260913T072309Z.zip`。见 [完整结果审查](../docs/V37_CLOUD_RESULT_REVIEW.md)。以下命令保留作历史复现，当前无需原样重跑；下一版方案尚未打包。

本文件对应 `sf02_v37_update.py` 单文件执行包（146,009 字节）。SHA-256：`cd3ec604306196c19d0eb2e165c7a0177befa7be270bde69a01ef95a3bbcb3d2`。仅使用已有的官方 `sf02_data/train.parquet`，不重新上传训练集，不安装软件，不修改 `cm-model`。

将执行包上传到平台 `/root/work/`，在平台终端运行：

```bash
cd /root/work
conda run --no-capture-output -n cm-model python sf02_v37_update.py
```

它会依次核对上传文件与官方数据、运行软件检查、重建全量输入和共同分组、逐行独立检查并与本地内容指纹对照，再执行五协议的新旧 B2 配对训练。每个协议、每个分支比较两个 C，随后按选定 C 重新拟合，合计 10 个模型、30 次拟合。训练保持原行等权，没有重复降权分支。

之后重新加载所有模型，回放全部校准／评价概率，并对完整 WAF 4,130 行及其他实际样例检查包装变换前后的概率、类别和全部冻结报警阈值。修复分支的等价变换不通过，就不会输出执行成功标记。旧对照的包装失败会作为对照证据保留。

`work/primary_decisions/` 保存冻结主决策的逐条三类 `prediction`，同表保留 `raw_argmax` 对照。模型目录原始评价表里的 `prediction` 是旧格式的 argmax 诊断列；最终决策以 `primary_decisions` 为准。程序调用入口为 `v37_infer.classify()`。

终端可能长时间停留在某个阶段；约每 20 秒会输出运行提示。不要因没有新成绩而再次同时启动。断线后重新执行同一命令，会核对并复用已完成且身份未变的模型；未完成的模型保留原记录，另开一次尝试。已完成结果发生内容变化时会停止。

成功结束会出现：

```text
BATCH_EXECUTION_COMPLETE; quality requires reviewing the results.
RETURN_THIS_ZIP: /root/work/sf02_v37_.../work/v37_results_....zip
```

请下载并回传 `RETURN_THIS_ZIP` 后面的一个 ZIP。失败时也会尽量生成 `RETURN_FAILURE_ZIP`，直接回传该文件即可，不用删除数据或重建环境。

数据若不在默认位置，只修改 `--data`：

```bash
conda run --no-capture-output -n cm-model python sf02_v37_update.py --data /实际目录/train.parquet
```

默认 8 个计算线程、4 个准备进程；使用 CPU，不调用昇腾卡。可以先仅验证执行包上传完整性：

```bash
conda run --no-capture-output -n cm-model python sf02_v37_update.py --verify-only
```

不要把 `model_trained: false` 的解包回执当成训练失败：它仅说明解包这一步尚未训练。正式执行以最后的批次与 ZIP 回执为准。

阶段 D 的条件分型不在这个包中根据目标错误自动启动。先审查 C 的匹配结果，确认风险排序可用且有支持的非冲突 ASA 分型仍有欠缺，再决定是否执行事前限定的条件分型对照。没有默认晋升新模型，也不把当前内部回归称为外部迁移通过。

输入修复、实测验收和已知限制见 [本次执行审查](../docs/V37_EXECUTION_REVIEW.md)，冻结方案见 [V37_NEXT_PLAN.md](../docs/V37_NEXT_PLAN.md)。
