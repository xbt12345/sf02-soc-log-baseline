# v3.3.1：上传一个文件，运行一条命令

2026-09-12 状态更新：**本批次已由用户在平台运行完成，9 个任务、27 次拟合及回传核对通过；来源迁移未通过。** 见[完整结果](../docs/V331_CLOUD_RESULT_REVIEW.md)和[当前 v3.5.1 方案](../docs/TRAINING_PLAN.md)。以下包与命令保留作历史复现，无需重复执行，也不是 v3.5 执行包。

## 1. 上传

将本地 [sf02_v331_update.py](sf02_v331_update.py) 上传到平台 JupyterLab 的 `/root/work`。只有这一个文件，120,086 字节，约 117 KiB；不用解压、拆分或再次上传大数据。

程序复用平台已有文件：

- `/root/work/sf02_data/train.parquet`
- `/root/work/sf02_v32_round/prepared/prepared_corpus.parquet`

保留上述文件及原文件夹名称。更新文件包含程序、配置摘要和小型回归样例，不包含完整训练集或训练好的模型。

## 2. 运行

在 JupyterLab 的 Terminal 粘贴：

```bash
conda run --no-capture-output -n cm-model python -u /root/work/sf02_v331_update.py
```

程序使用现有 `cm-model` 的 Python 和 CPU，不安装依赖，不使用昇腾卡，不修改官方数据或标签。请保持平台实例和终端会话运行；不要同时重复启动同一个任务。

它会依次完成：

1. 核验更新内容、官方原件和旧准备缓存；解出专用运行目录。
2. 执行 44 项原回归检查与 22 项新检查。显示 `SOFTWARE_CHECKS_ONLY` 和 `Ran ... tests / OK` 时只是程序检查通过，还没得到真实模型成绩。
3. 修复 ASA/Duo 输入，重建全部数据的分组；独立核对内容和 9 组拟合/选参/校准/评价职责，并与本地已核验的完整内容摘要比较。
4. 执行域内 3 折、ASA 模板族 3 折、AD/Duo/WAF 各一次来源留出，共 **9 个任务、27 次分类器拟合**。每个任务包括两个 C 候选和选定后的一次重新拟合。
5. 回放每个最终模型的全部校准/评价预测，复算选择依据和报警计数，生成一个结果 ZIP。

长阶段每 20 秒显示运行状态，词表准备每 20 万行显示进度。一个任务显示 `trained_task_complete` 后仍会继续其他任务。新批次尚无云端实测时长，不承诺几点完成；阶段日志会记录真实耗时。

## 3. 取回结果或恢复

全部任务执行和回放完成后，终端会显示：

```text
BATCH_EXECUTION_COMPLETE; model quality still requires result review.
RETURN_THIS_ZIP: /root/work/sf02_v331_8a10b64bb4/work/v331_results_时间.zip
```

下载 `RETURN_THIS_ZIP` 后面的那个 ZIP，发回本对话。它含模型、逐条分数、训练与阈值记录、各来源指标和检查证据，后续才能判断修复是否有效；执行完成不等于达到低误报或迁移目标。

若中断，保持运行目录原样，**重新运行同一条命令**。程序会核验并复用完整任务；未完成任务重新开始，不覆盖旧尝试。恢复粒度是任务，不是某次模型拟合的迭代数。已完成文件损坏、源码/数据/环境身份不符时会停止，不冒充缓存命中。

若失败并显示 `RETURN_FAILURE_ZIP`，下载该 ZIP 发回。如果启动阶段就提示 `Official train missing or damaged` 或 `Existing v3.2 prepared cache missing or damaged`，说明指定原文件不存在或校验不一致，把完整报错发回即可；不要删除已有训练结果。若小更新文件自身上传不完整，重新上传这一个文件。

程序只在上述检查通过后进入训练。实际 Linux/ARM 导入兼容性、可用内存和完整训练均以平台当次运行结果为准；本地演练没有代替这些检查。

## 交付身份

执行文件 SHA-256：

```text
fc39f5dc4bd60067b7f09ca241d29d5c695c012c40ab4d109e618e1558b08a3b
```

通常无需手动计算，程序会核验内嵌内容。完整生成记录见 [v331_bundle_receipt.json](v331_bundle_receipt.json)，本地实施证据见 [V331_IMPLEMENTATION_REVIEW.md](../docs/V331_IMPLEMENTATION_REVIEW.md)。
