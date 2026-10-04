# 官方数据

当前工作副本集中在 `official/`，原始数据不入库。

2026-10-04 进度核对：最新实际训练 V164、最新完整质量交付 V159，V169 平台执行包已交付但实际平台结果待回传。以下准备资格说明属于 2026-09-12 的历史阶段；当前情况见 [最新进度](../docs/CURRENT_PROGRESS.md)。数据使用边界和原始文件保持不变。

2026-09-12 当前约束：只用官方数据，不寻找外部训练集。本轮 v3 准备与对照只读取 `train.parquet`；其全量均为已审计开发材料，新切分不产生盲测。当前准备缓存尚未通过主训练资格，见 [实施审查](../docs/V3_PREPARATION_REVIEW.md)。

| 文件 | 行数 | 用途 |
|---|---:|---|
| train.parquet | 2,056,871 | 官方有标签训练数据 |
| valid_input.parquet | 2,014,052 | 官方验证输入 |
| valid_answer_private.parquet | 2,014,052 | 本地已有验证答案；已用于审计 |

目标列虽名为 `label_binary`，实际为 `benign / malicious / suspicious` 三类。输入只有 `event_id, timestamp, pipeline, src_ip, dst_ip, src_port, src_host, dst_host, username, message_sanitized, product_name, vendor_name`；不能假设存在独立的动作、严重度或 MITRE 标签。

`event_id` 只在各自数据集内部是主键。训练与验证重新使用同一编号空间，不能跨数据集按它关联。验证答案只能与验证输入按 ID 一对一关联。

此次移动已逐文件核对 SHA-256，见 [移动清单](../evidence/organization_moves_2026-09-11.json)。根目录临时解压副本经同哈希确认后删除；工作区以外的 `C:\Users\xiabutian\sf02_data\data` 未修改，当前配置不再指向它。

分布、空值、重复和时间关系见 [全量数据审计](../evidence/2026-09-11/data_summary.json)。新数据选择与标签可比性要求见 [当前方案](../docs/TRAINING_PLAN.md)。
