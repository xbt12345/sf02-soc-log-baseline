# Data Notes

原始比赛数据不入库。

## 当前路径

- 压缩包：`C:\Users\xiabutian\Downloads\data.zip`
- 工作目录：`C:\Users\xiabutian\sf02_data\data`
- 训练集：`train.parquet`
- 预测输入：`valid_input.parquet`

## 当前字段理解

- `event_id`
  - 唯一主键
  - 提交文件必须保留
- `label_binary`
  - 当前训练标签
  - 实际是三分类：`benign / malicious / suspicious`
- `timestamp`
  - Unix 时间戳，秒级浮点
- `message_sanitized`
  - 脱敏原始日志消息
  - 当前 baseline 不直接使用，后续 Route B 使用
- `src_host / src_ip / username / dst_ip`
  - 可作为分组或上下文候选实体

## 当前工程选择

- Route A baseline 只用结构化字段
- 文本列先留作后续增强
- 默认采用分组切分优先，避免随机切分泄漏
