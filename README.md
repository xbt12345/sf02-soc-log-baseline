# SF02

SOC 日志威胁检测 baseline 工程。

当前仓库只实现最小可行主线：

`结构化字段 -> CatBoost -> 防泄漏切分 -> 本地验证 -> 预测导出`

这版实现遵循两个原则：

1. 先把本地验证链路做可信，再谈复杂模型。
2. 先做 Route A 主线，Route B/C 只在说明中预留。

## 当前数据状态

- 训练集压缩包：`C:\Users\xiabutian\Downloads\data.zip`
- 解压工作目录：`C:\Users\xiabutian\sf02_data\data`
- 训练集：`train.parquet`
- 预测输入：`valid_input.parquet`

## 当前实现

- `scripts/00_inspect.py`
  - 检查数据形态、字段、缺失率、标签分布、实体可关联性
- `scripts/10_baseline.py`
  - 训练 CatBoost baseline
  - 输出 dummy baseline 对比
  - 输出本地验证指标
- `scripts/20_predict.py`
  - 读取模型
  - 对预测输入生成 `res.csv`
  - 校验 `event_id` 和输出格式
- `src/common.py`
  - 配置读取
  - 数据读取
  - 简单特征构造
  - 分组切分
  - 指标计算

## 计划中的后续路线

- Route B：`message_sanitized` 文本增强
  - TF-IDF
  - sentence embedding
  - 小型 transformer 微调
- Route C：实体/时间窗上下文增强
  - 依赖测试集实体是否还能稳定关联

## 运行方式

先激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

先做 inspect：

```powershell
python .\scripts\00_inspect.py --config .\config.yaml
```

跑 baseline：

```powershell
python .\scripts\10_baseline.py --config .\config.yaml
```

先做一版快速烟测：

```powershell
python .\scripts\10_baseline.py --config .\config.yaml --sample-rows 200000
```

生成预测文件：

```powershell
python .\scripts\20_predict.py --config .\config.yaml
```

## 目录

```text
SF02/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/
│   └── README.md
├── papers/
│   └── README.md
├── scripts/
│   ├── 00_inspect.py
│   ├── 10_baseline.py
│   └── 20_predict.py
├── src/
│   └── common.py
└── artifacts/
```
