# SF02 P_IS SOC 日志三分类

读取官方原始 Parquet/CSV，自动完成预处理、模型预测，生成比赛要求的 `res.csv`。输出标签为 `benign`（正常）、`suspicious`（可疑）、`malicious`（恶意）。使用已发布权重预测无需重新训练，也无需自行生成数值特征。

## 安装和下载

使用 Python 3.10–3.12；本次实测环境为 Python 3.12、Windows、CPU/NVIDIA CUDA。建议在虚拟环境中安装。

```bash
git clone https://github.com/xbt12345/sf02-soc-log-baseline.git
cd sf02-soc-log-baseline
python -m pip install -r requirements.txt
python download_weights.py
```

下载器只安装 `weights/` 下的 9 件模型组件，不覆盖代码；下载包和每件权重均核验 SHA-256。[当前权重发布页](https://github.com/xbt12345/sf02-soc-log-baseline/releases/tag/sf02-pis-competition-20261007)也提供手动下载。官方原始数据须由参赛者自行取得，仓库不包含数据或私人运行环境。

## 直接生成比赛文件

把官方测试文件放在本地，例如 `test.parquet`，执行：

```bash
python predict.py --input test.parquet --output res.csv
python validate_submission.py --input test.parquet --submission res.csv
```

CSV 输入同样支持：`--input test.csv`。原始文件需有 `event_id`、`message_sanitized`；使用官方 `src_port` 列时保留它，缺少该列会按未知端口处理。`event_id` 应为非空、唯一的字符串；CSV 会保留前导零。模型预测不读取 `label_binary`，也不根据记录 ID、产品名或人工答案查表。

输出恰为两列：

```csv
event_id,pred_label
event_001,benign
event_002,suspicious
```

每个输入 ID 输出一次，包括空正文、未知格式及难例。提交校验器独立检查 ID 集合、重复、遗漏、多余和标签枚举。运行另产生 `res.report.json`，记录行数、标签分布、模型折、回退情况和文件散列。已有输出不会被覆盖；输入错误时不会留下一个看似完成的 `res.csv`。

默认有 CUDA 时使用 GPU，否则使用 CPU。可指定：

```bash
python predict.py --input test.parquet --output res.csv --device cpu --batch-size 2048
```

推理按批读取输入，ID 去重使用临时磁盘数据库。稀疏神经网络只计算批次出现的特征列，CPU 分块计算成员，避免原先每次前向固定分配约 518 MiB 中间矩阵。

正文较复杂的大文件可加 `--workers 4` 并行预处理；队列最多保留每个进程两批数据，输出仍保持原行顺序。默认单进程适合内存较少的机器。

## 使用的模型和预处理

当前 9 件权重仍为 **V128r3 P_IS，第 50 轮残差终点及其冻结基座、完整分类器**。每折的三件组件共同构成一个完整三分类模型。

`sf02_model/preprocess.py` 自动从原始日志恢复匹配的输入：65,792 维 UTF-8 字节一元/二元统计、477 维正文事实、18 维记录源端口。事实与文本各自按原合同归一化；记录端口独立编码。ASA 路径还生成有序正文及独立长度，最多 176 字节，不静默截断。[输入合同](input_contract.json)给出细节。

默认对三折的完整预测概率平均，再选最大类别，不需要用户提供旧开发折映射。`--fold 0` 可显式仅用单折。ASA 正文不符合已注册头部规则或超过长度时，使用同一折的完整稀疏分类器预测，并在报告中计数；`--strict-asa` 可改为报错。未解析格式仍保留文本通道和可观察的记录端口，不丢弃样本。

**历史 Macro-F1 98.905972% 是原来源留出折的开发结果，不能直接作为默认三折平均的未知测试成绩。** 历史结果、来源覆盖限制及本次工程验证范围见 [模型说明](MODEL_CARD.md)、[指标](metrics.json)和 [验证记录](verification.json)。

当前默认入口已实际跑完 **2,014,052 条官方验证输入**，独立提交校验通过；按此前已查看的官方验证答案计分，Macro-F1 为 **92.139001%**。预测不读答案，评分不用于改参数或选择策略；此结果不是官方隐藏测试成绩。

## 从原始官方数据重新训练

`train.py` 提供独立可运行的同架构训练入口。输入包含 `label_binary`，标签必须为三类官方字符串，必须有完整三类及受支持 ASA 样本：

```bash
python train.py --input train.parquet --output my_model --device cuda
python predict.py --input test.parquet --weights my_model --output res_my_model.csv
```

默认完整读取训练文件，固定基座 25 轮、冻结后正文残差 50 轮，并拟合完整三分类的稀疏分类器。`my_model/manifest.json` 记录新权重 SHA、种子、轮次；另保留首个参数更新前的训练计划/源码/输入绑定、实际更新记录和训练结果。已有模型目录不会被覆盖。没有 CUDA 时可用 `--device cpu`。

该入口重新训练同一分类架构，采用自身训练基座的训练内分数；它不逐位复现原历史嵌套教师实验，生成的新权重也不继承历史分数。可用 `--validation separate_labelled.parquet` 独立评分；验证数据不参与梯度、轮次或检查点选择，共享输入情况会单独报告。

短轮次开发检查可显式使用 `--base-epochs 1 --residual-epochs 2`；`--limit-rows` 仅用于明确的子集试跑，子集仍须包含三类及 ASA。完整训练会在内存中组装稀疏特征，内存需求高于分批预测；短轮次或子集跑通只证明训练程序能运行。

## 检查代码

```bash
python -m unittest discover -s tests -v
```
