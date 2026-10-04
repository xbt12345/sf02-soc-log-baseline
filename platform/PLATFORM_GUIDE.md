# 赛事平台怎么用于 SOC 模型训练

2026-09-13 当前入口：[v3.6 单文件运行说明](V36_RUN.md)。本地输入、协议、试训和回放已完成，新执行包已生成；平台全量训练尚未运行。历史版本无需重复启动。下文配置与旧命令仅作历史说明。

历史方案状态：v1/v2 仅保留复现；v3 初版未通过准备审查；v3.2 与 v3.3.1 已执行并保留完整结果。此前输入修复记录见[实施审查](../docs/V331_IMPLEMENTATION_REVIEW.md)。本文其余部分说明平台能力和历史操作，不表示应再次启动旧批次。

核查日期：2026-09-11。结论：**平台能够承载本项目的模型训练。用户截图与容器回传确认 CPU 配额等效 40 核、内存上限 320 GiB，截图另有 2 张昇腾 910B；当前使用 CPU。** cm-model 七个训练包和 `/root/work/sf02_data` 两份 parquet 元数据已检查通过；真实空闲资源、持久挂载和额度规则仍需核实。

用户已回传 v1.1 云端真实日志试验：223.404 秒、进程峰值 RSS 1.1443 GiB；后续只读核对确认清单内容和逐行预测类别与本地一致。**环境与 v1 内容核对已结束，无需再执行下方保留的历史排查命令。** v2 已在本地完成行为解析与来源留出，仍未通过模型验收，当前不建议云端重复或扩大这个版本。最新结果见 [v2 复核](../docs/BEHAVIOR_V2_REVIEW.md)，复现入口见 [训练说明](../training/README.md)。

此次 Edge 浏览器连接仍失败，随后以不带登录凭据的 HTTP GET 成功读取主办方站点的公开 HTML 和前端 JavaScript。下面的菜单、提示与功能依据来自这些实际资源，不是对已登录页面的截图观察，也不是云端试跑成功的声明。前端功能可见性仍受账户权限和资源池策略影响。

[用户提供的截图](instance_configuration_user_screenshot.png) 是另一份证据：开发环境 `Mindie`，运行镜像 `Mindie12`，资源组 `huanxin-all-resource`，1 个运行实例，最近启动时间 2026-09-11 14:08:43。上述规格是页面配置值，不代表此刻空闲资源。截图下方只显示挂载数据集表头，不能判断是否已有数据挂载。

`Mindie12` 是镜像名称，不能据此确认内置框架或版本。华为的 MindIE 本身定位为推理引擎；这不妨碍容器中的 CPU 跑普通训练，但不能证明镜像已配好昇腾训练环境。CatBoost 官方 GPU 后端依赖 NVIDIA/CUDA，不能将两张昇腾卡直接用于 `task_type="GPU"`。当前 CatBoost 使用 `task_type="CPU"`；如后续独立验证证明需要神经网络，再核查昇腾训练框架与驱动兼容性。

来源：[CatBoost GPU 要求](https://catboost.ai/docs/en/concepts/cli-installation)、[华为 MindIE 定位](https://www.hiascend.com/document/detail/zh/mindie/310/index/index.html)。

## 1. 这个链接是什么

你给出的 `/train-dev/detail/dl-…` 对应 **模型训练 → 开发环境 → 开发环境详情**。公开路由同时包含新建开发环境、进入环境、实例监控，以及独立的训练任务、新建训练任务、任务日志和 Shell 终端。

| 功能 | 对本项目的用途 |
|---|---|
| 开发环境 | 在 Jupyter、VSCode 或 Shell 中检查数据、调试解析器和做小规模训练 |
| 训练任务 | 使用确定的镜像、数据、资源与执行命令进行正式实验 |
| 挂载数据集 | 将训练数据提供给容器，避免每次反复上传 |
| 保存镜像 | 保存新安装的软件与依赖，以便后续环境复用 |
| 实例监控/日志 | 查看资源占用、排队/失败状态和运行日志 |

开发环境列表的说明明确支持在云端开发和训练。详情页的公开实现存在“运行”“打开”“停止”“编辑”“保存镜像”等操作，按钮是否显示/可点由实例状态与权限决定。

来源：[平台入口](https://aihuanxin.cn/kunlun/kl-web)、[开发环境说明资源](https://aihuanxin.cn/kunlun/kl-web/js/chunk-a798272c.30ed8109.js)、[详情页资源](https://aihuanxin.cn/kunlun/kl-web/js/chunk-745364fa.df5ee451.js)。摘录与哈希见 [公开来源证据](public_platform_evidence.json)。

## 2. 从你这个详情页开始

1. 先查看“基本信息”和“运行实例”：记录环境状态、镜像/IDE、CPU、内存、加速卡型号、资源组和挂载数据集。此前若已有模型和运行日志，先确认来源和状态，避免重复开跑。
2. 截图中实例已经“运行中”，当前应使用“打开”进入开发环境，无需再次启动或新建。若之后已停止且存在“运行”，先核对额度与回收规则，再启动并等待就绪。排队、锁定、失败要看对应提示；不能假设重试一定解决。
3. 进入环境后，按所选镜像提供的 Jupyter/VSCode 使用；公开环境页也有“Shell终端”标签。需要先打开环境，不是在详情页直接输入 Python。
4. 先做下面的环境检查，确认 Python、安装包和数据位置；之后再执行训练。

如果需要新建环境，入口是“模型训练 → 开发环境 → 新建”，流程是命名、选择训练镜像/IDE、选择资源组、配置单实例资源和实例数、挂载数据。当前先用 **1 个实例**。公开创建表单支持 CPU 等芯片类别与 X86/ARM 架构，但你账号实际可选项未知。

截图中的 40 核、320 GB 配置足以作为当前约 205 万条训练日志的 CPU 基线与特征工程起点，无需为了开始训练另开更大实例；具体运行时间和峰值内存仍需实测。先用单实验、少量线程完成检查，再逐步扩大；大量长文本仍采用分块读取，避免不同实验各自复制全量数据。加速卡是否参与训练取决于软件适配，不影响先使用 CPU。

## 3. 环境和数据怎么准备

后续已成功读取官方帮助中心的《开发环境》《文件管理》《存储相关问题》正文，以下上传方式与默认路径有官方文档依据。文档更新日期为 2025-06-27；当前实例的实际挂载和镜像能力仍以容器检查为准。

### 3.1 小脚本优先通过 Jupyter 上传

进入“开发环境 → 打开 → Jupyter”，将本地 `check_environment.py` 拖入左侧文件列表；确认出现文件后，在对应目录的 Terminal 中运行。官方支持通过 Jupyter 直接上传 100 MB 以内的文件。自定义镜像若没有安装 Jupyter，该页可能不可用，此时使用可用的 VSCode 编辑器或 Shell。

来源：[官方开发环境说明，第 3 节](https://aihuanxin.cn/portal/common-helpcenter#/document/383?platformCode=DMX_KFPT)。

### 3.2 当前“文件获取失败”的准确含义

用户截图对应“文件管理 → 对象存储桶 → 本地上传”，目标 `jtdlp-…` 是对象存储桶，不是容器目录。当前脚本 4,768 字节，远小于这个入口的 100 MiB 上限。公开上传组件的前置校验只限制大小，未见扩展名白名单；官方文件管理文档也说明对象存储可保存任意类型数据，不能据此把 `.py` 当作不允许上传的类型。

同一提示覆盖两种情况：上传请求进入错误状态；或上传返回后响应不满足 `code === 0`。因此截图无法区分会话/权限、请求连通性、上传服务或对象存储业务错误。未取得真实失败请求的状态码和响应正文，不给出确定根因。换用 Jupyter 是小文件的官方替代路径，不代表对象存储故障已经修复。

若需要定位该弹窗：在 Edge 按 F12 打开 Network，保留日志，重试一次上传；查看这次失败的上传请求，记录 HTTP 状态码、响应正文中的 `code/msg`。不要复制 Authorization、Cookie、AK/SK 或包含登录凭据的完整请求。401 通常查登录会话；403 查权限/访问策略；5xx 查网关/服务；HTTP 200 仍需看业务错误码；没有 HTTP 响应时查看网络或浏览器错误。上述是排查方向，具体归因以响应为准。

来源：[官方文件管理说明](https://aihuanxin.cn/portal/common-helpcenter#/document/551?platformCode=DMX_KFPT)、[当前公开上传组件](https://aihuanxin.cn/kunlun/kl-web/js/chunk-9d77d14a.e40a9ec5.js)。

### 3.3 数据文件与持久目录

官方默认 `/root/work` 为项目高性能文件存储，`/root/work/filestorage` 为普通文件存储，项目内开发环境共享相关目录。先检查实际挂载，再新建项目子目录；避免覆盖他人的文件。

本地 `train.parquet` 为 127,118,646 字节（约 121.23 MiB），已经超过网页 100 MiB 上限，不能原样走这个本地上传弹窗。本次 Jupyter 小文件上传已成功，因此已准备 [upload_bundle](upload_bundle/)：将训练文件按原始字节分成 64 MiB 和 57.23 MiB 两部分，另带 66.70 MiB 的完整验证输入和还原脚本，共 4 个文件，每个都低于 100 MB。分片是本项目的传输处理方式，不是修改 parquet 内容，也不是声称官方专门推荐此方式。本地实际还原后 SHA-256 与原件一致，验证记录见 [upload_bundle_verification.json](upload_bundle_verification.json)。

本次上传步骤：

1. 在云端终端执行 `mkdir -p /root/work/sf02_data`。
2. 刷新 Jupyter 左侧文件列表，进入 `sf02_data`，将本地 `platform/upload_bundle` 内的 4 个文件全部拖入，等待上传结束。
3. 执行 `python3 /root/work/sf02_data/restore_and_verify.py`。它只依赖标准库，可使用已经确认可用的系统 python3。
4. 仅在输出 `verified: true` 且状态为 `restored_and_verified` 或 `existing_original_verified` 后，执行 `conda run --no-capture-output -n cm-model python /root/work/check_environment.py --data-dir /root/work/sf02_data`。

还原脚本校验每个分片、合并结果与完整验证文件；同名输出存在时只核验，不覆盖；失败时保留输入材料。它不会训练模型、修改样本或上传验证答案。官方原件保留在 `data/official`。本地还原已验证；用户后来回传 v1.1 真实训练完成，其必经 SHA-256 核验先于训练，支持云端输入身份检查通过。本次没有清理传输副本。

持续传输更大数据时，官方《存储相关问题》推荐使用 S3 客户端传入对象存储；Windows 可用文档所称的 S3 Browser（当前官网已使用 CS Browser 名称），选择 S3 Compatible Storage，在本机填写项目详情中给出的 Endpoint、AK、SK。之后在文件管理中选择文件并“同步”到目标文件存储，或者使用开发环境已有的 `mc` 客户端。页面上传对象存储成功并不等于容器中已有该文件。

官方存储 FAQ 部分内容还要求启动本地 MinIO 服务，但连接现有远端对象存储并不需要自建存储服务；本任务采用文档已有的 S3 客户端路线。不要仅为迁就文本问答数据集的导入规则把 SOC parquet 转成问答格式；这里可以作为普通文件存储后由训练代码直接读取。

来源：[官方存储相关问题](https://aihuanxin.cn/portal/common-helpcenter#/document/404?platformCode=DMX_KFPT)。

### 3.4 环境检查历史（当前已完成）

用户已通过 Jupyter 上传并用 `python3` 成功执行检查脚本，回传结果为 Linux/aarch64、Python 3.8.10、工作目录 `/root/work`。`python` 命令在当前终端不存在。容器 CPU 配额 4,000,000 / 100,000 = 40 核等效额度，内存限制为 320 GiB；操作系统报告的 192 逻辑 CPU 不等于获配额度。当前 Python 的七个依赖均未发现，但这不能代表其他预装环境也缺失。没有传入数据目录或微型训练参数，两项都尚未执行。记录见 [云端检查回传](environment_probe_cloud_user.json)。

当时用于定位预装环境和安装工具的命令如下，当前无需重复：

```bash
command -v python3 conda
conda env list
python3 -m pip --version
```

上面的命令用于定位预装环境。当前实例的结果已回传，下一段给出实际缺包情况与补齐方法。已核对 PyPI 的对应二进制发行；但安装包存在不等于已在本实例通过安装或运行验收。

**后续已确认 `cm-model` 存在。** 用户回传其路径为 `/opt/miniconda3/envs/cm-model`、Python 3.8.18，已有 numpy 1.24.4、pandas 2.0.3、pyarrow 14.0.1、PyYAML 6.0.1；缺少 scikit-learn、catboost、matplotlib。系统 `/usr/bin/python3` 没有 pip，只说明系统解释器状态，不应在系统 Python 中补装。检查脚本因缺包返回非零状态，所以 Conda 报命令失败；它已经成功启动了该脚本，不能据此判断 Conda 损坏。

以下是首次补齐依赖的命令记录，用户后来已补齐所需包并跑完真实试验，当前无需重复安装。原命令锁定四个基础包，选择已核对 CPython 3.8/Linux ARM64 二进制包的版本，并禁止回退到源码编译；需要重建环境时再按实际缺项参考，不全局升级 Python 或依赖。

```bash
conda run --no-capture-output -n cm-model python -m pip install --only-binary=:all: \
  "numpy==1.24.4" "pandas==2.0.3" "pyarrow==14.0.1" "PyYAML==6.0.1" \
  "scikit-learn==1.3.2" "catboost==1.2.10" "matplotlib==3.7.5"
```

仅在安装成功后检查依赖与微型 CPU 训练：

```bash
conda run --no-capture-output -n cm-model python -m pip check
conda run --no-capture-output -n cm-model python /root/work/check_environment.py --smoke-fit
```

预期依赖检查无冲突，JSON 中 `requested_checks_passed` 与 `synthetic_cpu_fit.succeeded` 为 true。若 `pip check` 报其他框架冲突，不能直接认定由本次安装造成；应依据具体包和原有环境核查，不自动升级昇腾框架。上述验证只证明基础环境与微型 CPU 拟合，未检查官方数据或模型迁移能力。需要停止/重启实例前，按平台规则保存环境。

版本来源：[scikit-learn 1.3.2](https://pypi.org/project/scikit-learn/1.3.2/)、[CatBoost 1.2.10](https://pypi.org/project/catboost/1.2.10/)、[Matplotlib 3.7.5](https://pypi.org/project/matplotlib/3.7.5/)。

**最新回传：微型 CPU 训练已通过，整个镜像的依赖检查未通过。** 用户确认 `requested_checks_passed` 和 `synthetic_cpu_fit.succeeded` 均为 true；`pip check` 则报告 `op-compile-tool` 将 getopt/inspect/multiprocessing 声明为缺失依赖，schedule-search 缺 absl-py，te 缺 cloudpickle/synr，以及 apex 的非标准版本号弃用提示。

三种标准库不能按普通第三方包补装，相关报告指向工具包的依赖声明问题。其余三个是实际的第三方依赖缺口，涉及昇腾工具链；当前通过的 CatBoost CPU 微型拟合没有覆盖这些工具。apex 提示是安装工具对非标准版本号的兼容性警告，不能据此直接升级 apex 或 pip。没有安装前的完整 `pip check` 记录，不能断言缺口全部预先存在，也不能标记整个镜像依赖健康。当前保留这些问题记录，推进 CPU 数据检查；如后续使用昇腾相关功能，再按对应 CANN 版本修复和验证。

当时用于查找官方数据位置的命令如下，当前已确认 `/root/work/sf02_data`：

```bash
find /root/work -maxdepth 4 -type f \( -name 'train.parquet' -o -name 'valid_input.parquet' \) -print
```

该搜索没有输出只说明这个目录深度内未发现这两个文件，不等于整个平台没有数据。后续官方数据读取和 v1 真实抽样训练均已通过；全量训练性能、NPU 训练或迁移效果仍未验收。

参考：[Python 标准库](https://docs.python.org/3/library/)、[CANN 示例对 op-compile-tool 依赖声明的说明](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/locomotion/LQC/README.md)、[昇腾迁移依赖说明](https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/foundmodeltrain/PT_LMTMOG_000008.html)、[pip 非标准版本号处理](https://github.com/pypa/pip/issues/12063)。

在环境里的终端先运行以下只读命令（适用于常见 Linux 镜像）：

```bash
pwd
python --version
uname -m
df -h
```

按上面的 Jupyter 路径上传 [check_environment.py](check_environment.py) 后，在脚本所在目录执行：

```bash
python check_environment.py
```

脚本会列出 Python、架构、包版本、容器 CPU/内存限制。操作系统 CPU 数不一定等于容器获配额度，需一起看容器限制和平台资源信息。它不安装依赖、不读取登录信息、不访问网络。

若报告缺少依赖，在合适的项目环境中安装缺少的包。已有平台环境先检查后安装，避免无依据地整体升级；本项目基础依赖为 `pandas pyarrow scikit-learn catboost pyyaml matplotlib`。例如缺少这些包时才执行：

```bash
python -m pip install pandas pyarrow scikit-learn catboost pyyaml matplotlib
```

依赖安装成功后记录版本，正式训练保持该镜像和版本。镜像内没有网络或目标架构没有兼容包时，需要使用平台预置镜像或合适的离线包，不能把安装错误当成数据问题。

代码与数据分开准备：

- 代码上传 `SF02` 的必要源码/配置/脚本，排除 Windows `.venv`、`.git`、旧大预测文件和不需要的历史材料；云端使用自己的 Python 环境。
- 数据优先检查平台是否已提供官方集。有则挂载并核对；没有时，在允许的数据上传入口导入本地 `SF02/data/official` 中所需文件。不要覆盖已有同名数据。
- 必须从容器中确认实际挂载路径。官方默认目录规则见 3.3；本次尚未在用户容器执行挂载检查。

确认路径后，用实际值替换下方示例的 `/实际挂载路径`：

```bash
python check_environment.py --data-dir '/实际挂载路径' --smoke-fit
```

它只读取 `train.parquet`、`valid_input.parquet` 的元数据，核对行数/关键列；不会将 205 万条长文本全部加载。`--smoke-fit` 仅在 96 条人工样本上用 CPU 拟合 12 棵树并检查预测形状，用于证明训练库可以执行，**不评价 SOC 检测质量**。行数与列匹配也不能替代 SHA-256 身份核验。

可用 `sha256sum` 对训练文件核验，官方本地训练集哈希为：

```text
6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742
```

以上哈希已在本地重新计算核对。完整数据身份记录见 [data/README.md](../data/README.md) 链接的原始移动清单。

## 4. 怎么开始正式训练

先在开发环境完成预处理、冻结切分与小规模完整流程，再将同一套可复现命令放到 **模型训练 → 训练任务 → 新建训练任务 → 执行命令**。新建表单公开实现包含训练镜像、执行命令、资源组、实例数、单实例资源、挂载数据集；训练任务还提供日志与监控。

执行命令需要明确：切换到真实持久化项目目录，调用确定的 Python，传入当前实验配置，把模型/日志/指标输出到该实验的独立持久目录。不要把临时容器目录中的成功文件视为已经保存，也不要让实验覆盖旧模型。定时停止、最长运行时间、任务重试/断点机制与计费以平台实际规则为准，目前尚未核查。

**`scripts/10_baseline.py` 是旧结构化基线；v1 入口为 `training/soc_signal_pilot.py`，v2 为 `training/soc_behavior_v2.py`。** 旧程序保留作历史对照。v1 核验 SHA-256 后均匀抽样并按消息组隔离用途；v2 在冻结样本上检查新/旧分组连通、隔离跨区关联，再比较行为、权重、上下文与来源留出。详情与复现命令见 [training/README.md](../training/README.md)。两版均为有限试验，尚无模型通过检测与迁移验收。

正式每个实验保存：数据/划分/模型/配置哈希、包版本、三类指标、每万正常误报数、固定误报率下召回、分来源表现、表示变化测试、耗时和峰值内存。算力能跑通与检测效果合格是不同验收。

## 5. 保存与结束

平台公开提示了一个真实风险：停止/重启或闲置回收后，新安装的软件与依赖可能丢失。需要复用安装环境时使用“保存镜像”，并等镜像保存成功。

另一个公开提示是：保存镜像不包含挂载的数据和文件，而且保存过程中实例可能短暂中断。因此应先停止关键运行/保存检查点，再做镜像保存；代码、训练数据、模型、指标另外存入已确认的持久存储，并下载必要备份。确认产物保存后再停止实例。不要把关闭网页等同于停止算力。

来源：[环境页与镜像保存提示](https://aihuanxin.cn/kunlun/kl-web/js/chunk-e8d11d9e.1e4031c3.js)、[创建页提示](https://aihuanxin.cn/kunlun/kl-web/js/chunk-0d448551.cb23609b.js)。

## 6. 本次实际做到哪一步

已完成：公开平台和帮助核查、配置核查、环境与传输脚本；用户补齐 cm-model 七个包并回传 v1.1 的 10 万条真实训练及两端内容核对通过结果。本地完成 v1 表示诊断、v2 有限解析、权重和来源留出；v2 共 4 组主试验与 16 次留出拟合，另 2 次因训练缺类报告 N/A。最新证据见 [云端核对](pilot_content_verification_cloud_user.json) 与 [v2 复核](../docs/BEHAVIOR_V2_REVIEW.md)。未完成：可信的完整类别覆盖、完整行为解析、独立外部验证、最终模型验收、助手实时登录交互和 NPU 验证。全局依赖仍有昇腾工具问题。云端由用户执行；助手没有直接操作实例，也没有执行云端 v2。计费或额度消耗以平台账单为准。
