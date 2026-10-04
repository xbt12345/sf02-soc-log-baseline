# V169 平台执行包与源码

2026-10-02 已交付 Linux aarch64 CPU 离线包；截至 2026-10-04，平台实际执行结果仍待回传。最新实际训练 V164、最新完整质量交付 V159，完整质量未通过。

`execution_source/` 保留已交付包中 `run_platform.py`、`PLATFORM_README.md` 与 `platform_runtime/` 的原字节副本，包括安装、数值资格、完整训练入口、结果回收和包内清单。这是已交付源码的存档，清单所列大型数据、模型、数组和 vendor 文件没有全部存入 Git。请对完整执行包使用下面的命令；仅克隆仓库或进入 `execution_source/` 运行不会恢复这些缺件。

完整执行包名为 `SF02_V169_CPU_platform_v1.zip`，548,455,564 字节，1,598 个文件；其 SHA-256、原始回执和独立核验见 [当前进度](../../docs/CURRENT_PROGRESS.md)。本地保留位置为 `C:\Users\xiabutian\Downloads\SF02_V169_Platform_20261002\`。

将完整 ZIP 上传到平台 `/root/work` 后执行：

```bash
cd /root/work
python3 -m zipfile -e SF02_V169_CPU_platform_v1.zip sf02_v169_run
nohup python3 sf02_v169_run/SF02/run_platform.py > sf02_v169_run/console.log 2>&1 &
```

查看进度：

```bash
tail -f /root/work/sf02_v169_run/console.log
```

启动器先核验包内容，再离线安装私有运行时，执行 CPU/内存/磁盘及完整数值资格检查，重新绑定真实平台源码、数据与依赖，再进入原科学预算内的三角色双臂训练。平台要求 Linux aarch64、glibc 至少 2.28；这些要求不等于任意 Codex Cloud 环境已经兼容。不得降低门槛或重复启动来追认成功。

结束或报错后回传 `sf02_v169_run/SF02_V169_CPU_review_v1.zip`；完整 `SF02_V169_CPU_result_v1.zip` 保留在平台。启动器正常返回只说明有界入口结束，分类质量仍须独立审核完整逐行结果。
