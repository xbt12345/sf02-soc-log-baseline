# v3.3.1 平台更新包

只上传外层 sf02_v331_update.py。复用 /root/work/sf02_data/train.parquet 和 /root/work/sf02_v32_round/prepared。
运行：conda run --no-capture-output -n cm-model python -u /root/work/sf02_v331_update.py

CPU 模式，最多 16 个库线程；不安装包、不使用昇腾卡、不修改官方标签。
程序自行校验、准备、分组、执行 9 个任务共 27 次拟合、复算结果并生成一个 ZIP。
再次运行同一命令会核验并复用完整任务；未完成任务重新执行，不覆盖旧尝试。
程序输出 RETURN_THIS_ZIP 后下载它；出错时优先带回 RETURN_FAILURE_ZIP。
自动化执行完成不等于迁移效果通过，需要审查返回结果。
