# V169后台清理完成，物理RAM仍不足，等待真实资源变化

主聊天已完成用户授权的后台清理，并要求本线程保持原封存、预算和学习因子，不启动训练、不重复正式入口或资源诊断、不下调门槛；等待真实资源变化。用户明确要求保留Edge，已保留。

根记录39次直接停止动作，父进程退出也结束部分子进程。对象为确认不用的MCP文件/内存/Playwright、Codex Security工具后台、Windows Widgets/Phone Link、Java更新、Intel DSA/图形设置托盘、SOLIDWORKS后台下载及NVIDIA浮层；活动Codex、CUA/网络代理、系统驱动和安全组件保留，启动配置未修改。三个可选系统服务Stop-Service因权限被拒，部分工具/浮层辅助进程被宿主拉起。这不是全部后台均已清除的证明。

清理期间物理可用RAM约3.4–4.0GiB，仍不足固定5.25GiB。21:46:18的GlobalMemoryStatusEx实测物理3990192128、提交余量11996278784bytes，磁盘20173533184bytes：物理不通过，提交和磁盘通过。根同时报告GPU可用7956MiB；这些是OS/驱动快照，不是实际入口通过或CUDA进程内零步资格。

根最终CIM快照（2026-10-02T21:49:32.9573450+08:00）物理可用3749335040bytes（3.492GiB），距固定门槛缺1887809536bytes（1.758GiB）。快照不能预测未来就绪；真实变化后仍必须核完整源和正式入口全部即时门槛。

恢复条件不变：物理可用RAM>=5637144576bytes、可用commit>=6710886400bytes、GPU>=536870912bytes、完整轮起始磁盘>=14283971948bytes，并保持稳定。只有真实恢复后按既有授权核源/即时门槛并继续同一sealed入口；本次等待不重新sealer、不改任何正式训练源或科学约束、不追加fit预算。起始0后台/0调用资源失败不算正式fit；进入过正式fit后的失败仍遵守原全局停止约束。

V169目录仍仅registration.json和run_seal.json，没有role/backend；两次原始入口失败、纠正的同进程诊断和清理记录全部保留。新增官方heads/features/derivatives/fits/updates均0。历史19178头/768完整导数、V159以来9fits/170更新不变。最新实际训练V164、完整质量交付V159；三个总目标、完整三分类与独立来源泛化均未完成，无模型晋升。

## 清理原始证据 artifacts/v169_background_cleanup_20261002/cleanup.json

SHA256 `0bffc6f8b6fa9ac1fc6d955f1792179bccb6b19e1621dcc7fab658af97b60954`

{
  "timestamp": "2026-10-02T21:39:12.3275610+08:00",
  "scope": "Verified unused MCP filesystem/memory/playwright trees, Windows Widgets, Phone Link background, NVIDIA overlay only",
  "before_free_physical_bytes": 3996131328,
  "before_free_virtual_bytes": 8986144768,
  "after_free_physical_bytes": 4195188736,
  "after_free_virtual_bytes": 11803594752,
  "stopped": [
    {
      "pid": 38300,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T21:36:40.749004+08:00",
      "rss_bytes": 62902272,
      "private_bytes": 61665280,
      "result": "stopped"
    },
    {
      "pid": 35544,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T21:36:40.607344+08:00",
      "rss_bytes": 97394688,
      "private_bytes": 62386176,
      "result": "stopped"
    },
    {
      "pid": 40320,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T21:36:40.508806+08:00",
      "rss_bytes": 63471616,
      "private_bytes": 62218240,
      "result": "stopped"
    },
    {
      "pid": 7244,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T21:06:01.862997+08:00",
      "rss_bytes": 62410752,
      "private_bytes": 61337600,
      "result": "stopped"
    },
    {
      "pid": 4452,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T21:06:01.504471+08:00",
      "rss_bytes": 63311872,
      "private_bytes": 61857792,
      "result": "stopped"
    },
    {
      "pid": 39476,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T21:06:01.431183+08:00",
      "rss_bytes": 97222656,
      "private_bytes": 62320640,
      "result": "stopped"
    },
    {
      "pid": 21416,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:52:31.015331+08:00",
      "rss_bytes": 20561920,
      "private_bytes": 62099456,
      "result": "stopped"
    },
    {
      "pid": 32940,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:52:30.257035+08:00",
      "rss_bytes": 20672512,
      "private_bytes": 60633088,
      "result": "stopped"
    },
    {
      "pid": 36316,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:52:30.151103+08:00",
      "rss_bytes": 21954560,
      "private_bytes": 62001152,
      "result": "stopped"
    },
    {
      "pid": 35876,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:52:07.30031+08:00",
      "rss_bytes": 19030016,
      "private_bytes": 61239296,
      "result": "stopped"
    },
    {
      "pid": 35888,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:52:05.79477+08:00",
      "rss_bytes": 19066880,
      "private_bytes": 62640128,
      "result": "stopped"
    },
    {
      "pid": 36252,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:52:04.400412+08:00",
      "rss_bytes": 19759104,
      "private_bytes": 62427136,
      "result": "stopped"
    },
    {
      "pid": 34132,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:49.027647+08:00",
      "rss_bytes": 19562496,
      "private_bytes": 61997056,
      "result": "stopped"
    },
    {
      "pid": 22780,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:48.945368+08:00",
      "rss_bytes": 19185664,
      "private_bytes": 62636032,
      "result": "stopped"
    },
    {
      "pid": 15440,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:48.912581+08:00",
      "rss_bytes": 19546112,
      "private_bytes": 60981248,
      "result": "stopped"
    },
    {
      "pid": 21332,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:42.544967+08:00",
      "rss_bytes": 19353600,
      "private_bytes": 62300160,
      "result": "stopped"
    },
    {
      "pid": 31988,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:42.059203+08:00",
      "rss_bytes": 19689472,
      "private_bytes": 61796352,
      "result": "stopped"
    },
    {
      "pid": 30788,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:41.639206+08:00",
      "rss_bytes": 20594688,
      "private_bytes": 61071360,
      "result": "stopped"
    },
    {
      "pid": 26648,
      "name": "node.exe",
      "path": "C:\\nvm4w\\nodejs\\node.exe",
      "creation_time": "2026-10-02T20:51:37.395142+08:00",
      "rss_bytes": 28053504,
      "private_bytes": 89227264,
      "result": "stopped"
    },
    {
      "pid": 26072,
      "name": "PhoneExperienceHost.exe",
      "path": "C:\\Program Files\\WindowsApps\\Microsoft.YourPhone_1.26072.257.0_x64__8wekyb3d8bbwe\\PhoneExperienceHost.exe",
      "creation_time": "2026-10-02T20:50:40.751266+08:00",
      "rss_bytes": 93863936,
      "private_bytes": 81346560,
      "result": "stopped"
    },
    {
      "pid": 25052,
      "name": "NVIDIA Overlay.exe",
      "path": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\CEF\\NVIDIA Overlay.exe",
      "creation_time": "2026-10-02T20:50:35.135388+08:00",
      "rss_bytes": 26513408,
      "private_bytes": 52940800,
      "result": "stopped"
    },
    {
      "pid": 24716,
      "name": "NVIDIA Overlay.exe",
      "path": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\CEF\\NVIDIA Overlay.exe",
      "creation_time": "2026-10-02T20:50:34.860496+08:00",
      "rss_bytes": 7163904,
      "private_bytes": 10170368,
      "result": "stopped"
    },
    {
      "pid": 24580,
      "name": "NVIDIA Overlay.exe",
      "path": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\CEF\\NVIDIA Overlay.exe",
      "creation_time": "2026-10-02T20:50:34.793975+08:00",
      "rss_bytes": 17911808,
      "private_bytes": 12488704,
      "result": "stopped"
    },
    {
      "pid": 23248,
      "name": "NVIDIA Overlay.exe",
      "path": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\CEF\\NVIDIA Overlay.exe",
      "creation_time": "2026-10-02T20:50:34.781172+08:00",
      "rss_bytes": 14721024,
      "private_bytes": 138977280,
      "result": "stopped"
    },
    {
      "pid": 21312,
      "name": "NVIDIA Overlay.exe",
      "path": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\CEF\\NVIDIA Overlay.exe",
      "creation_time": "2026-10-02T20:50:34.096602+08:00",
      "rss_bytes": 75104256,
      "private_bytes": 126898176,
      "result": "stopped"
    },
    {
      "pid": 19372,
      "name": "Widgets.exe",
      "path": "C:\\Program Files\\WindowsApps\\MicrosoftWindows.Client.WebExperience_526.21100.40.0_x64__cw5n1h2txyewy\\Dashboard\\Widgets.exe",
      "creation_time": "2026-10-02T20:50:29.131894+08:00",
      "rss_bytes": 42205184,
      "private_bytes": 9428992,
      "result": "stopped"
    }
  ],
  "skipped": [],
  "original_pids_still_present": [],
  "original_frequency_or_model_modified": false
}


## 清理原始证据 artifacts/v169_background_cleanup_20261002/cleanup_stage2.json

SHA256 `e02e68987aa45ee607ccd69446c2b45ae58796903db19ec1c50a6fcb87cb4f94`

{
  "timestamp": "2026-10-02T21:41:45.7828263+08:00",
  "actions": [
    {
      "PID": 32988,
      "Name": "node.exe",
      "Kind": "Unused Codex Security plugin MCP",
      "Result": "stopped"
    },
    {
      "PID": 31180,
      "Name": "node.exe",
      "Kind": "Unused Codex Security plugin MCP",
      "Result": "stopped"
    },
    {
      "PID": 29180,
      "Name": "node.exe",
      "Kind": "Unused Codex Security plugin MCP",
      "Result": "stopped"
    },
    {
      "PID": 27276,
      "Name": "node.exe",
      "Kind": "Unused Codex Security plugin MCP",
      "Result": "stopped"
    },
    {
      "PID": 26576,
      "Name": "node.exe",
      "Kind": "Unused Codex Security plugin MCP",
      "Result": "stopped"
    },
    {
      "PID": 27948,
      "Name": "node.exe",
      "Kind": "Unused Codex Security plugin MCP",
      "Result": "stopped"
    },
    {
      "Name": "DSAUpdateService",
      "Kind": "Optional service",
      "Result": "Service 'Intel(R) Driver & Support Assistant Updater (DSAUpdateService)' cannot be stopped due to the following error: Cannot open 'DSAUpdateService' service on computer '.'."
    },
    {
      "Name": "HpTouchpointAnalyticsService",
      "Kind": "Optional service",
      "Result": "Service 'HP Insights Analytics (HpTouchpointAnalyticsService)' cannot be stopped due to the following error: Cannot open 'HpTouchpointAnalyticsService' service on computer '.'."
    },
    {
      "Name": "PCManager Service Store",
      "Kind": "Optional service",
      "Result": "Service 'Microsoft PC Manager Service (PCManager Service Store)' cannot be stopped due to the following error: Cannot open 'PCManager Service Store' service on computer '.'."
    }
  ],
  "free_physical_bytes": 4314046464,
  "free_virtual_bytes": 12393496576,
  "startup_settings_changed": false
}


## 清理原始证据 artifacts/v169_background_cleanup_20261002/cleanup_stage3.json

SHA256 `82a16dd02340ec4c9b34dcec21f1d9ac73c45c37f85e5fac36043c2dd26a871d`

{
  "timestamp": "2026-10-02T21:44:09.4100375+08:00",
  "actions": [
    {
      "PID": 11248,
      "Name": "DSATray.exe",
      "Path": "C:\\Program Files (x86)\\Intel\\Driver and Support Assistant\\x86\\DSATray.exe",
      "Result": "stopped"
    },
    {
      "PID": 7408,
      "Name": "sldBgDwld.exe",
      "Path": "C:\\Program Files (x86)\\Common Files\\SOLIDWORKS 安装管理程序\\BackgroundDownloading\\sldBgDwld.exe",
      "Result": "stopped"
    },
    {
      "PID": 3860,
      "Name": "jusched.exe",
      "Path": "C:\\Program Files (x86)\\Common Files\\Java\\Java Update\\jusched.exe",
      "Result": "stopped"
    },
    {
      "PID": 37928,
      "Name": "IntelGraphicsSoftware.exe",
      "Path": "C:\\Program Files\\Intel\\Intel Graphics Software\\IntelGraphicsSoftware.exe",
      "Result": "stopped"
    },
    {
      "PID": 18752,
      "Name": "jucheck.exe",
      "Path": "C:\\Program Files (x86)\\Common Files\\Java\\Java Update\\jucheck.exe",
      "Result": "stopped"
    }
  ],
  "free_physical_bytes": 4199469056,
  "free_virtual_bytes": 12400672768,
  "edge_preserved_at_user_request": true
}


## 清理原始证据 artifacts/v169_background_cleanup_20261002/cleanup_stage4_overlay.json

SHA256 `9d45c9bdf2a423205f18c9784f26fc692a8a92af5114d47ca39866b098e8f772`

{
  "timestamp": "2026-10-02T21:48:35.5685237+08:00",
  "actions": [
    {
      "PID": 19232,
      "Name": "nvcontainer.exe",
      "Path": "C:\\Program Files\\NVIDIA Corporation\\NvContainer\\nvcontainer.exe",
      "Result": "stopped"
    },
    {
      "PID": 38968,
      "Name": "NVIDIA Overlay.exe",
      "Path": "C:\\Program Files\\NVIDIA Corporation\\NVIDIA App\\CEF\\NVIDIA Overlay.exe",
      "Result": "stopped"
    }
  ],
  "remaining": [
    {
      "ProcessId": 34108,
      "Name": "nvcontainer.exe"
    }
  ],
  "driver_and_system_containers_preserved": true,
  "free_physical_bytes": 3681771520,
  "free_virtual_bytes": 10107944960
}


## 清理原始证据 artifacts/v169_background_cleanup_20261002/post_cleanup_resources.json

SHA256 `fe5cda0f5f65cfdf8705dff71ff4a78115e19f24158d2da816a72a6a5689d3ad`

{
  "timestamp": "2026-10-02T21:46:18.4569168+08:00",
  "physical_free_bytes": 3990192128,
  "physical_total_bytes": 16858472448,
  "available_commit_bytes": 11996278784,
  "commit_limit_bytes": 32910151680,
  "physical_required_bytes": 5637144576,
  "commit_required_bytes": 6710886400,
  "disk_free_bytes": 20173533184,
  "disk_required_bytes": 14283971948,
  "physical_gate_passed": false,
  "commit_gate_passed": true,
  "disk_gate_passed": true,
  "edge_preserved": true,
  "edge_process_count": 16,
  "edge_rss_sum_bytes": 1713340416.0,
  "unconfirmed_or_active_cua_codex_proxy_and_phone_preserved": true,
  "system_driver_antivirus_and_network_services_preserved": true,
  "three_optional_service_stop_attempts_denied": true,
  "startup_configuration_modified": false,
  "training_model_calls_during_cleanup": 0,
  "original_optional_apps_remaining": [
    {
      "ProcessId": 38968,
      "Name": "NVIDIA Overlay.exe"
    },
    {
      "ProcessId": 24412,
      "Name": "NVIDIA Overlay.exe"
    },
    {
      "ProcessId": 10260,
      "Name": "NVIDIA Overlay.exe"
    },
    {
      "ProcessId": 3020,
      "Name": "NVIDIA Overlay.exe"
    },
    {
      "ProcessId": 36268,
      "Name": "NVIDIA Overlay.exe"
    }
  ],
  "scope": "Actual OS resource snapshot after safe process cleanup; no actual in-runtime/CUDA qualification and no training executed"
}


## 清理原始证据 artifacts/v169_background_cleanup_20261002/summary.json

SHA256 `0d5d9929c711ad759ea8321e07e47360c8162c879ffc827f856edaf19b2c556e`

{
  "timestamp": "2026-10-02T21:49:32.9573450+08:00",
  "direct_stop_actions": 39,
  "failed_actions": [
    {
      "Name": "DSAUpdateService",
      "Kind": "Optional service",
      "Result": "Service 'Intel(R) Driver & Support Assistant Updater (DSAUpdateService)' cannot be stopped due to the following error: Cannot open 'DSAUpdateService' service on computer '.'."
    },
    {
      "Name": "HpTouchpointAnalyticsService",
      "Kind": "Optional service",
      "Result": "Service 'HP Insights Analytics (HpTouchpointAnalyticsService)' cannot be stopped due to the following error: Cannot open 'HpTouchpointAnalyticsService' service on computer '.'."
    },
    {
      "Name": "PCManager Service Store",
      "Kind": "Optional service",
      "Result": "Service 'Microsoft PC Manager Service (PCManager Service Store)' cannot be stopped due to the following error: Cannot open 'PCManager Service Store' service on computer '.'."
    }
  ],
  "edge_preserved_by_explicit_reply": true,
  "essential_os_driver_security_proxy_and_active_codex_preserved": true,
  "startup_configuration_modified": false,
  "final_cim_free_physical_bytes": 3749335040,
  "final_cim_free_virtual_bytes": 10231443456,
  "physical_required_bytes": 5637144576,
  "physical_gate_passed": false,
  "training_started": false,
  "training_model_calls": 0,
  "known_overlay_or_helper_respawns": [
    {
      "ProcessId": 34108,
      "Name": "nvcontainer.exe"
    }
  ],
  "note": "Only verified nonessential background processes were terminated. Parent exits also ended child processes. Some helper processes restart automatically; no permanent startup/driver/security changes. The physical RAM prerequisite still fails."
}


## Previous complete direction and all inherited scientific constraints

# V169当前实际停止：真实RAM和commit不足，未进入任何训练

同一已封存v14入口的首次启动及retry1均在initial检查拒绝，未建role/backend、未完成真实零步；0新官方头/特征/导数/fit/更新。不能将registration、进程启动或0调用资源观察算成训练实验。

根授权一次同进程实际require(initial)观察。首版广域trace因观察器开销过大，经核实PID归属后只停止该诊断和包装进程，原source/console及终止凭据保留。纠正版按v2/v5函数code对象直接过滤，非目标frame返回None，一次require在54.7秒内完成，满足180秒上限。没有改变任何gate返回值、系统状态、阈值、模型/训练源或预算。

实际v2前CUDA读取瞬间RAM3294277632<5637144576，commit4405325824<6710886400bytes；诊断开始时两项已不足。完整封存哈希及继承基础契约核验后，进程常驻仅增加7159808bytes、私有commit增加7630848bytes。没有支持继续降低门槛、修复preCUDA检查器或第三次main盲重试的证据；后续v3/v5政策检查没有因前段哈希完成而自动通过。

CUDA初始化后保存时点RAM还缺2433847296bytes（约2.27GiB）、commit还缺2483826688bytes（约2.31GiB）。这些是保存时点，不是未来 readiness，也不重构retry1原失败瞬间。只读应用私有工作集观察不授予关闭用户应用或未归属node服务的权限；SF02/v169可归属node/python helper复查为0。没有用户进程被关闭或trim。

根要求现在停止新增训练、资源诊断和同构资格准备，等待人或外部明确释放足够RAM与commit并保持稳定。恢复信号后按已有授权核实完整源和即时RAM5.25GiB/commit6.25GiB/GPU512MiB/完整磁盘门槛，再继续同一sealed、0后台/0调用入口；起始资源失败不追加fit，进入正式fit后的失败仍严格执行原全局停止约束。原封存、数据/源码、失败日志全保留。

历史19178头/768完整导数、V159以来9fits/170更新不变；最新实际训练V164、完整交付V159，三个总目标、完整三分类与独立来源质量均未完成，无模型晋升。


## 实际原始来源 artifacts/v169_same_process_initial_resource_diagnostic_v2_20261002/review.json；SHA256 `84ea115750f1c9e65a69303e36415f9d29e22d99fd6b9dca59f5f381bcdbf16c`

{
  "status": "V169_one_actual_same_process_resource_gate_code_filtered_observer_completed",
  "actual_require_calls": 1,
  "diagnostic_time_limit_seconds": 180,
  "elapsed_seconds": 54.72658309999997,
  "baseline": {
    "timestamp_UTC": "2026-10-02T13:13:02.294823+00:00",
    "process": {
      "peak_working": 570716160,
      "working": 568705024,
      "pagefile": 2475053056,
      "peak_pagefile": 2477170688
    },
    "system": {
      "available_physical_bytes": 2917183488,
      "available_commit_bytes": 4190146560,
      "total_physical_bytes": 16858472448,
      "total_commit_bytes": 32910151680
    },
    "CUDA_initialized": false
  },
  "events": [
    {
      "source": "training/v169_pair_execution_review_v5.py",
      "event": "call",
      "line_number": 34,
      "source_line": "def require_run_seal(path,current_trainer,phase='running'):",
      "elapsed_seconds": 0.0031596000001172797,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:02.298824+00:00",
        "process": {
          "peak_working": 570716160,
          "working": 569118720,
          "pagefile": 2475548672,
          "peak_pagefile": 2477170688
        },
        "system": {
          "available_physical_bytes": 2916528128,
          "available_commit_bytes": 4190298112,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      }
    },
    {
      "source": "training/v169_pair_execution_review_v2.py",
      "event": "call",
      "line_number": 9,
      "source_line": "def require_run_seal(path,current_trainer,phase='running'):",
      "elapsed_seconds": 0.003271000000040658,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:02.298824+00:00",
        "process": {
          "peak_working": 570716160,
          "working": 569122816,
          "pagefile": 2475548672,
          "peak_pagefile": 2477170688
        },
        "system": {
          "available_physical_bytes": 2916540416,
          "available_commit_bytes": 4190298112,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      }
    },
    {
      "source": "training/v169_pair_execution_review_v2.py",
      "event": "line",
      "line_number": 12,
      "source_line": "if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise RuntimeError('Actual physical RAM unavailable')",
      "elapsed_seconds": 54.53805639999996,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
        "process": {
          "peak_working": 580816896,
          "working": 575864832,
          "pagefile": 2482683904,
          "peak_pagefile": 2487287808
        },
        "system": {
          "available_physical_bytes": 3294277632,
          "available_commit_bytes": 4405325824,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      },
      "actual_gate_state": {
        "length": 64,
        "available_physical_bytes": 0,
        "available_commit_bytes": 0
      }
    },
    {
      "source": "training/v169_pair_execution_review_v2.py",
      "event": "line",
      "line_number": 13,
      "source_line": "if state.avail_phys<plan['resources']['minimum_free_RAM_bytes']:raise RuntimeError('Registered physical RAM prerequisite not met')",
      "elapsed_seconds": 54.53811420000011,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
        "process": {
          "peak_working": 580816896,
          "working": 575864832,
          "pagefile": 2482683904,
          "peak_pagefile": 2487287808
        },
        "system": {
          "available_physical_bytes": 3294277632,
          "available_commit_bytes": 4405325824,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      },
      "actual_gate_state": {
        "length": 64,
        "available_physical_bytes": 3294277632,
        "available_commit_bytes": 4405325824
      }
    },
    {
      "source": "training/v169_pair_execution_review_v2.py",
      "event": "exception",
      "line_number": 13,
      "source_line": "if state.avail_phys<plan['resources']['minimum_free_RAM_bytes']:raise RuntimeError('Registered physical RAM prerequisite not met')",
      "elapsed_seconds": 54.53813050000008,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
        "process": {
          "peak_working": 580816896,
          "working": 575864832,
          "pagefile": 2482683904,
          "peak_pagefile": 2487287808
        },
        "system": {
          "available_physical_bytes": 3294277632,
          "available_commit_bytes": 4405325824,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      },
      "actual_gate_state": {
        "length": 64,
        "available_physical_bytes": 3294277632,
        "available_commit_bytes": 4405325824
      },
      "exception": {
        "type": "RuntimeError",
        "message": "Registered physical RAM prerequisite not met"
      }
    },
    {
      "source": "training/v169_pair_execution_review_v2.py",
      "event": "return",
      "line_number": 13,
      "source_line": "if state.avail_phys<plan['resources']['minimum_free_RAM_bytes']:raise RuntimeError('Registered physical RAM prerequisite not met')",
      "elapsed_seconds": 54.53814239999997,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
        "process": {
          "peak_working": 580816896,
          "working": 575864832,
          "pagefile": 2482683904,
          "peak_pagefile": 2487287808
        },
        "system": {
          "available_physical_bytes": 3294277632,
          "available_commit_bytes": 4405325824,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      },
      "actual_gate_state": {
        "length": 64,
        "available_physical_bytes": 3294277632,
        "available_commit_bytes": 4405325824
      }
    },
    {
      "source": "training/v169_pair_execution_review_v5.py",
      "event": "exception",
      "line_number": 35,
      "source_line": "plan=prior_require(path,current_trainer,phase)",
      "elapsed_seconds": 54.538152999999966,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
        "process": {
          "peak_working": 580816896,
          "working": 575864832,
          "pagefile": 2482683904,
          "peak_pagefile": 2487287808
        },
        "system": {
          "available_physical_bytes": 3294277632,
          "available_commit_bytes": 4405325824,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      },
      "exception": {
        "type": "RuntimeError",
        "message": "Registered physical RAM prerequisite not met"
      }
    },
    {
      "source": "training/v169_pair_execution_review_v5.py",
      "event": "return",
      "line_number": 35,
      "source_line": "plan=prior_require(path,current_trainer,phase)",
      "elapsed_seconds": 54.538161500000115,
      "observed": {
        "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
        "process": {
          "peak_working": 580816896,
          "working": 575864832,
          "pagefile": 2482683904,
          "peak_pagefile": 2487287808
        },
        "system": {
          "available_physical_bytes": 3294277632,
          "available_commit_bytes": 4405325824,
          "total_physical_bytes": 16858472448,
          "total_commit_bytes": 32910151680
        },
        "CUDA_initialized": false
      }
    }
  ],
  "outcome": {
    "status": "actual_same_process_initial_require_refused",
    "type": "RuntimeError",
    "message": "Registered physical RAM prerequisite not met",
    "execution_started": false
  },
  "after_require": {
    "timestamp_UTC": "2026-10-02T13:13:56.833361+00:00",
    "process": {
      "peak_working": 580816896,
      "working": 575864832,
      "pagefile": 2482683904,
      "peak_pagefile": 2487287808
    },
    "system": {
      "available_physical_bytes": 3294277632,
      "available_commit_bytes": 4405325824,
      "total_physical_bytes": 16858472448,
      "total_commit_bytes": 32910151680
    },
    "CUDA_initialized": false
  },
  "after_gc": {
    "timestamp_UTC": "2026-10-02T13:13:56.883804+00:00",
    "process": {
      "peak_working": 580816896,
      "working": 575864832,
      "pagefile": 2482683904,
      "peak_pagefile": 2487287808
    },
    "system": {
      "available_physical_bytes": 3294412800,
      "available_commit_bytes": 4405321728,
      "total_physical_bytes": 16858472448,
      "total_commit_bytes": 32910151680
    },
    "CUDA_initialized": false
  },
  "after_CUDA_initialization_metadata_only": {
    "timestamp_UTC": "2026-10-02T13:13:57.015390+00:00",
    "process": {
      "peak_working": 668745728,
      "working": 668745728,
      "pagefile": 2666917888,
      "peak_pagefile": 2666921984
    },
    "system": {
      "available_physical_bytes": 3203297280,
      "available_commit_bytes": 4227059712,
      "total_physical_bytes": 16858472448,
      "total_commit_bytes": 32910151680
    },
    "CUDA_initialized": true
  },
  "after_CUDA_GPU_metadata": {
    "free_bytes": 7451181056,
    "total_bytes": 8585216000
  },
  "observer_first_condition_is_code_object_set_membership": true,
  "non_target_frames_return_None": true,
  "no_gate_return_or_state_or_threshold_replaced": true,
  "no_system_or_user_process_trim": true,
  "main_and_backend_not_called": true,
  "registered_files_unchanged": true,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "execution_authority": false,
  "source_sha256": {
    "training/v169_same_process_initial_resource_diagnostic_v2.py": "0c98e12c61dda6790d1a147e9902d2c649bb77c08a52a9e94855cc2d7e97f8ac",
    "training/v169_prior_pair_training_entry_v14.py": "5c2cf876ced21ce56cf85f4147d103188335b2dedd1aa1d2561a3827fd968877",
    "training/v169_pair_execution_review_v2.py": "eb601c14d3c3717415664de96feb1c5c7930b3d68c1e132729c2bfcb99476a2b",
    "training/v169_pair_execution_review_v5.py": "940e2eb33804266bbca28b9948b84b951cb88dcb7c760679f246c4062778a458",
    "training/review_policy/v169_prior_pair_execution_contract.json": "ea97901399d7ef548a434b4b7b461d0d326d3f002fa1b012897406306f140088",
    "artifacts/v169_prior_pair_training/registration.json": "23479670c2cef2430d8a3fcfa3afec74f61b0f9815da5083a1f946a059dfe03f",
    "artifacts/v169_prior_pair_training/run_seal.json": "e96cac5fbbb59ba18c59526b6ef6ae05548d500eaf5fe30e401228bc0101b633",
    "artifacts/v169_prior_pair_training_entry_v14_original_console_20261002.txt": "61bbec3c738df6197ef77e64721474f26e4b9be4e114c87b5e74a3549aea0bd8",
    "artifacts/v169_prior_pair_training_entry_v14_retry1_original_console_20261002.txt": "61bbec3c738df6197ef77e64721474f26e4b9be4e114c87b5e74a3549aea0bd8",
    "training/v169_same_process_initial_resource_diagnostic.py": "0dc93e481d0e905cc6d47a1dc94809302a2e93a3f4163cb09f5cd6d9d8ed2cb4",
    "artifacts/v169_same_process_initial_resource_diagnostic_20261002/termination.json": "3c3e65e3e10ffad738f3b44f4fa3e72e2e89df81b872c0c4d8ad9d385cdb5f1e"
  }
}

## 实际原始来源 artifacts/v169_same_process_initial_resource_diagnostic_v2_20261002/decision.json；SHA256 `e75f744619cef84f00518f79d0884c3df516fe6453c5afd52d8428c1f9c6a4f7`

{
  "status": "actual_same_process_gate_refusal_resource_wait_supported_no_checker_revision_supported",
  "require_elapsed_to_real_RAM_read_seconds": 54.53811420000011,
  "diagnostic_elapsed_seconds": 54.72658309999997,
  "checker_resident_increment_bytes": 7159808,
  "checker_private_commit_increment_bytes": 7630848,
  "checker_peak_resident_increment_over_baseline_working_bytes": 12111872,
  "before_checker_already_below_RAM_and_commit": true,
  "actual_refusal_instant": {
    "length": 64,
    "available_physical_bytes": 3294277632,
    "available_commit_bytes": 4405325824
  },
  "actual_refusal_RAM_shortfall_bytes": 2342866944,
  "actual_refusal_commit_shortfall_bytes": 2305560576,
  "after_CUDA_RAM_shortfall_bytes": 2433847296,
  "after_CUDA_commit_shortfall_bytes": 2483826688,
  "prior_simple_preview_does_not_prove_later_startup_readiness": true,
  "no_claim_this_diagnostic_reconstructs_the_earlier_retry1_failure_instant": true,
  "no_claim_all_system_cache_changes_are_unrelated_to_IO": true,
  "no_main_retry_without_new_actual_resource_change": true,
  "no_source_or_threshold_changed": true,
  "no_task_node_or_python_helper_remained_after_diagnostic": true,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "execution_authority": false,
  "source_sha256": {
    "artifacts/v169_same_process_initial_resource_diagnostic_v2_20261002/review.json": "84ea115750f1c9e65a69303e36415f9d29e22d99fd6b9dca59f5f381bcdbf16c",
    "artifacts/v169_same_process_initial_resource_diagnostic_v2_20261002/process_memory_snapshot.json": "58d5a1162018a9936797acc381b24170b0e0525bb497bf6adca54cc105c7d88c",
    "training/review_policy/v169_prior_pair_execution_contract.json": "ea97901399d7ef548a434b4b7b461d0d326d3f002fa1b012897406306f140088",
    "artifacts/v169_same_process_initial_resource_diagnostic_20261002/termination.json": "3c3e65e3e10ffad738f3b44f4fa3e72e2e89df81b872c0c4d8ad9d385cdb5f1e"
  }
}


## Previous complete direction and all inherited scientific constraints

# V169当前实际：封存成功，首次启动在RAM检查拒绝

根新最终独审通过18967项/35194537679bytes/93递归模块/18资格；v14只更换资源政策import，原数据、完整参数、逐类目标、20更新日程、预算、保护、数值与质量门槛全保持。资源v9原环境BLAS24/完整触页/原生SVD生命周期证据经根独立复算，资源v10与policy_v5登记物理RAM5.25GiB、可用commit6.25GiB，其余资源不变。原6GiB封存拒绝和BLAS4资格的混杂记录保留。

sealer_v5实际exit0，registration绑定18971项。封存时RAM5874417664、commit10196668416、GPU7451181056、磁盘15053062144bytes过门槛。但随后v14第一次启动实际exit1，main首句require(initial)经v2的前CUDA物理RAM检查拒绝，未进入configure/ActualBackend、未建角色目录、未执行零步/官方前向/导数/拟合/更新。正式目录仅registration.json和run_seal.json；不能将registration或进程启动称为训练完成。

失败后CUDA初始化快照为2026-10-02T07:53:22.194167+00:00：RAM4573757440，须5637144576，缺1063387136bytes；commit/GPU/磁盘通过。这是失败后的保存时点，不是故障瞬间值，也不能预测可用时刻。根允许0调用且未建角色目录的同一sealed入口在新真实全资源过门槛时启动重试，不追加fit；不得降低资源/科学门槛、改已封存源码或重复sealer。失败日志/封存全保留。已核查当前python/pythonw进程，未找到带SF02/v169命令的驻留helper，无对象被关闭。

累计19178头/768完整导数，自V159以来9fits/170更新不变；最新实际训练V164、最新完整交付V159。当前仍解决第一问题，真实零步/配对训练/完整原行质量均未完成；三个总目标未完成，无模型晋升。


## 实际原始来源 artifacts/v169_root_final_preseal_review_v2_20261002/review.json；SHA256 `baf3911b2d60bbba86201b7fe90fb663ba385930f14f632579b7461be4f7603b`

{
  "status": "V169_independent_resource_only_final_source_and_resource_design_review_passed",
  "reviewed_entry_sha256": "5c2cf876ced21ce56cf85f4147d103188335b2dedd1aa1d2561a3827fd968877",
  "supports_physical_seal": true,
  "execution_authority": false,
  "new_fit_permission": false,
  "actual_resource_gates_must_still_pass_before_seal_and_each_fit": true,
  "official_zero_step_and_all_registered_real_quality_gates_still_required": true,
  "hash_verified_files": 18967,
  "bytes_hashed": 35194537679,
  "recursive_project_execution_and_sealing_modules": 93,
  "qualifications": [
    "artifacts/v169_pair_model_storage_qualification_20261002/qualification.json",
    "artifacts/v169_pair_lifecycle_qualification_v2_20261002/qualification.json",
    "artifacts/v169_saved_state_quality_qualification_20261002/qualification.json",
    "artifacts/v169_working_solver_full_size_qualification_20261002/qualification.json",
    "artifacts/v169_factorized_storage_qualification_v2_20261002/qualification.json",
    "artifacts/v169_measurement_references_qualification_20261002/qualification.json",
    "artifacts/v169_dataframe_and_current_guard_qualification_20261002/qualification.json",
    "artifacts/v169_actual_backend_synthetic_qualification_v4_20261002/qualification.json",
    "artifacts/v169_saved_correction_storage_qualification_20261002/qualification.json",
    "artifacts/v169_execution_contract_candidate_v3_20261002/qualification.json",
    "artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json",
    "artifacts/v169_cuda_host_buffer_qualification_v2_20261002/qualification.json",
    "artifacts/v169_physical_and_commit_policy_qualification_20261002/qualification.json",
    "artifacts/v169_execution_contract_candidate_v4_20261002/qualification.json",
    "artifacts/v169_resource_policy_v5_qualification_20261002/qualification.json",
    "artifacts/v169_actual_backend_synthetic_qualification_v5_20261002/qualification.json",
    "artifacts/v169_full_live_new_solver_resource_qualification_v2_20261002/armA/qualification.json",
    "artifacts/v169_full_live_new_solver_resource_qualification_v2_20261002/armB/qualification.json"
  ],
  "independent_storage_upper_bytes": 12136488300,
  "metadata_and_logs_upper_bytes": 135809024,
  "maximum_files": 128000,
  "maximum_directories": 17000,
  "independent_incremental_RAM_bound_bytes": 5376860272,
  "original_environment_full_touch_resource_basis_verified": true,
  "scientific_contract_and_entry_unchanged_except_resource_import": true,
  "independent_incremental_commit_bound_bytes": 6008197232,
  "minimum_free_commit_256MiB_quantum": 6174015488,
  "actual_resource_gate_negative_cases_refused": [
    "physical_shortfall",
    "commit_shortfall"
  ],
  "accepted_206_mixed_current_correct_protection_verified": true,
  "prospective_caps": {
    "heads": 56328,
    "features": 56328,
    "fixed_target_derivatives": 504,
    "margin_derivatives": 29184,
    "QP": 342,
    "proposals": 1146,
    "fits": 6,
    "updates": 120,
    "original_class_derivatives": 0,
    "all_complete_derivatives": 29688
  },
  "resources": {
    "full_run_worst_storage_bytes": 12136488300,
    "fixed_free_disk_reserve_bytes": 2147483648,
    "minimum_free_disk_start_bytes": 14283971948,
    "minimum_free_RAM_bytes": 5637144576,
    "minimum_free_GPU_bytes": 536870912,
    "minimum_free_commit_bytes": 6710886400
  },
  "elapsed_seconds": 42.39147100000264,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "source_sha256": {
    "training/v169_root_final_preseal_review_v2.py": "8a04c1bf0ef223d82544f2718703c1798ba3f9817e8c78d86079ad228a38546d",
    "training/v169_seal_prior_pair_training_v5.py": "be7b57cc2d12c520d131cced43078d5477a7429b7f68eb91528e7d834a0260c7",
    "artifacts/v169_full_preseal_bundle_v6_20261002/bundle.json": "d673dff43639b706eabc7b87761d76b74e1da4fa56ba331bf6218800be56104d",
    "artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json": "61b74d70e17025eaaab4fd9f40b5ece5d1485ac64011ae5c379231aadf091d64",
    "artifacts/v169_factorized_resource_budget_v10_20261002/review.json": "77c8e05c79a7f6f28af28aff6b5791f6d4c4453c1455edfefafb482fd31d25f3",
    "artifacts/v169_root_full_live_resource_review_20261002/review.json": "30fa4853194b82d0049267c011c5c52189813ce9e53cb201654631ca24688ed7",
    "training/v169_prior_pair_training_entry_v13.py": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
    "training/review_policy/v169_prior_pair_execution_contract_candidate_v4.json": "1beed13c3424c8ae515c39829f19ccc919c0105ce82f84d27461df94cd4e9c8c",
    "artifacts/v169_saved_storage_filecount_review_20261002/review.json": "d71e3f9b2afee7d47096aa8f1a391a5a786385273abfeb8d03744bf82d3dc073"
  }
}

## 实际原始来源 artifacts/v169_registered_initial_resource_refusal_20261002/snapshot.json；SHA256 `3f05ab145d70ec90b46611d332ea5a1d1a65cd2452d8b074e4a94f9198976c0c`

{
  "status": "V169_v14_registered_first_startup_terminal_RAM_refusal_before_backend_or_official_calls",
  "runner_observed_exit_code": 1,
  "failure_boundary": "main first require(initial), before configure and all ActualBackend construction",
  "official_zero_step_completed": false,
  "formal_registered_directory_contents": [
    "registration.json",
    "run_seal.json"
  ],
  "preseal_snapshot": {
    "free_disk_bytes": 15053062144,
    "free_RAM_bytes": 5874417664,
    "free_commit_bytes": 10196668416,
    "free_GPU_bytes": 7451181056,
    "total_GPU_bytes": 8585216000
  },
  "post_failure_CUDA_initialized_resource_snapshot": {
    "timestamp_UTC": "2026-10-02T07:53:22.194167+00:00",
    "available": {
      "physical_RAM": 4573757440,
      "available_commit": 10214883328,
      "GPU_free": 7451181056,
      "disk_free": 15049834496
    },
    "required": {
      "physical_RAM": 5637144576,
      "available_commit": 6710886400,
      "GPU_free": 536870912,
      "disk_free": 14283971948
    },
    "gaps": {
      "physical_RAM": 1063387136,
      "available_commit": 0,
      "GPU_free": 0,
      "disk_free": 0
    },
    "all_passed": false,
    "snapshot_is_after_failure_not_exact_failure_instant": true
  },
  "registered_failed_state_retained": true,
  "same_failed_entry_or_sealer_not_restarted": true,
  "source_or_threshold_not_modified": true,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "historical_heads": 19178,
  "historical_complete_derivatives": 768,
  "fits_since_V159": 9,
  "accepted_updates_since_V159": 170,
  "latest_actual_training": "V164",
  "latest_complete_delivery": "V159",
  "full_quality_acceptance": false,
  "three_goals_complete": false,
  "execution_authority": false,
  "source_sha256": {
    "training/v169_record_registered_initial_resource_refusal.py": "e909f5019207abaa4de87b9e62be569f16ec95edaa759227c98231a70f0e281d",
    "training/v169_prior_pair_training_entry_v14.py": "5c2cf876ced21ce56cf85f4147d103188335b2dedd1aa1d2561a3827fd968877",
    "training/review_policy/v169_prior_pair_execution_contract.json": "ea97901399d7ef548a434b4b7b461d0d326d3f002fa1b012897406306f140088",
    "artifacts/v169_prior_pair_training/registration.json": "23479670c2cef2430d8a3fcfa3afec74f61b0f9815da5083a1f946a059dfe03f",
    "artifacts/v169_prior_pair_training/run_seal.json": "e96cac5fbbb59ba18c59526b6ef6ae05548d500eaf5fe30e401228bc0101b633",
    "artifacts/v169_prior_pair_training_entry_v14_original_console_20261002.txt": "61bbec3c738df6197ef77e64721474f26e4b9be4e114c87b5e74a3549aea0bd8",
    "training/v169_pair_execution_review_v5.py": "940e2eb33804266bbca28b9948b84b951cb88dcb7c760679f246c4062778a458"
  }
}

## 实际原始来源 artifacts/v169_prior_pair_training/registration.json；SHA256 `23479670c2cef2430d8a3fcfa3afec74f61b0f9815da5083a1f946a059dfe03f`

{
  "status": "V169_physically_sealed_before_official_calls",
  "physical_sources": 18971,
  "entry_sha256": "5c2cf876ced21ce56cf85f4147d103188335b2dedd1aa1d2561a3827fd968877",
  "new_caps": {
    "heads": 56328,
    "features": 56328,
    "fixed_target_derivatives": 504,
    "margin_derivatives": 29184,
    "QP": 342,
    "proposals": 1146,
    "fits": 6,
    "updates": 120,
    "original_class_derivatives": 0,
    "all_complete_derivatives": 29688
  },
  "prior_actual_costs": {
    "heads": 19178,
    "features": 19178,
    "original_class_derivatives": 368,
    "fixed_target_derivatives": 32,
    "margin_derivatives": 368,
    "all_complete_derivatives": 768,
    "fits_since_V159": 9,
    "accepted_updates_since_V159": 170
  },
  "resources": {
    "full_run_worst_storage_bytes": 12136488300,
    "fixed_free_disk_reserve_bytes": 2147483648,
    "minimum_free_disk_start_bytes": 14283971948,
    "minimum_free_RAM_bytes": 5637144576,
    "minimum_free_GPU_bytes": 536870912,
    "minimum_free_commit_bytes": 6710886400
  },
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "run_seal_sha256": "e96cac5fbbb59ba18c59526b6ef6ae05548d500eaf5fe30e401228bc0101b633"
}


## Previous complete direction, original research, draft and all inherited limits

# V168独立结果与下一轮训练决定

**建议开展下一轮有界、配对训练；不直接重复V164入口，也不将V168称为已经训练成功。** 本次用户要求先判断与优化方案，因此新拟合尚未启动。当前仍只解决“训练侧没有学会分类”，不把同类支持和独立来源迁移同时混入本轮。

## 1. 实际结果与继续依据

根端 `training/v168_independent_actual_decision_floor_review.py` 实际 exit0；从官方独立gold、全部原行、25个完整输入函数、50次配对全参数裕量导数、原点两类目标梯度、实际候选、逐行保护、完全恢复和调用账本复算，并独立重放1次CPU保存向量QP。本审查0新官方调用。

|角色|相对V164的M修复/新增错|相对V164的S修复/新增错|纯错误：V164→安全候选|本次新收益|
|---|---:|---:|---:|---|
|0|356/0|0/0|3278→2922|冻结V166/V167参考，没有新增|
|1|4/0|0/0|1952→1948|消除V167新增2条S退化，保留4M修复|
|2|8/0|0/0|1586→1578|冻结V166/V167参考，没有新增|

角色1候选M错误864/38886，召回97.78%；S错误1156/2071，召回44.18%。消除两条S退化只是恢复V164已经具有的正确判断，相对V164没有新S修复。不能把这两条称为原S学习缺口取得突破。

实际两条S的q差约1.776e-15，现有argmax确实判对；没有认证跨硬件/跨状态稳定距离。下一入口必须重新核对真实分类，不能仅凭概率close认定旧正确不变。

V168新增98头/特征、50完整裕量导数、1QP、1真实候选、0拟合/永久更新，全部恢复V164。累计19178头/特征、768完整导数；自V159以来9拟合/170永久更新不变。最新实际训练仍V164，最新完整三分类交付仍V159；第一问题及总体质量均未通过。

独立证据：`artifacts/v168_independent_actual_decision_floor_review_20261002/review.json`。三角色均存在已实际核验的安全候选，支持另登记训练；这项许可不等于训练掌握或模型晋升。

## 2. 进步慢的两个主要卡点

### 2.1 训练被计算数量限制截断

V164只接受1/1/3个更新，之后完整保护函数并集25/27/32超过当轮上限24，在新导数测量前停止。不是“充分训练后仍无法分类”，也没有局部/全局不可行证明。

旧正确原行的永久保护与临时QP矩阵中的函数管理应分开。修复计算管理不授权漏查保护行、删除困难记录、放宽每类错误门槛或在旧运行中追认扩大容量。

### 2.2 固定错误先验给纠错分支施加了很大负担

根端 `training/v168_root_saved_prior_burden_review.py` 实际 exit0，仅使用保存的基座概率、逐行结果、原gold和CPU checkpoint，没有新官方分类器/梯度调用。

|V164状态|S错误|其中真值基座均值概率≤1e-12|错例固定先验中位log差|已学纠正中位值|当前读出最大纠正幅度|
|---|---:|---:|---:|---:|---:|
|角色0|1414|831|-27.631|+0.176|0.894|
|角色1|1156|813|-27.631|+0.143|0.981|
|角色2|892|319|-2.708|+0.528|2.550|

当前函数为 `z=log(max(mean_p,1e-12))+delta(x,p)`。基座不是威胁真值，但它的log意见系数被固定为1；基座将S赋零时，纠错分支需跨过约27.63的差距。角色1若固定当前输出矩阵、只改变tanh隐藏表示，最多提供约0.981的M/S差修正，1082条现有S错误仍无法翻转。

这个界限只针对当前固定读出。现有输出矩阵本来可训练，不能由此推导模型整体容量不足、原数据缺信息或错误必然无解。它支持检查“固定先验过强+接受轨迹太短”，尚未证明可学习先验权重必然有效。

独立证据：`artifacts/v168_root_saved_prior_burden_review_20261002/review.json`。

## 3. 下一轮的唯一学习问题与配对设计

**问题：在保持已经判对原行的条件下，能否真正推翻错误基座意见，学会当前训练侧M/S分类？**

采用三角色、两臂，共最多6次拟合。两臂共同先修复执行管理，唯一模型变化为固定先验的系数是否可学习：

- A控制：`z=log(max(mean_p,1e-12))+delta(x,p)`。
- B预登记候选：`z=exp(beta)*log(max(mean_p,1e-12))+delta(x,p)`，唯一新增标量beta初始化0。所有类别共享这个标量，不按标签、来源、时间、端口或错例编号配置系数。beta和现有完整头共同训练。

两臂相同输入、16个合法基座意见、原频次、gold、固定纯错误队列、类别完整分母、已验收能力、种子、初始化和有限验收；同一角色从同一个安全候选开始。beta=0必须真实重现A的完整q/logq和逐行判决。基座概率不改成标签、不重新伪标，不更改1e-12的底层零概率表示。

这不是对最终输出做后处理温度校准。正温度对全部最终logits的缩放保持argmax，不能单独修复错类；B只让基座意见相对于原始信息纠错分支的贡献可学习，其是否修复错类必须实测。

不得把B改成全局类别偏置、单独S权重、按来源路由或多参数注意力搜索；这些都超出本次单因素。若共享标量不足，只记录失败，不在本轮追加更复杂门控。

## 4. 两臂共用的执行修复

采用“完整保护账本＋动态局部工作集”的约束管理：

1. 全部注册正确原行、旧技能及每次真实接受的新修复永久留在完整账本；每个候选仍全量扫描实际argmax、原M/S错误数、原风险和所有部署/旧技能。
2. 每个新接受参数点重新计算两类完整固定错误目标梯度并复测。失效的旧点Jacobian不得复用。
3. QP工作集随实际阻塞更新；可以移出当前不活动的局部函数，但不得移出其原行保护。移出后仍完整扫描，一旦再次阻塞必须重新纳入并在当前点测量。
4. 同时需要更多活动函数时，分块/精确稀疏存储和资源前置资格处理；不得把未测函数当安全，也不得靠丢行凑数量上限。真正达到事前物理预算即停止并恢复，不声称模型不可学。
5. V168已验证的判决地板规则、原单位数值复核和真实有限验收共同保留。局部地板仅辅助求方向；不能放宽实际判决。
6. 新入口首次真实重放三个安全候选。逐行判决与完整保护通过后才提交初始化为新训练状态并冻结其修复；诊断候选不是已永久掌握成果，不能跳过这一步。

工作集方法是针对既有容量截断的执行修复，不能将传统线性QP的收敛性质直接套到神经网络。工作集、局部修正和候选总次数均须另登记；没有自动循环到成功的权限。

## 5. 预算、关键记录与验收

前瞻设计上限为每臂每角色20个真实接受更新、每个新点最多2次修正；最多6拟合/120接受更新。采用固定注册日程，不因早期弱切片未翻类而提前结束。提前结束只允许真正掌握并完成末5个真实不同状态验收、技术失败、原保护无法满足或达到事前资源/调用限制；停止原因分别记录。

该数字不是执行授权。实施前依据实际调用图计算头/意见、完整梯度、QP、有限候选、磁盘/RAM/GPU的精确最坏上限，并与现有累计成本共同绑定。配对完整导数需能无损恢复全部参数；精确稀疏/压缩只能改变保存方式，不能改变数值、参数范围或独立可复算性。预算不足时事前减小两臂相同范围或使用平台，不在看到结果后调整。

必须每点保存：

- 实际参数身份、原点/线性化点、完整目标与裕量导数身份、接受/拒绝/恢复及真实成本。
- 每类原始分母、正确/错误、召回/精确率、修复/新增错误；混类原行仍完整计分。重叠角色不得当独立记录相加。
- 初始固定的低先验S切片及其他S切片：实际错数、修复原行/独立输入数、裕量分布、基座先验与学习残差分解。始终使用同一个初始切片，不用“剩余错例集合”偷换分母。
- B的beta/实际先验系数、输出矩阵纠正界限及两类实际修复；系数变化、loss下降不算分类成功。
- 所有累计保护回归；两条S微小正间隔在以后状态中仍必须真实判对。

预登记判定：

1. 安全：全部已验收范围与累计接受修复新增错0，原M/S逐类错误不增加；任何一项失败就拒绝候选。
2. 学习：各角色纯错误是否减少，尤其初始低先验S是否出现真实修复；完整混类错误独立报告。不只看总准确率或CE。
3. B机制支持：匹配终点B在三角色各类错误均不高于A，并且各角色S有实际额外修复；低先验S切片是否改善单列。否则只能称局部或无支持，不换种子挑折晋升。B是唯一候选，A结果更好时保留旧正式模型、记录对照发现，不事后更换候选。
4. 第一问题真正关闭：三个注册角色纯错误0，末5个真实不同接受参数状态均掌握，全部旧技能与累计修复保持。未达到就不称解决；有限诊断、少量修复不能替代。
5. 来源泛化与完整N/M/S质量：当前已反复查看的开发角色不能称盲测。第一问题掌握也不自动通过后两问题或2056871行完整质量验收；后续单独检验。

如果日程完成仍无深S分类收益，停止本配置，依据beta是否真正变化、读出幅度、有限保护阻塞及每类错误分解归因。不要再把“降低了损失但未翻类”写成有效方案，也不机械追加训练轮数。

## 6. 调研取舍与反证

- [NEOS的QP工作集方法](https://neos-guide.org/guide/algorithms/qp/)说明局部工作集可随活动约束更新。借鉴它管理计算，不借此取消完整神经网络逐行验收。
- [scikit-learn 1.3.2 StackingClassifier](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.StackingClassifier.html)将基模型输出作为最终学习器输入，默认最终学习器是逻辑回归。借鉴“基座意见应由最终监督学习器决定贡献”，本项目共享先验系数是最小可检验改法，不是该库效果的复现或保证。其关于同数据预拟合堆叠的过拟合限制仍适用，不能偷换来源外验收。
- [ICML 2017校准论文](https://proceedings.mlr.press/v70/guo17a.html)及[作者实现](https://github.com/gpleiss/temperature_scaling)支持概率校准。正标量缩放最终logits不改变排序，故不采纳它作为本轮错类修复的替代品。
- [SciPy 1.15.3 SLSQP](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-slsqp.html)中的精度/迭代参数只描述优化停止，不证明原神经网络分类安全。保持原单位复算与实际分类门槛。

本次审查保留四个反证：平手修复不等于S学习；容量截断不等于充分训练；当前固定读出界限不等于全模型不可学；学习先验权重不等于来源泛化。新方案须先验证真实零步、完整参数/预算、异常恢复、工作集移出后再阻塞、两臂同人口及历史反例，再封存执行。当前新方案0拟合，尚无效果声明。


## 当前实际状态：设计审查通过，封存因RAM门槛拒绝

根实际独立审查已通过最终v13、v5完整依赖、v7资源、契约候选v3和封存器v4，完整18767项哈希及93模块闭包；只支持满足实际资源之后封存。随后实际执行封存器停在physical RAM检查，未生成正式OUT、PLAN、run_seal或registration，0官方头/特征/导数/fit/永久更新，最新训练V164、完整交付V159；三个目标未完成。

CUDA初始化后快照：可用RAM5,762,314,240 bytes，须6,442,450,944，缺680,136,704 bytes（约649MiB）；commit9,966,776,320≥6,710,886,400，显存7,451,181,056≥536,870,912，磁盘14,338,809,856≥14,283,971,948。这是保存时点的实测，启动前必须再次读取。不得降低门槛、关闭用户应用或在资源未变化时盲重试。未写正式目录/契约，资源变化满足后可按同一根审查、同一封存器重试；若以后已写目录后失败，保留失败状态且不能从中重启。


### 来源 artifacts/v169_root_final_preseal_review_20261002/review.json；SHA256 `e23022a95bb1c44fb2a18d5b450dafe228a7e26079dd81c1e17cb937dde2c378`

{
  "status": "V169_independent_final_source_and_resource_design_review_passed",
  "reviewed_entry_sha256": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
  "supports_physical_seal": true,
  "execution_authority": false,
  "new_fit_permission": false,
  "actual_resource_gates_must_still_pass_before_seal_and_each_fit": true,
  "official_zero_step_and_all_registered_real_quality_gates_still_required": true,
  "hash_verified_files": 18767,
  "bytes_hashed": 35129747761,
  "recursive_project_execution_and_sealing_modules": 93,
  "qualifications": [
    "artifacts/v169_pair_model_storage_qualification_20261002/qualification.json",
    "artifacts/v169_pair_lifecycle_qualification_v2_20261002/qualification.json",
    "artifacts/v169_saved_state_quality_qualification_20261002/qualification.json",
    "artifacts/v169_working_solver_full_size_qualification_20261002/qualification.json",
    "artifacts/v169_factorized_storage_qualification_v2_20261002/qualification.json",
    "artifacts/v169_measurement_references_qualification_20261002/qualification.json",
    "artifacts/v169_dataframe_and_current_guard_qualification_20261002/qualification.json",
    "artifacts/v169_actual_backend_synthetic_qualification_v4_20261002/qualification.json",
    "artifacts/v169_saved_correction_storage_qualification_20261002/qualification.json",
    "artifacts/v169_execution_contract_candidate_v3_20261002/qualification.json",
    "artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json",
    "artifacts/v169_cuda_host_buffer_qualification_v2_20261002/qualification.json",
    "artifacts/v169_physical_and_commit_policy_qualification_20261002/qualification.json"
  ],
  "independent_storage_upper_bytes": 12136488300,
  "metadata_and_logs_upper_bytes": 135809024,
  "maximum_files": 128000,
  "maximum_directories": 17000,
  "independent_incremental_RAM_bound_bytes": 5936980096,
  "independent_incremental_commit_bound_bytes": 6568317056,
  "minimum_free_commit_256MiB_quantum": 6710886400,
  "actual_resource_gate_negative_cases_refused": [
    "physical_shortfall",
    "commit_shortfall"
  ],
  "accepted_206_mixed_current_correct_protection_verified": true,
  "prospective_caps": {
    "heads": 56328,
    "features": 56328,
    "fixed_target_derivatives": 504,
    "margin_derivatives": 29184,
    "QP": 342,
    "proposals": 1146,
    "fits": 6,
    "updates": 120,
    "original_class_derivatives": 0,
    "all_complete_derivatives": 29688
  },
  "resources": {
    "full_run_worst_storage_bytes": 12136488300,
    "fixed_free_disk_reserve_bytes": 2147483648,
    "minimum_free_disk_start_bytes": 14283971948,
    "minimum_free_RAM_bytes": 6442450944,
    "minimum_free_GPU_bytes": 536870912,
    "minimum_free_commit_bytes": 6710886400
  },
  "elapsed_seconds": 41.78392850000091,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "source_sha256": {
    "artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json": "e2375da9a3cff4fcd1dd1bd135a3c1d9f8a1f87922d17aeedea373ab322e1901",
    "training/v169_seal_prior_pair_training_v4.py": "6cec63af110f225e4e3d121df8464d5b8a2c0f7d09befa776f5659558dd13206",
    "training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json": "2888379069f0ac64c7ef684b3dd7aaf4521613f820ac675e79b1d559a0e9cb80",
    "artifacts/v169_factorized_resource_budget_v7_20261002/review.json": "7ffc89a38e96d0f894af4ae3eade28135614dce67016cd7bc6643da2cbd85ad2",
    "training/v169_root_final_preseal_review.py": "204a289756c24f14c10fd9281baa5d688c3e0d1f3c266b05c6cbea5fa6340378",
    "artifacts/v169_saved_storage_filecount_review_20261002/review.json": "d71e3f9b2afee7d47096aa8f1a391a5a786385273abfeb8d03744bf82d3dc073",
    "artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json": "61b74d70e17025eaaab4fd9f40b5ece5d1485ac64011ae5c379231aadf091d64"
  }
}

### 来源 artifacts/v169_actual_preseal_resource_shortfall_20261002/snapshot.json；SHA256 `8c909469661c5f723aec7b4b5ea4d126990ac18029e976cca5b89656da30141c`

{
  "status": "V169_actual_physical_RAM_preseal_refusal_recorded_no_registered_or_official_training",
  "timestamp_UTC": "2026-10-02T06:53:16.574089+00:00",
  "metadata_snapshot_after_CUDA_initialization": true,
  "resources": {
    "physical_RAM": {
      "actual_available_bytes": 5762314240,
      "required_available_bytes": 6442450944,
      "shortfall_bytes": 680136704,
      "passed": false
    },
    "available_commit": {
      "actual_available_bytes": 9966776320,
      "required_available_bytes": 6710886400,
      "shortfall_bytes": 0,
      "passed": true
    },
    "GPU_free": {
      "actual_available_bytes": 7451181056,
      "required_available_bytes": 536870912,
      "shortfall_bytes": 0,
      "passed": true
    },
    "disk_free": {
      "actual_available_bytes": 14338809856,
      "required_available_bytes": 14283971948,
      "shortfall_bytes": 0,
      "passed": true
    }
  },
  "GPU_total_bytes": 8585216000,
  "all_current_resource_prerequisites_met": false,
  "formal_OUT_exists": false,
  "formal_PLAN_exists": false,
  "registration_completed": false,
  "failed_seal_attempt_has_no_written_run_directory": true,
  "root_design_review_remains_valid": true,
  "root_review_support_is_conditional_on_actual_resources": true,
  "latest_actual_training": "V164",
  "latest_complete_quality_delivery": "V159",
  "current_direction": "V169_same_v13_and_registered_candidate_wait_actual_resources",
  "no_threshold_lowered_or_application_closed": true,
  "no_blind_repeated_seal_attempt": true,
  "resource_change_required_before_same_sealer_retry": true,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "execution_authority": false,
  "source_sha256": {
    "training/v169_record_actual_resource_shortfall.py": "6f4ad53d2c06724c4b66fb32f3043ecad6285df56e44e61f54c3d09257736576",
    "training/v169_prior_pair_training_entry_v13.py": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
    "training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json": "2888379069f0ac64c7ef684b3dd7aaf4521613f820ac675e79b1d559a0e9cb80",
    "artifacts/v169_root_final_preseal_review_20261002/review.json": "e23022a95bb1c44fb2a18d5b450dafe228a7e26079dd81c1e17cb937dde2c378",
    "training/v169_seal_prior_pair_training_v4.py": "6cec63af110f225e4e3d121df8464d5b8a2c0f7d09befa776f5659558dd13206",
    "artifacts/v169_seal_prior_pair_training_v4_original_console_20261002.txt": "8236045a8d00a8e2302277ffaaa7f83043e3429bae3708328a790493a86bca01",
    "artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json": "e2375da9a3cff4fcd1dd1bd135a3c1d9f8a1f87922d17aeedea373ab322e1901"
  }
}


## 完整上一方向、原始研究、草案及全部历史限制保留

# V168独立结果与下一轮训练决定

**建议开展下一轮有界、配对训练；不直接重复V164入口，也不将V168称为已经训练成功。** 本次用户要求先判断与优化方案，因此新拟合尚未启动。当前仍只解决“训练侧没有学会分类”，不把同类支持和独立来源迁移同时混入本轮。

## 1. 实际结果与继续依据

根端 `training/v168_independent_actual_decision_floor_review.py` 实际 exit0；从官方独立gold、全部原行、25个完整输入函数、50次配对全参数裕量导数、原点两类目标梯度、实际候选、逐行保护、完全恢复和调用账本复算，并独立重放1次CPU保存向量QP。本审查0新官方调用。

|角色|相对V164的M修复/新增错|相对V164的S修复/新增错|纯错误：V164→安全候选|本次新收益|
|---|---:|---:|---:|---|
|0|356/0|0/0|3278→2922|冻结V166/V167参考，没有新增|
|1|4/0|0/0|1952→1948|消除V167新增2条S退化，保留4M修复|
|2|8/0|0/0|1586→1578|冻结V166/V167参考，没有新增|

角色1候选M错误864/38886，召回97.78%；S错误1156/2071，召回44.18%。消除两条S退化只是恢复V164已经具有的正确判断，相对V164没有新S修复。不能把这两条称为原S学习缺口取得突破。

实际两条S的q差约1.776e-15，现有argmax确实判对；没有认证跨硬件/跨状态稳定距离。下一入口必须重新核对真实分类，不能仅凭概率close认定旧正确不变。

V168新增98头/特征、50完整裕量导数、1QP、1真实候选、0拟合/永久更新，全部恢复V164。累计19178头/特征、768完整导数；自V159以来9拟合/170永久更新不变。最新实际训练仍V164，最新完整三分类交付仍V159；第一问题及总体质量均未通过。

独立证据：`artifacts/v168_independent_actual_decision_floor_review_20261002/review.json`。三角色均存在已实际核验的安全候选，支持另登记训练；这项许可不等于训练掌握或模型晋升。

## 2. 进步慢的两个主要卡点

### 2.1 训练被计算数量限制截断

V164只接受1/1/3个更新，之后完整保护函数并集25/27/32超过当轮上限24，在新导数测量前停止。不是“充分训练后仍无法分类”，也没有局部/全局不可行证明。

旧正确原行的永久保护与临时QP矩阵中的函数管理应分开。修复计算管理不授权漏查保护行、删除困难记录、放宽每类错误门槛或在旧运行中追认扩大容量。

### 2.2 固定错误先验给纠错分支施加了很大负担

根端 `training/v168_root_saved_prior_burden_review.py` 实际 exit0，仅使用保存的基座概率、逐行结果、原gold和CPU checkpoint，没有新官方分类器/梯度调用。

|V164状态|S错误|其中真值基座均值概率≤1e-12|错例固定先验中位log差|已学纠正中位值|当前读出最大纠正幅度|
|---|---:|---:|---:|---:|---:|
|角色0|1414|831|-27.631|+0.176|0.894|
|角色1|1156|813|-27.631|+0.143|0.981|
|角色2|892|319|-2.708|+0.528|2.550|

当前函数为 `z=log(max(mean_p,1e-12))+delta(x,p)`。基座不是威胁真值，但它的log意见系数被固定为1；基座将S赋零时，纠错分支需跨过约27.63的差距。角色1若固定当前输出矩阵、只改变tanh隐藏表示，最多提供约0.981的M/S差修正，1082条现有S错误仍无法翻转。

这个界限只针对当前固定读出。现有输出矩阵本来可训练，不能由此推导模型整体容量不足、原数据缺信息或错误必然无解。它支持检查“固定先验过强+接受轨迹太短”，尚未证明可学习先验权重必然有效。

独立证据：`artifacts/v168_root_saved_prior_burden_review_20261002/review.json`。

## 3. 下一轮的唯一学习问题与配对设计

**问题：在保持已经判对原行的条件下，能否真正推翻错误基座意见，学会当前训练侧M/S分类？**

采用三角色、两臂，共最多6次拟合。两臂共同先修复执行管理，唯一模型变化为固定先验的系数是否可学习：

- A控制：`z=log(max(mean_p,1e-12))+delta(x,p)`。
- B预登记候选：`z=exp(beta)*log(max(mean_p,1e-12))+delta(x,p)`，唯一新增标量beta初始化0。所有类别共享这个标量，不按标签、来源、时间、端口或错例编号配置系数。beta和现有完整头共同训练。

两臂相同输入、16个合法基座意见、原频次、gold、固定纯错误队列、类别完整分母、已验收能力、种子、初始化和有限验收；同一角色从同一个安全候选开始。beta=0必须真实重现A的完整q/logq和逐行判决。基座概率不改成标签、不重新伪标，不更改1e-12的底层零概率表示。

这不是对最终输出做后处理温度校准。正温度对全部最终logits的缩放保持argmax，不能单独修复错类；B只让基座意见相对于原始信息纠错分支的贡献可学习，其是否修复错类必须实测。

不得把B改成全局类别偏置、单独S权重、按来源路由或多参数注意力搜索；这些都超出本次单因素。若共享标量不足，只记录失败，不在本轮追加更复杂门控。

## 4. 两臂共用的执行修复

采用“完整保护账本＋动态局部工作集”的约束管理：

1. 全部注册正确原行、旧技能及每次真实接受的新修复永久留在完整账本；每个候选仍全量扫描实际argmax、原M/S错误数、原风险和所有部署/旧技能。
2. 每个新接受参数点重新计算两类完整固定错误目标梯度并复测。失效的旧点Jacobian不得复用。
3. QP工作集随实际阻塞更新；可以移出当前不活动的局部函数，但不得移出其原行保护。移出后仍完整扫描，一旦再次阻塞必须重新纳入并在当前点测量。
4. 同时需要更多活动函数时，分块/精确稀疏存储和资源前置资格处理；不得把未测函数当安全，也不得靠丢行凑数量上限。真正达到事前物理预算即停止并恢复，不声称模型不可学。
5. V168已验证的判决地板规则、原单位数值复核和真实有限验收共同保留。局部地板仅辅助求方向；不能放宽实际判决。
6. 新入口首次真实重放三个安全候选。逐行判决与完整保护通过后才提交初始化为新训练状态并冻结其修复；诊断候选不是已永久掌握成果，不能跳过这一步。

工作集方法是针对既有容量截断的执行修复，不能将传统线性QP的收敛性质直接套到神经网络。工作集、局部修正和候选总次数均须另登记；没有自动循环到成功的权限。

## 5. 预算、关键记录与验收

前瞻设计上限为每臂每角色20个真实接受更新、每个新点最多2次修正；最多6拟合/120接受更新。采用固定注册日程，不因早期弱切片未翻类而提前结束。提前结束只允许真正掌握并完成末5个真实不同状态验收、技术失败、原保护无法满足或达到事前资源/调用限制；停止原因分别记录。

该数字不是执行授权。实施前依据实际调用图计算头/意见、完整梯度、QP、有限候选、磁盘/RAM/GPU的精确最坏上限，并与现有累计成本共同绑定。配对完整导数需能无损恢复全部参数；精确稀疏/压缩只能改变保存方式，不能改变数值、参数范围或独立可复算性。预算不足时事前减小两臂相同范围或使用平台，不在看到结果后调整。

必须每点保存：

- 实际参数身份、原点/线性化点、完整目标与裕量导数身份、接受/拒绝/恢复及真实成本。
- 每类原始分母、正确/错误、召回/精确率、修复/新增错误；混类原行仍完整计分。重叠角色不得当独立记录相加。
- 初始固定的低先验S切片及其他S切片：实际错数、修复原行/独立输入数、裕量分布、基座先验与学习残差分解。始终使用同一个初始切片，不用“剩余错例集合”偷换分母。
- B的beta/实际先验系数、输出矩阵纠正界限及两类实际修复；系数变化、loss下降不算分类成功。
- 所有累计保护回归；两条S微小正间隔在以后状态中仍必须真实判对。

预登记判定：

1. 安全：全部已验收范围与累计接受修复新增错0，原M/S逐类错误不增加；任何一项失败就拒绝候选。
2. 学习：各角色纯错误是否减少，尤其初始低先验S是否出现真实修复；完整混类错误独立报告。不只看总准确率或CE。
3. B机制支持：匹配终点B在三角色各类错误均不高于A，并且各角色S有实际额外修复；低先验S切片是否改善单列。否则只能称局部或无支持，不换种子挑折晋升。B是唯一候选，A结果更好时保留旧正式模型、记录对照发现，不事后更换候选。
4. 第一问题真正关闭：三个注册角色纯错误0，末5个真实不同接受参数状态均掌握，全部旧技能与累计修复保持。未达到就不称解决；有限诊断、少量修复不能替代。
5. 来源泛化与完整N/M/S质量：当前已反复查看的开发角色不能称盲测。第一问题掌握也不自动通过后两问题或2056871行完整质量验收；后续单独检验。

如果日程完成仍无深S分类收益，停止本配置，依据beta是否真正变化、读出幅度、有限保护阻塞及每类错误分解归因。不要再把“降低了损失但未翻类”写成有效方案，也不机械追加训练轮数。

## 6. 调研取舍与反证

- [NEOS的QP工作集方法](https://neos-guide.org/guide/algorithms/qp/)说明局部工作集可随活动约束更新。借鉴它管理计算，不借此取消完整神经网络逐行验收。
- [scikit-learn 1.3.2 StackingClassifier](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.StackingClassifier.html)将基模型输出作为最终学习器输入，默认最终学习器是逻辑回归。借鉴“基座意见应由最终监督学习器决定贡献”，本项目共享先验系数是最小可检验改法，不是该库效果的复现或保证。其关于同数据预拟合堆叠的过拟合限制仍适用，不能偷换来源外验收。
- [ICML 2017校准论文](https://proceedings.mlr.press/v70/guo17a.html)及[作者实现](https://github.com/gpleiss/temperature_scaling)支持概率校准。正标量缩放最终logits不改变排序，故不采纳它作为本轮错类修复的替代品。
- [SciPy 1.15.3 SLSQP](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-slsqp.html)中的精度/迭代参数只描述优化停止，不证明原神经网络分类安全。保持原单位复算与实际分类门槛。

本次审查保留四个反证：平手修复不等于S学习；容量截断不等于充分训练；当前固定读出界限不等于全模型不可学；学习先验权重不等于来源泛化。新方案须先验证真实零步、完整参数/预算、异常恢复、工作集移出后再阻塞、两臂同人口及历史反例，再封存执行。当前新方案0拟合，尚无效果声明。


## V169当前准备状态（本节覆盖旧附录的未实现状态）

调用图已独立核算，完整参数、DataFrame接口、206条旧正确混类记录保护、实际CPU合成后台、终点同参数重放和全局故障停止已取得资格。最新入口v13；没有新增模型方法。原QP去掉一份完整矩阵副本；四次A/B完整64函数对照证明SVD输入、完整修正/位移及原单位验证逐位相同。物理存储/RAM工作区仍须根独立审查，执行预算尚未注册，根preseal/正式封存和真实零步尚未执行。最新实际训练V164、完整2056871行质量交付V159，训练分类未掌握，三个目标未完成。

拟议新范围：56328头/特征、504固定目标导数、29184间隔导数、342QP、1146试探、6fit和最多120接受更新。历史19178头、768完整导数、9fit/170更新不重置；旧82174/2426只绑定各自旧试验。两臂20接受更新均含安全初始化，每次新方向最多8回溯/2纠错/64当前函数；完整永久保护不可缩减。非匹配20终点或技术故障为inconclusive，不选早期状态晋升。

磁盘上界计三份接受保护掩码，按文件类型落实JSON/计数日志上限，并保留全部不同测量q/logq，未假设可复用。文件数上界已纠正为128000、目录17000，同一640MiB开销仍覆盖126522/16693最坏实际清单。整轮12,136,488,300 bytes，固定reserve2GiB，起始至少14,283,971,948 bytes；RAM6GiB、可用OS提交空间6.25GiB、GPU512MiB。保存快照磁盘/GPU满足而RAM不足；启动前重新核查。CPU合成资格代替了legacy跨角色保留oracle及官方frozen helper，仅证明后台接线；正式旧能力及训练收益须真实零步和完整原行验收。



### 原始来源 docs/V169_INCREMENTAL_RESEARCH_AND_TRAINING_REVIEW_20261002.md

SHA256 `42c566eb2c7c09b1999ed57c2fb001ac27db0b1ab2d03582c583ee1a86bc563c`

# V169递进审查：下一轮是否值得训练

建议开展有界配对试验，但不能原样重跑旧入口。最新真实训练为V164；V168是零拟合的诊断修复。下一轮目前仍处于实现与资格审查阶段，没有新官方模型调用或分类效果。

## 实际结果支持什么

V168根独立审查重算官方gold与原行判决后通过：角色1保留4条M修复，消除诊断候选新增的2条S退化；相对V164没有新增S修复。角色0/2的356/8条M修复是旧诊断安全参考，不能重复当成本轮新收益；角色人口重叠，不能把各角色修复数相加当独立样本收益。

角色1安全候选的M错误864/38886、S错误1156/2071，分别对应召回97.78%和44.18%。总体正确率不能掩盖S仍有超过半数判错。

旧V164分别只接受1/1/3个更新，因局部保护函数并集25/27/32超过24而停止。这证明执行容量受限，尚不能说明模型充分训练后仍不可学。

保存证据还发现：三个角色的低先验S错分别831/813/319。基座给真实类别接近零概率时，固定log先验可带来约27.63的逆向差距；角色1已学纠正中位数约0.143，当前读出最大纠正幅度约0.981。这个上界只适用于冻结当前输出矩阵，不证明完整可训练模型无解，也不证明原始日志缺少信息。

证据分别为 `artifacts/v168_independent_actual_decision_floor_review_20261002/review.json`、`artifacts/v168_root_saved_prior_burden_review_20261002/review.json` 和 `artifacts/v164_independent_capacity_stop_review_20261002/review.json`。

## 借鉴的成熟方法及适用边界

|来源|已有证据支持的做法|本项目的取舍|
|---|---|---|
|[StackingClassifier官方文档](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.StackingClassifier.html)|最终监督学习器学习如何组合基模型输出；同数据预拟合堆叠存在高过拟合风险|把固定基座贡献作为可学习对象；保留合法意见银行和原行验证，不声称堆叠必然解决跨来源问题|
|[GEM论文](https://arxiv.org/abs/1706.08840)及[作者实现](https://github.com/facebookresearch/GradientEpisodicMemory)|约束更新以减轻旧任务遗忘，并允许有益迁移；原实验并非SOC|借鉴保留旧能力的思想；本项目仍实际检查全部旧正确原行，不能由平均旧任务loss或局部梯度符号推断逐行分类不退化|
|[NEOS工作集方法](https://neos-guide.org/guide/algorithms/qp/)|局部求解工作集随活动约束更新|完整保护账本永久保留，局部函数可以更换；每次候选仍全面检查，不能把工作集移出解释为保护撤销|
|[SciPy 1.15.3线搜索文档](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.line_search.html)|下降方向、步长条件和附加条件需要分别验收，并明确函数/梯度调用次数|借鉴有界步长搜索；不声称本项目双目标回退搜索等同强Wolfe实现。实际M/S下降与全部分类保护共同决定接受|
|[温度校准论文](https://proceedings.mlr.press/v70/guo17a.html)|温度缩放可改善概率校准|不采用最终全部logits统一正缩放作为错类修复；其argmax排序不变。本轮只调整先验与纠错分支的相对贡献|

上述来源2026-10-02重新读取。外部任务有效性只提供方法依据，不能代替官方SOC数据上的配对结果。

## 唯一模型变化与公平对照

A保持 `z=log(max(mean_p,1e-12))+delta`；B使用 `z=exp(beta)*log(max(mean_p,1e-12))+delta`，所有类别共享一个beta，初始化0。B的完整参数数为1060833，旧四个参数区段不变，新beta必须进入状态、哈希、复测、提交、恢复和自己的数值检查区段。

两臂共同修复动态局部工作集和步长政策。共同修复的收益由A揭示，B相对A的额外收益才支持可学习先验。相同原始输入、标签、频次、保护人口、初始化、三角色、固定终点；不得用新数据、更高S权重或来源规则混入本轮。

不机械沿用固定1/16步长。条件性算术反例是：方向每坐标绝对值不超过1、没有额外修正时，20次1/16更新最多将beta减到-1.25，系数仍约0.286；27.63的错误先验差距仍约7.92。它不是实际训练结果或全模型不可行证明，但足以说明新标量不能不经审查继承旧步幅。

两臂采用事前相同的有限步长政策，按真实双目标Armijo与完整分类保护选择第一个合格提议。旧 `finite_step_review` 中的'A'是平均风险模式，不能与新实验控制臂A混用；两个实验臂均须明确采用严格逐M/S模式。

## 实现审查中已实际暴露的问题

根端 `training/v169_root_preparation_adversarial_review.py` 使用合成后端实际执行，0官方模型调用。20次包含初始化、当前参数点目标覆盖、早期失败恢复、未匹配终点拒绝和提交后状态审查失败的账本保持通过。最终重放抛异常时，初版未保证关闭计数器或记录该终端异常；报告 `artifacts/v169_root_preparation_adversarial_review_20261002/review.json` 因而明确未通过。

必须先修复并重新实测，不能把脚本exit0当成资格通过。入口还需拒绝坏零步、绑定V164原始保存判决、检查全部源码/数据/环境和精确最坏调用图，并保证所有异常的全参数、保护和风险基线恢复。

64个同时局部函数若达到上限，只说明本轮资源界限，不说明该方法无效。历史82174/2426也不是项目永久预算。V160原2426是完整类目标导数上限，裕量导数另计；后续某轮才注册合并导数上限。保留真实累计19178头/768导数及9拟合/170更新，不能混单位、隐去历史成本或事后扩大旧预算。

## 下一轮记录与胜负判定

事前计划仍为3角色×2臂，最多6拟合；每拟合20次真实接受更新，包括初始化提交，总上限120。具体执行权必须来自实现资格、根独立审查和新物理封存，草案数字不是执行许可。

每次接受保存原行M/S分母、错误、召回、精确率、修复/退化、累计保护、beta、先验贡献、残差及拒绝原因。初始831/813/319条低先验S切片固定不变，同时报告独立输入数和重复原行数。保留完整混类计分，避免只统计修复后剩余错例。

必须区分四层结论：

1. **执行有效**：新入口资格、零步、源码/数据身份、调用/资源预算、全部恢复及配对终点合格。技术早停或终点不匹配只允许“结论不足”。
2. **旧能力保留**：全部注册旧正确和累计真实修复新增错0，M/S各类错误都不增加；失败候选不得提交。
3. **机制有分类收益**：匹配终点B三角色每类错误不高于A且每个角色S有额外修复。低先验S必须单独核查；若它们仍完全未修复，不得宣称固定先验造成的深错误已解决。
4. **第一问题关闭**：三个角色纯错误0，最后5个真实不同接受状态均掌握，旧能力与累计修复保持。此前只称改善。完整三分类和新来源迁移后续单独检验，反复查看的角色不得称盲测。

若完成日程仍没有深S收益，停止本配置，按beta/读出实际变化、真实保护阻塞、错例行为和完整参数复测归因，不靠改种子、换分母或追加相同训练追认成功。


### 原始来源 docs/V169_PRETRAINING_INDEPENDENT_FINDINGS_20261002.md

SHA256 `48774a5b99d1c11ff7155120dad03d55aa218ca461a7c41c0b7e2e1ee09100d7`

# V169执行前独立审查补充

建议下一轮进行有界配对训练，学习变化仍只检验“固定基座贡献是否阻碍纠错”。目前没有新的官方模型推理、梯度、拟合或准确率结果；本文件记录执行前实际发现，不能冒充模型训练进展。

## 为什么仍值得训练

最近真实训练V164仅完成1/1/3次接受更新，受到局部函数容量限制。V168是零拟合诊断：角色1相对V164修好4条M，没有新增S修复；安全候选仍有1156/2071条S错误，S召回44.18%。因此，现有证据同时指向未充分完成的学习过程，以及固定先验与纠错分支之间的明显幅度失衡，尚不支持原样重复训练或宣布更强模型无效。

唯一新学习因素保持为共享可学习先验系数：A固定系数1，B用exp(beta)，beta从0开始。两臂共同采用相同局部工作集、有限步长搜索、全部原行保护、频次、初始化及固定20次接受更新日程。A揭示执行策略修复的收益，B相对A的收益才支持改变先验贡献。

方法依据及原始来源见[调研与训练审查](V169_INCREMENTAL_RESEARCH_AND_TRAINING_REVIEW_20261002.md)。堆叠学习提供学习基模型组合的依据，GEM提供保留既有能力的参考；均不替代本项目实际分类验收。

## 本次实际执行的审查及反例

|审查对象|实际结果|对下一轮的影响|
|---|---|---|
|完整输入与调用图|原CSR保留66287列，实际观察列并集412；3角色×2臂的最坏完整导数29688次|不混淆完整模型维度、真实支持与物理存储；必须预算整个日程|
|62条保存的完整真实导数|逐位检查通过，结构支持外没有非零位，负零也检查|可做无损存储；不授权裁剪参数或丢弃微小值|
|保存的真实QP修正与位移|各有26个数据支持外非零坐标；现有数据支持加前80坐标的编码逐位恢复通过|模型导数支持不能直接套在QP输出上；新求解结果仍须运行时检查，禁止按小数阈值归零|
|真实pandas阻挡表与生命周期v2接线|空表、非空表均复现真假值歧义异常，进入纠错前提前停止|v2不具执行资格；v3明确定义行数后，两个分支及以前的终端故障独立回放通过|
|V164当前正确原行保护|角色0/1/2有106/6/94条混类正确记录不在protected_correct内，共206个独立原行|须显式并入两臂永久保护及阻挡函数；不能由每类总错误不增代替逐行保留|

根端审查凭据：

- `artifacts/v169_root_input_support_and_callgraph_review_20261002/review.json`
- `artifacts/v169_root_saved_derivative_structural_support_review_20261002/review.json`
- `artifacts/v169_root_saved_qp_storage_support_review_20261002/review.json`
- `artifacts/v169_root_actual_blocker_type_counterexample_20261002/review.json`
- `artifacts/v169_root_real_frame_lifecycle_regression_review_20261002/review.json`
- `artifacts/v169_root_existing_correct_mixed_guard_review_20261002/review.json`

第一次QP审查推断v5仍只有数据支持，因此会失败；重新读实际源码发现v5已经加入前80坐标，实测也覆盖这两条真实QP向量。该推断已撤回并记录，不能据它宣称现有v5再次失败或增加无必要的新存储架构。

## 最小执行条件与真实验收

正式训练前补齐206条旧正确混类保护的实际接线与资格，完成完整后端及最坏物理存储/内存预算，绑定最终入口、完整依赖、数据与环境。真实零步必须复现V164判决、风险和合法意见银行，B的beta=0必须与A一致；未通过不得提交。技术或恢复失败应保留最后真实提交、保护和计数，结束批次，不能将失败重放作有效学习。

完成匹配日程后按官方gold重算原行分类：每角色M/S分母、错误、召回、精确率、逐行修复和退化；最初831/813/319条低先验S错误人口保持固定，另列独立输入覆盖。历史正确与累计修复新增错必须为0，不能用总体准确率或较低CE盖过S仍错。

B必须在匹配终点各角色每类不差于A，并且各角色S有额外实际修复，才支持新机制。如果低先验S仍没有修复，就不能宣称根本难例已解决。未完成固定终点只允许结论不足。训练侧问题关闭仍要求三个角色纯错误0、最后5个不同真实接受状态保持掌握及旧能力；完整N/M/S与独立来源泛化分别验收。

上述执行修复降低已知浪费与退化风险，不能保证本轮显著提升。若充分完成后仍无分类收益，停止这个配置，依据真实beta变化、读出变化与保护阻塞定位下一问题，避免再次追加同构训练。


### 原始来源 training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json

SHA256 `2888379069f0ac64c7ef684b3dd7aaf4521613f820ac675e79b1d559a0e9cb80`

{
  "status": "V169_concrete_execution_contract_candidate_not_authority",
  "protocol": "V169-three-role-paired-learnable-prior-bounded-training-v1",
  "execution_authority": false,
  "new_fit_permission": false,
  "entry": "training/v169_prior_pair_training_entry_v13.py",
  "schedule": {
    "accepted_updates": 20,
    "corrections": 2,
    "working_functions": 64,
    "backtracks": 8
  },
  "complete_parameter_widths": {
    "A": 1060832,
    "B": 1060833
  },
  "objective_policy": "strict_individual_M_S_actual_16eps_Armijo_and_full_original_row_protection",
  "roles": [
    0,
    1,
    2
  ],
  "arms": [
    "A",
    "B"
  ],
  "candidate": "B",
  "same_safe_initialization": true,
  "bootstrap_in20": true,
  "all_original_frequency_and_denominators_preserved": true,
  "permanent_full_guard_ledger": true,
  "full_guard_each_candidate": true,
  "fresh_derivatives_at_actual_parameter_point": true,
  "no_outer_selection": true,
  "no_new_confirmations_or_model_promotion": true,
  "unmatched_terminal_result": "inconclusive",
  "low_prior_slice_no_repair_cannot_claim_deep_error_solved": true,
  "prior_actual_costs": {
    "heads": 19178,
    "features": 19178,
    "original_class_derivatives": 368,
    "fixed_target_derivatives": 32,
    "margin_derivatives": 368,
    "all_complete_derivatives": 768,
    "fits_since_V159": 9,
    "accepted_updates_since_V159": 170
  },
  "inherited_observed_actions": {
    "V168-ACTUAL-SAFE-TINY-MARGIN-NOT-STABILITY": "Keep exact argmax and all original guards. Do not claim independent transfer, multiple-state repeat stability or mastery from this single tiny decision margin",
    "V168-S-RESTORATION-NOT-NEW-S-ERROR-REPAIR": "Report both baselines and remaining per-class errors. Do not count restoring original correctness as repairing V164 S errors or hide serious remaining errors with total loss",
    "V168-FINITE-DIAGNOSTIC-NOT-NEW-TRAINED-MODEL": "Keep cumulative19178heads/768derivatives, latest real trainingV164 and full deliveryV159. No second correction or fit under this contract; unchanged safe controls have no new gains",
    "V168-FROZEN-READOUT-BURDEN-NOT-GLOBAL-INFEASIBILITY": "Track deep prior margins, learned residuals, readout scale and actual repairs. Do not claim model-wide impossibility, missing information or solve classification using temperature calibration",
    "V168-ELIGIBLE-FOR-REGISTRATION-NOT-AUTOMATIC-TRAINING": "Register a new single-issue bounded training contract with complete protection, real accepted trajectory and deep-error progress. Preserve V164 capacity stops and failed budgets; do not rerun the old entry or substitute this result for pure-error-zero and five distinct mastered states"
  },
  "applicable_failure_registry_chain": [
    {
      "path": "training/review_policy/v168_observed_runtime_boundaries.json",
      "sha256": "bd4345f14dbfaf4da2b3fdbc1a6bfd2919453ead2e6fa0e38ef1576e88148e5e",
      "actions": {
        "V168-ACTUAL-SAFE-TINY-MARGIN-NOT-STABILITY": "Keep exact argmax and all original guards. Do not claim independent transfer, multiple-state repeat stability or mastery from this single tiny decision margin",
        "V168-S-RESTORATION-NOT-NEW-S-ERROR-REPAIR": "Report both baselines and remaining per-class errors. Do not count restoring original correctness as repairing V164 S errors or hide serious remaining errors with total loss",
        "V168-FINITE-DIAGNOSTIC-NOT-NEW-TRAINED-MODEL": "Keep cumulative19178heads/768derivatives, latest real trainingV164 and full deliveryV159. No second correction or fit under this contract; unchanged safe controls have no new gains",
        "V168-FROZEN-READOUT-BURDEN-NOT-GLOBAL-INFEASIBILITY": "Track deep prior margins, learned residuals, readout scale and actual repairs. Do not claim model-wide impossibility, missing information or solve classification using temperature calibration",
        "V168-ELIGIBLE-FOR-REGISTRATION-NOT-AUTOMATIC-TRAINING": "Register a new single-issue bounded training contract with complete protection, real accepted trajectory and deep-error progress. Preserve V164 capacity stops and failed budgets; do not rerun the old entry or substitute this result for pure-error-zero and five distinct mastered states"
      }
    },
    {
      "path": "training/review_policy/v167_observed_runtime_boundaries.json",
      "sha256": "46a6421b8907df551fd854253d6b453117f34c64af20bfe29430bb8bbc30d9f8",
      "actions": {
        "V167-CLOSED-ZERO-MARGIN-LOSES-ARGMAX-TIE": "Reject the candidate on exact original argmax and registered protection. A zero maximum negative margin is not classifier safety; do not forgive ties or change the classifier, tolerances, gold or acceptance retrospectively",
        "V167-ALL-BLOCKERS-COVERED-NOT-CAPACITY": "Keep complete function identities and trial-point gradients. Do not enlarge the25-function capacity to explain this known decision-boundary failure",
        "V167-TWO-CORRECTIONS-STOP": "Stop at two, retain all failed outputs and restore V164. Fixed-target Armijo and residual reduction permit only the already registered continuation, never override classification or authorize a third correction",
        "V167-PARTIAL-S-RECOVERY-NOT-JOINT-GAIN": "Report both baselines and per-class regressions. Preserve all original class masses, mixed rows, original gold and previous eight accepted repairs. Do not promote the lower total error count",
        "V167-SAFE-REPLAYS-NOT-NEW-ALL-ROLE-TRAINING-GATE": "Label the safe controls as unchanged replays. Keep all three roles and require a separately qualified prospectively sealed contract for any new actual diagnostic or training",
        "V167-RESTORED-MODEL-AND-CPU-PREVIEWS-NOT-TRAINING": "Keep cumulative19080heads/718derivatives without reset; latest real trained model remains V164 and latest full-population delivery remains V159. Keep CPU local previews separate from actual nonlinear candidates and full quality acceptance"
      }
    },
    {
      "path": "training/review_policy/v166_observed_runtime_boundaries_v2.json",
      "sha256": "859f5e2f74b25230907987f95566f9a896e2231241da73d5adb69ddc8e213d21",
      "actions": {
        "V166-COVERED-LINEAR-SAFE-FINITE-UNSAFE": "Keep actual original-row finite classification and retention gates authoritative. Preserve all five complete function identities and actual residuals; qualify a new bounded nonlinear correction method before any new official call. Do not claim covered linear constraints certify nonlinear safety",
        "V166-KNOWN-RESIDUAL-NOT-MISSING-CAPACITY": "Do not enlarge the function limit or add unrelated normals to explain these known-function failures. Evaluate predicted versus actual margins and distinguish local approximation error from future genuinely uncovered functions",
        "V166-CLASS-TRADEOFF-NOT-JOINT-GAIN": "Reject the actual candidate on any registered class or cumulative protection failure even when total errors or both fixed-target risks improve. Preserve complete original class masses, mixed rows and all earlier eight repairs",
        "V166-TWO-PASSES-NOT-THREE-ROLE-TRAINING-GATE": "Keep all roles and the original all-role gate. Do not discard role1, choose only successful roles or promote temporary candidates. A new mechanism or training range requires a separate prospectively reviewed contract and physical source seal",
        "V166-TEMPORARY-REPAIRS-NOT-PERMANENT-MODEL": "Preserve exact complete V164 parameters and original q/logq repeat policy on every failure or exception. Keep temporary candidate gains separate from submitted training gains, retain cumulative actual cost18784headsand618derivatives without budget reset, and keep latest full-population deliveryV159 until truly reverified"
      }
    },
    {
      "path": "training/review_policy/v165_observed_boundaries.json",
      "sha256": "ca9d3aae3f81eecceca54bbc042b93436d69a9725d940c2a29868f094e0863a6",
      "actions": {
        "V165-LESS-REGRESSION-IS-NOT-SAFE-GAIN": "Evaluate both actual origin and matched rejected control. Reject every original guard failure even if treatment has fewer errors than the rejected control; preserve origin checkpoint and original denominator",
        "V165-KNOWN-CONSTRAINT-SAFETY-NOT-FULL-SAFETY": "Check actual complete original population after every proposal; measured-function safety cannot bypass original-row class counts, cumulative repair or mastery checks. Bind and audit complete input identities of fresh blockers",
        "V165-FIXED-25-IS-ONLY-CURRENT-COVERAGE": "Use a new prospectively bound method and budget for adding fresh functions; never modify V164/V165 contracts, silently drop additional blockers, or treat new capacity exhaustion as global infeasibility"
      }
    },
    {
      "path": "training/review_policy/v164_root_failure_constraints.json",
      "sha256": "09c3a563ad61122bb0fedc2aa58f0b2b2956557048ff1487d3c0dd6c9858dfac",
      "actions": {
        "V164-HISTORICAL-EIGHT-NOT-NEW-GAIN": "Always report matched previous-round endpoint and original fixed-endpoint deltas separately; refuse to recount the eight old repairs as V164 gain",
        "V164-FUNCTION-CAP-NOT-INFEASIBILITY": "Preserve capped stop and distinguish resource truncation from lack of a qualified direction; do not raise a bound retrospectively or infer global model inability",
        "V164-PROTECT-DECISION-NOT-ORIGINAL-CONFIDENCE": "Specify exactly which margin is protected; any decision-floor variant still needs original complete finite class-count, prior-mastery and original-row protection checks; local vector qualification cannot authorize fit or promotion",
        "V164-READOUT-BOUND-DOES-NOT-EQUAL-CLASSIFICATION": "Treat fixed-readout override bounds as state-dependent necessary conditions, not permanent capacity or learning success; score real class repairs, regressions and unresolved margins",
        "V164-NO-LOSS-FOR-CLASS-TRADEOFF": "Require each original-class count and registered correct-row retention separately; neither aggregate error reduction nor lower two-class target loss can override rejection"
      }
    }
  ],
  "role_caps": {
    "0": {
      "heads": 9940,
      "features": 9940,
      "fixed_target_derivatives": 84,
      "margin_derivatives": 4864,
      "QP": 57,
      "proposals": 191,
      "updates": 20,
      "fits": 1
    },
    "1": {
      "heads": 8284,
      "features": 8284,
      "fixed_target_derivatives": 84,
      "margin_derivatives": 4864,
      "QP": 57,
      "proposals": 191,
      "updates": 20,
      "fits": 1
    },
    "2": {
      "heads": 9940,
      "features": 9940,
      "fixed_target_derivatives": 84,
      "margin_derivatives": 4864,
      "QP": 57,
      "proposals": 191,
      "updates": 20,
      "fits": 1
    }
  },
  "new_caps": {
    "heads": 56328,
    "features": 56328,
    "fixed_target_derivatives": 504,
    "margin_derivatives": 29184,
    "QP": 342,
    "proposals": 1146,
    "fits": 6,
    "updates": 120,
    "original_class_derivatives": 0,
    "all_complete_derivatives": 29688
  },
  "future_cumulative_caps": {
    "heads": 75506,
    "original_class_derivatives": 368,
    "fixed_target_derivatives": 536,
    "margin_derivatives": 29552,
    "all_complete_derivatives": 30456,
    "fits_since_V159": 15,
    "updates_since_V159": 290
  },
  "full_task_quality_acceptance": false,
  "model_promoted": false,
  "resource_review_path": "artifacts/v169_factorized_resource_budget_v7_20261002/review.json",
  "resources": {
    "full_run_worst_storage_bytes": 12136488300,
    "fixed_free_disk_reserve_bytes": 2147483648,
    "minimum_free_disk_start_bytes": 14283971948,
    "minimum_free_RAM_bytes": 6442450944,
    "minimum_free_GPU_bytes": 536870912,
    "minimum_free_commit_bytes": 6710886400
  },
  "batch_global_fault_stops_remaining_fits": true,
  "same_point_final_replay_required": true,
  "training_issue_mastery": {
    "all_roles_pure_errors_zero": true,
    "last_five_distinct_accepted_states_mastered": true,
    "all_registered_old_and_accepted_repairs_preserved": true
  },
  "B_support_requires": {
    "fixed20_matched_all_three_roles": true,
    "no_more_M_S_errors_each_role": true,
    "extra_S_repair_each_role": true,
    "all_initial_low_prior_S_slices_reported": true,
    "no_low_prior_repair_means_deep_problem_unresolved": true
  },
  "full_N_M_S_quality_and_unseen_source_generalization_separate": true,
  "viewed_roles_are_development_not_blind": true,
  "zero_step_is_registered_targets_point0_before_bootstrap_no_extra_heads": true,
  "no_beta_clipping_or_exponential_fallback": true,
  "source_sha256": {
    "training/v169_prepare_execution_contract_candidate_v3.py": "66767c38602240c8ea042751e120b8598d9d08d413c82ee6143d31c743475c50",
    "artifacts/v169_factorized_resource_budget_v7_20261002/review.json": "7ffc89a38e96d0f894af4ae3eade28135614dce67016cd7bc6643da2cbd85ad2",
    "training/v169_prior_pair_training_entry_v13.py": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
    "training/v169_pair_execution_review.py": "e18abb8aed236253b4c54ee2a35c563f606f84f1c1c74f35af556a8150d95f14",
    "training/v169_pair_execution_review_v2.py": "eb601c14d3c3717415664de96feb1c5c7930b3d68c1e132729c2bfcb99476a2b",
    "training/v169_pair_execution_review_v3.py": "e439b5b2ca0e083e6aed7dbb59de7725b090f542d919aaeeea1ef1383f4e65e7",
    "training/v169_pair_execution_review_v4.py": "6ec79675ad75d26ab2e6bcce122f0ea9cc3dcc034d2e5ee2bdc1264099741c99",
    "training/v169_dynamic_trial_restoration_v2.py": "1f2bf54000cb04a5254628839d284e94f9eed0485fdac41c323644386bd56085",
    "training/v169_working_joint_restoration_v2.py": "ace824d2ae8ed36ca53a589877839cb640e5c427e49fef9e61803fbfb624c9e6",
    "docs/EXPERIMENT_REVIEW_RULES.md": "00de5f109bd5fbd5060bd8138c4fb5c4f9a3c59cb65cff9f8eeb09292520ab41",
    "training/review_policy/history_cases.json": "beee71dcd8a4bc85b457e82b3efbb7790f3c348d60d7421633949a3e8315b347",
    "docs/V169_INCREMENTAL_RESEARCH_AND_TRAINING_REVIEW_20261002.md": "42c566eb2c7c09b1999ed57c2fb001ac27db0b1ab2d03582c583ee1a86bc563c",
    "docs/V169_PRETRAINING_INDEPENDENT_FINDINGS_20261002.md": "48774a5b99d1c11ff7155120dad03d55aa218ca461a7c41c0b7e2e1ee09100d7",
    "training/review_policy/v169_learnable_prior_pair_draft.json": "109272638d24ff36eb9b9f8a0d301487d2b943c19a57a72947fba54c84cc18e5",
    "training/review_policy/v168_observed_runtime_boundaries.json": "bd4345f14dbfaf4da2b3fdbc1a6bfd2919453ead2e6fa0e38ef1576e88148e5e",
    "training/review_policy/v167_observed_runtime_boundaries.json": "46a6421b8907df551fd854253d6b453117f34c64af20bfe29430bb8bbc30d9f8",
    "training/review_policy/v166_observed_runtime_boundaries_v2.json": "859f5e2f74b25230907987f95566f9a896e2231241da73d5adb69ddc8e213d21",
    "training/review_policy/v165_observed_boundaries.json": "ca9d3aae3f81eecceca54bbc042b93436d69a9725d940c2a29868f094e0863a6",
    "training/review_policy/v164_root_failure_constraints.json": "09c3a563ad61122bb0fedc2aa58f0b2b2956557048ff1487d3c0dd6c9858dfac"
  }
}


### 原始来源 artifacts/v169_factorized_resource_budget_v7_20261002/review.json

SHA256 `7ffc89a38e96d0f894af4ae3eade28135614dce67016cd7bc6643da2cbd85ad2`

{
  "status": "V169_v13_factorized_whole_run_resource_candidate_runtime_caps_implemented_pending_independent_review",
  "entry_sha256": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
  "role_details": {
    "0": {
      "target_coordinate_upper": {
        "1": 6177,
        "2": 6145
      },
      "original_scopes": {
        "OOF": {
          "rows": 92337,
          "arrow_logical_bytes": 29951831,
          "columns": 24
        },
        "deployment": {
          "rows": 92337,
          "arrow_logical_bytes": 29951831,
          "columns": 24
        }
      }
    },
    "1": {
      "target_coordinate_upper": {
        "1": 6209,
        "2": 6097
      },
      "original_scopes": {
        "OOF": {
          "rows": 40957,
          "arrow_logical_bytes": 13285434,
          "columns": 24
        },
        "deployment": {
          "rows": 40957,
          "arrow_logical_bytes": 13285434,
          "columns": 24
        }
      }
    },
    "2": {
      "target_coordinate_upper": {
        "1": 5841,
        "2": 6129
      },
      "original_scopes": {
        "OOF": {
          "rows": 92320,
          "arrow_logical_bytes": 29946300,
          "columns": 24
        },
        "deployment": {
          "rows": 92320,
          "arrow_logical_bytes": 29946300,
          "columns": 24
        }
      }
    }
  },
  "prepared_callgraph": {
    "heads": 56328,
    "features": 56328,
    "fixed_target_derivatives": 504,
    "margin_derivatives": 29184,
    "QP": 342,
    "proposals": 1146,
    "fits": 6,
    "updates": 120,
    "original_class_derivatives": 0,
    "all_complete_derivatives": 29688
  },
  "storage_components_upper_bytes": {
    "target_vectors": 54198144,
    "margin_vectors": 1928011776,
    "QP_vectors": 41342400,
    "margin_raw_outputs": 3107979264,
    "chunk_ids": 240943104,
    "margin_references": 239075328,
    "normal_identity_JSON": 239075328,
    "candidate_outputs": 2489808768,
    "candidate_masks": 267941676,
    "target_outputs": 549561600,
    "accepted_outputs": 288769680,
    "checkpoints": 1077577200,
    "final_tables": 671088640,
    "other_metadata_and_logs": 135809024,
    "allocation_and_directories": 671088640,
    "emergency_dense_vector_and_failure": 134217728
  },
  "metadata_components_upper_bytes": {
    "candidate_proofs": 18776064,
    "candidate_recipes": 9388032,
    "candidate_original_reference": 4694016,
    "line_search": 3735552,
    "common_QP": 7471104,
    "correction_QP": 7471104,
    "target_repeats": 2064384,
    "target_identities": 4128768,
    "commits": 491520,
    "accepted_quality": 1966080,
    "accepted_reference": 1966080,
    "initial_protection": 393216,
    "final_replay": 98304,
    "fit_results": 3145728,
    "fit_counts": 196608,
    "pair_results": 12288,
    "batch_results": 8388608,
    "failure_receipts": 1179648,
    "head_and_feature_log_lines": 28839936,
    "derivative_log_lines": 30400512,
    "QP_log_lines": 350208,
    "proposal_update_fit_log_lines": 651264
  },
  "maximum_files": 128000,
  "maximum_directories": 17000,
  "filesystem_padding_bound_formula": "128000*4096+17000*8192 < 640MiB",
  "accepted_full_masks_three_copies_counted": true,
  "resources": {
    "full_run_worst_storage_bytes": 12136488300,
    "fixed_free_disk_reserve_bytes": 2147483648,
    "minimum_free_disk_start_bytes": 14283971948,
    "minimum_free_RAM_bytes": 6442450944,
    "minimum_free_GPU_bytes": 536870912,
    "minimum_free_commit_bytes": 6710886400
  },
  "RAM_complete_live_phase_accounting_upper_bytes": 5936980096,
  "QP_eight_matrix_accounting_upper_bytes": 4480958592,
  "actual_RAM_live_buffer_review": "artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json",
  "commit_increment_plus_eight_matrix_and128MiB_bytes": 6568317056,
  "minimum_commit_256MiB_rounding_bytes": 6710886400,
  "physical_RAM_prerequisite_retained6GiB": true,
  "same_full_SVD_inputs_one_matrix_copy_removed_qualification": "artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json",
  "GPU_tensor_accounting_upper_bytes": 303558784,
  "snapshot": {
    "free_disk_bytes": 14351208448,
    "free_RAM_bytes": 6236590080,
    "free_GPU_bytes": 8342470656,
    "GPU_query": "7956, 8188, NVIDIA GeForce RTX 4060 Laptop GPU"
  },
  "current_disk_prerequisite_met": true,
  "current_RAM_prerequisite_met": false,
  "current_GPU_prerequisite_met": true,
  "worst_case_does_not_assume_any_same_point_output_reuse": true,
  "full_vector_bits_and_full_original_row_frequency_preserved": true,
  "initial_whole_run_check_then_fixed_reserve_each_fit": true,
  "JSON_and_final_table_caps_runtime_enforced": true,
  "per_operation_fixed_disk_reserve_enforced": true,
  "RAM_workspace_estimate_requires_backend_independent_review": true,
  "official_calls": 0,
  "fits": 0,
  "permanent_updates": 0,
  "execution_authority": false,
  "new_fit_permission": false,
  "supports_physical_seal": false,
  "source_sha256": {
    "training/v169_factorized_resource_budget_v7.py": "f803ae386be2fc3fbff639a49fe631eddbb17b757317485f4f6bd7a8f7737a41",
    "training/v169_prior_pair_training_entry_v13.py": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
    "artifacts/v124_header_trial_20260929/B_header_ASA.npz": "88743fb2ea387f9aa1ad72de92a85b18025da449afbfaeeca92c98f91d3814ce",
    "training/v169_structural_vector_storage.py": "3384dbefab6a9f6a04211907ac56eb9bab809c56de9fc6b45957e4f9ae654c8e",
    "training/v169_measurement_output_references_v2.py": "585341206d6d7b5508b0e2e07ebd573089738925c96202e3e20d651b67deec2a",
    "training/v169_factorized_row_evidence.py": "a8550fbdd45ada6b7ecc604b5beed86432ba92ac3a7684d848d84008243cd414",
    "artifacts/v169_saved_budget_scope_and_callgraph_v2_20261002/review.json": "d41ef7caf83a87abd37f6d0cd04a56a70c6cb8d3b1cfe1286152f97ea1279552",
    "artifacts/v161_independent_frozen_error_cohort_review_20261002/role0/target_original_counts.npy": "15594bc567c7cdffac50339c1cd2cd286408a5ec18dcb16306e4332ddd6f1b4c",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/endpoint/OOF_original_rows.parquet": "e8d38d97f78a49e00e7365e16e30e6b33d00060b18d52ad02a172f8a64f303a3",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/endpoint/deployment_original_rows.parquet": "3ba1071f3e75263452eb3b6ea4e45fa755217199d75060184e983b960c1487e3",
    "artifacts/v161_independent_frozen_error_cohort_review_20261002/role1/target_original_counts.npy": "1f3aa89f201f8391250af6fbf8146b12fc1ff3e3bbc07f6a1bd7c2de5894b603",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/endpoint/OOF_original_rows.parquet": "4f948c69359aac4ab62c3c7e82bdf7f2e48203556ac98585f0353f53a9ce6771",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/endpoint/deployment_original_rows.parquet": "49bb7365ae0cfe12aa003f5185849e2a9889057190a41cc3ad1247369e85b084",
    "artifacts/v161_independent_frozen_error_cohort_review_20261002/role2/target_original_counts.npy": "4cb7c2600e997850102c178ed3a6d76bb64e24067fe642ff1f01e6da53e70661",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/endpoint/OOF_original_rows.parquet": "fd10479c37471b5f728b444243853c52c31cefe75133c4e6a2f635320d232a79",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/endpoint/deployment_original_rows.parquet": "e7dc57e27466e1d41b08d78f02e03545ee914b93869aaba0edbd17eee45e82d5",
    "artifacts/v169_saved_storage_filecount_review_20261002/review.json": "d71e3f9b2afee7d47096aa8f1a391a5a786385273abfeb8d03744bf82d3dc073",
    "artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json": "61b74d70e17025eaaab4fd9f40b5ece5d1485ac64011ae5c379231aadf091d64",
    "artifacts/v169_cuda_host_buffer_qualification_v2_20261002/qualification.json": "e1a41403cffd0336a580bb35272b4e30e04953ab5051d4ddd89721ad5bf3272c",
    "artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json": "ed6c95c458190ff93d1d8a90753fa8e3fe13965e9772ccf69c26669aa467a72e",
    "training/v169_pair_execution_review_v4.py": "6ec79675ad75d26ab2e6bcce122f0ea9cc3dcc034d2e5ee2bdc1264099741c99"
  }
}


### 原始来源 artifacts/v169_actual_backend_synthetic_qualification_v4_20261002/qualification.json

SHA256 `a05e4e57a55d713d8e9eebe162284b6e7f45305bdeef321f2dd82c7c4c31627e`

{
  "status": "V169_actual_complete_CPU_backend_synthetic_wiring_qualified",
  "cases": [
    {
      "arm": "A",
      "complete_parameters": 1060832,
      "actual_synthetic_counter": {
        "head_attempts": 74,
        "head_completed": 74,
        "feature_attempts": 74,
        "feature_completed": 74,
        "gradient_attempts": 8,
        "gradient_completed": 8
      },
      "margin_derivatives": 2,
      "committed_state": 2,
      "full_class_quality": {
        "0": {
          "original_rows": 0,
          "errors": 0,
          "precision": null,
          "recall": null,
          "repairs_vs_V164": 0,
          "new_errors_vs_V164": 0
        },
        "1": {
          "original_rows": 2,
          "errors": 1,
          "precision": 0.5,
          "recall": 0.5,
          "repairs_vs_V164": 0,
          "new_errors_vs_V164": 0
        },
        "2": {
          "original_rows": 2,
          "errors": 1,
          "precision": 0.5,
          "recall": 0.5,
          "repairs_vs_V164": 0,
          "new_errors_vs_V164": 0
        }
      },
      "trial_and_failed_commit_restoration_exact": true,
      "last_commit_output_drift_refused": true,
      "missing_last_observation_refused": true,
      "hooks_and_log_closed": true
    },
    {
      "arm": "B",
      "complete_parameters": 1060833,
      "actual_synthetic_counter": {
        "head_attempts": 74,
        "head_completed": 74,
        "feature_attempts": 74,
        "feature_completed": 74,
        "gradient_attempts": 8,
        "gradient_completed": 8
      },
      "margin_derivatives": 2,
      "committed_state": 2,
      "full_class_quality": {
        "0": {
          "original_rows": 0,
          "errors": 0,
          "precision": null,
          "recall": null,
          "repairs_vs_V164": 0,
          "new_errors_vs_V164": 0
        },
        "1": {
          "original_rows": 2,
          "errors": 1,
          "precision": 0.5,
          "recall": 0.5,
          "repairs_vs_V164": 0,
          "new_errors_vs_V164": 0
        },
        "2": {
          "original_rows": 2,
          "errors": 1,
          "precision": 0.5,
          "recall": 0.5,
          "repairs_vs_V164": 0,
          "new_errors_vs_V164": 0
        }
      },
      "trial_and_failed_commit_restoration_exact": true,
      "last_commit_output_drift_refused": true,
      "missing_last_observation_refused": true,
      "hooks_and_log_closed": true
    }
  ],
  "actual_global_batch_fault_started_fits": [
    [
      0,
      "A"
    ]
  ],
  "bootstrap_and_trial_Jacobians_actual_backend": true,
  "full_target_gradients_and_q_logq_actual": true,
  "complete_model_parameters_preserved": true,
  "legacy_multi_role_retention_and_initial_frozen_helper_stubbed_only_in_synthetic_fixture": true,
  "not_official_old_skill_acceptance": true,
  "actual_official_zero_step_still_required": true,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "execution_authority": false,
  "source_sha256": {
    "training/v169_actual_backend_synthetic_qualification_v4.py": "1a288ca5bb5429512b4ebc00e7c5b8530617b8f09b00a28557abcf70f8365826",
    "training/v169_prior_pair_training_entry_v13.py": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
    "training/v169_pair_lifecycle_v3.py": "af2c306f01fd5b8b9f46afbf5ed7bc591c0fd550eea0be3d062c260331706d65",
    "training/v169_prior_pair_model.py": "0d563d82ca036f489dbd643f0edebabd3687153a197d11c668dc261ac6c08e60",
    "training/v169_pair_execution_review_v4.py": "6ec79675ad75d26ab2e6bcce122f0ea9cc3dcc034d2e5ee2bdc1264099741c99"
  }
}


## 完整依赖与资格清单摘要

```json
{
  "status": "V169_final_v13_full_dependency_and_qualification_bundle_ready_for_independent_preseal_review",
  "entry_path": "training/v169_prior_pair_training_entry_v13.py",
  "entry_sha256": "7c342dbca1728a44ca7cfd029e42cc4a9ecb8d466417396c450559a2c98f1260",
  "candidate_contract_path": "training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json",
  "candidate_contract_sha256": "2888379069f0ac64c7ef684b3dd7aaf4521613f820ac675e79b1d559a0e9cb80",
  "qualifications": [
    {
      "path": "artifacts/v169_pair_model_storage_qualification_20261002/qualification.json",
      "sha256": "2e5bd564633d3fd87e5e20ee54a33fffd816a9e8dd0db289f7eb7772ba693dcf",
      "status": "V169_complete_CPU_synthetic_model_parameter_repeat_strict_objective_and_lossless_storage_qualified"
    },
    {
      "path": "artifacts/v169_pair_lifecycle_qualification_v2_20261002/qualification.json",
      "sha256": "639c30f5e611c2864771e8aab6ab5473ed0145c454aede82207a0de8584a2f71",
      "status": "V169_full_parameter_scripted_lifecycle_v2_commit_terminal_failure_and_counter_cleanup_qualified"
    },
    {
      "path": "artifacts/v169_saved_state_quality_qualification_20261002/qualification.json",
      "sha256": "1d3c303d37edc3f7ef285d1fe161490393061b1968273c478feb43512f251152",
      "status": "V169_saved_actual_original_gold_full_class_and_fixed_initial_S_reporting_qualified"
    },
    {
      "path": "artifacts/v169_working_solver_full_size_qualification_20261002/qualification.json",
      "sha256": "f1381f7ae76abaa091959b21956b9e9d498c3fbcf58a05af28697b4321837827",
      "status": "V169_full_size_A_B64_current_function_CPU_QPs_and_old_arithmetic_identity_qualified"
    },
    {
      "path": "artifacts/v169_factorized_storage_qualification_v2_20261002/qualification.json",
      "sha256": "56c260ce43b6ba27a051fe8faeb73f5b43f41c29f45d9bf79818cd3601ef543b",
      "status": "V169_saved_full_vectors_signed_zero_subnormals_recipes_and_all_original_rows_bit_exact_qualified"
    },
    {
      "path": "artifacts/v169_measurement_references_qualification_20261002/qualification.json",
      "sha256": "b7b313d19dda5e423b118210d25887ccde30a7fc11afa0d060c61456c0d7cfa6",
      "status": "V169_exact_same_point_references_one_ulp_fallback_and_source_change_refusal_qualified"
    },
    {
      "path": "artifacts/v169_dataframe_and_current_guard_qualification_20261002/qualification.json",
      "sha256": "daa638da42e59d80a33b9c000108f6ff9eb1b560185b82c039d897a03c603525",
      "status": "V169_real_DataFrame_and_all206_current_mixed_correct_guard_interfaces_qualified"
    },
    {
      "path": "artifacts/v169_actual_backend_synthetic_qualification_v4_20261002/qualification.json",
      "sha256": "a05e4e57a55d713d8e9eebe162284b6e7f45305bdeef321f2dd82c7c4c31627e",
      "status": "V169_actual_complete_CPU_backend_synthetic_wiring_qualified"
    },
    {
      "path": "artifacts/v169_saved_correction_storage_qualification_20261002/qualification.json",
      "sha256": "2ea7d594e68b526f1fa496add11d12dfac9a99d7b6132da2a75b16bb9a421e07",
      "status": "V169_actual_V168_correction_and_displacement_prefix_tiny_bits_qualified"
    },
    {
      "path": "artifacts/v169_execution_contract_candidate_v3_20261002/qualification.json",
      "sha256": "0e74a14cf37ec8860517f530b06bc36a7f189a2281f9de267bbec908be8a7e01",
      "status": "V169_concrete_contract_and_dedicated_negative_scope_checks_passed_not_execution"
    },
    {
      "path": "artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json",
      "sha256": "ed6c95c458190ff93d1d8a90753fa8e3fe13965e9772ccf69c26669aa467a72e",
      "status": "V169_same_full_SVD_inputs_full_displacement_and_original_unit_certificates_byte_exact_with_one_matrix_copy_removed"
    },
    {
      "path": "artifacts/v169_cuda_host_buffer_qualification_v2_20261002/qualification.json",
      "sha256": "e1a41403cffd0336a580bb35272b4e30e04953ab5051d4ddd89721ad5bf3272c",
      "status": "V169_complete_A_B_max_batch_CUDA_synthetic_host_and_device_buffers_measured"
    },
    {
      "path": "artifacts/v169_physical_and_commit_policy_qualification_20261002/qualification.json",
      "sha256": "c4625380a0150bf602909bdc873177ba6211a01dc3fb0098e231e0f6ec38fba2",
      "status": "V169_separate_physical_RAM_and_available_commit_boundary_policy_qualified"
    }
  ],
  "resource_review_path": "artifacts/v169_factorized_resource_budget_v7_20261002/review.json",
  "resources": {
    "full_run_worst_storage_bytes": 12136488300,
    "fixed_free_disk_reserve_bytes": 2147483648,
    "minimum_free_disk_start_bytes": 14283971948,
    "minimum_free_RAM_bytes": 6442450944,
    "minimum_free_GPU_bytes": 536870912,
    "minimum_free_commit_bytes": 6710886400
  },
  "applicable_failure_registry_chain": [
    {
      "path": "training/review_policy/v168_observed_runtime_boundaries.json",
      "sha256": "bd4345f14dbfaf4da2b3fdbc1a6bfd2919453ead2e6fa0e38ef1576e88148e5e",
      "actions": {
        "V168-ACTUAL-SAFE-TINY-MARGIN-NOT-STABILITY": "Keep exact argmax and all original guards. Do not claim independent transfer, multiple-state repeat stability or mastery from this single tiny decision margin",
        "V168-S-RESTORATION-NOT-NEW-S-ERROR-REPAIR": "Report both baselines and remaining per-class errors. Do not count restoring original correctness as repairing V164 S errors or hide serious remaining errors with total loss",
        "V168-FINITE-DIAGNOSTIC-NOT-NEW-TRAINED-MODEL": "Keep cumulative19178heads/768derivatives, latest real trainingV164 and full deliveryV159. No second correction or fit under this contract; unchanged safe controls have no new gains",
        "V168-FROZEN-READOUT-BURDEN-NOT-GLOBAL-INFEASIBILITY": "Track deep prior margins, learned residuals, readout scale and actual repairs. Do not claim model-wide impossibility, missing information or solve classification using temperature calibration",
        "V168-ELIGIBLE-FOR-REGISTRATION-NOT-AUTOMATIC-TRAINING": "Register a new single-issue bounded training contract with complete protection, real accepted trajectory and deep-error progress. Preserve V164 capacity stops and failed budgets; do not rerun the old entry or substitute this result for pure-error-zero and five distinct mastered states"
      }
    },
    {
      "path": "training/review_policy/v167_observed_runtime_boundaries.json",
      "sha256": "46a6421b8907df551fd854253d6b453117f34c64af20bfe29430bb8bbc30d9f8",
      "actions": {
        "V167-CLOSED-ZERO-MARGIN-LOSES-ARGMAX-TIE": "Reject the candidate on exact original argmax and registered protection. A zero maximum negative margin is not classifier safety; do not forgive ties or change the classifier, tolerances, gold or acceptance retrospectively",
        "V167-ALL-BLOCKERS-COVERED-NOT-CAPACITY": "Keep complete function identities and trial-point gradients. Do not enlarge the25-function capacity to explain this known decision-boundary failure",
        "V167-TWO-CORRECTIONS-STOP": "Stop at two, retain all failed outputs and restore V164. Fixed-target Armijo and residual reduction permit only the already registered continuation, never override classification or authorize a third correction",
        "V167-PARTIAL-S-RECOVERY-NOT-JOINT-GAIN": "Report both baselines and per-class regressions. Preserve all original class masses, mixed rows, original gold and previous eight accepted repairs. Do not promote the lower total error count",
        "V167-SAFE-REPLAYS-NOT-NEW-ALL-ROLE-TRAINING-GATE": "Label the safe controls as unchanged replays. Keep all three roles and require a separately qualified prospectively sealed contract for any new actual diagnostic or training",
        "V167-RESTORED-MODEL-AND-CPU-PREVIEWS-NOT-TRAINING": "Keep cumulative19080heads/718derivatives without reset; latest real trained model remains V164 and latest full-population delivery remains V159. Keep CPU local previews separate from actual nonlinear candidates and full quality acceptance"
      }
    },
    {
      "path": "training/review_policy/v166_observed_runtime_boundaries_v2.json",
      "sha256": "859f5e2f74b25230907987f95566f9a896e2231241da73d5adb69ddc8e213d21",
      "actions": {
        "V166-COVERED-LINEAR-SAFE-FINITE-UNSAFE": "Keep actual original-row finite classification and retention gates authoritative. Preserve all five complete function identities and actual residuals; qualify a new bounded nonlinear correction method before any new official call. Do not claim covered linear constraints certify nonlinear safety",
        "V166-KNOWN-RESIDUAL-NOT-MISSING-CAPACITY": "Do not enlarge the function limit or add unrelated normals to explain these known-function failures. Evaluate predicted versus actual margins and distinguish local approximation error from future genuinely uncovered functions",
        "V166-CLASS-TRADEOFF-NOT-JOINT-GAIN": "Reject the actual candidate on any registered class or cumulative protection failure even when total errors or both fixed-target risks improve. Preserve complete original class masses, mixed rows and all earlier eight repairs",
        "V166-TWO-PASSES-NOT-THREE-ROLE-TRAINING-GATE": "Keep all roles and the original all-role gate. Do not discard role1, choose only successful roles or promote temporary candidates. A new mechanism or training range requires a separate prospectively reviewed contract and physical source seal",
        "V166-TEMPORARY-REPAIRS-NOT-PERMANENT-MODEL": "Preserve exact complete V164 parameters and original q/logq repeat policy on every failure or exception. Keep temporary candidate gains separate from submitted training gains, retain cumulative actual cost18784headsand618derivatives without budget reset, and keep latest full-population deliveryV159 until truly reverified"
      }
    },
    {
      "path": "training/review_policy/v165_observed_boundaries.json",
      "sha256": "ca9d3aae3f81eecceca54bbc042b93436d69a9725d940c2a29868f094e0863a6",
      "actions": {
        "V165-LESS-REGRESSION-IS-NOT-SAFE-GAIN": "Evaluate both actual origin and matched rejected control. Reject every original guard failure even if treatment has fewer errors than the rejected control; preserve origin checkpoint and original denominator",
        "V165-KNOWN-CONSTRAINT-SAFETY-NOT-FULL-SAFETY": "Check actual complete original population after every proposal; measured-function safety cannot bypass original-row class counts, cumulative repair or mastery checks. Bind and audit complete input identities of fresh blockers",
        "V165-FIXED-25-IS-ONLY-CURRENT-COVERAGE": "Use a new prospectively bound method and budget for adding fresh functions; never modify V164/V165 contracts, silently drop additional blockers, or treat new capacity exhaustion as global infeasibility"
      }
    },
    {
      "path": "training/review_policy/v164_root_failure_constraints.json",
      "sha256": "09c3a563ad61122bb0fedc2aa58f0b2b2956557048ff1487d3c0dd6c9858dfac",
      "actions": {
        "V164-HISTORICAL-EIGHT-NOT-NEW-GAIN": "Always report matched previous-round endpoint and original fixed-endpoint deltas separately; refuse to recount the eight old repairs as V164 gain",
        "V164-FUNCTION-CAP-NOT-INFEASIBILITY": "Preserve capped stop and distinguish resource truncation from lack of a qualified direction; do not raise a bound retrospectively or infer global model inability",
        "V164-PROTECT-DECISION-NOT-ORIGINAL-CONFIDENCE": "Specify exactly which margin is protected; any decision-floor variant still needs original complete finite class-count, prior-mastery and original-row protection checks; local vector qualification cannot authorize fit or promotion",
        "V164-READOUT-BOUND-DOES-NOT-EQUAL-CLASSIFICATION": "Treat fixed-readout override bounds as state-dependent necessary conditions, not permanent capacity or learning success; score real class repairs, regressions and unresolved margins",
        "V164-NO-LOSS-FOR-CLASS-TRADEOFF": "Require each original-class count and registered correct-row retention separately; neither aggregate error reduction nor lower two-class target loss can override rejection"
      }
    }
  ],
  "new_caps": {
    "heads": 56328,
    "features": 56328,
    "fixed_target_derivatives": 504,
    "margin_derivatives": 29184,
    "QP": 342,
    "proposals": 1146,
    "fits": 6,
    "updates": 120,
    "original_class_derivatives": 0,
    "all_complete_derivatives": 29688
  },
  "prior_actual_costs": {
    "heads": 19178,
    "features": 19178,
    "original_class_derivatives": 368,
    "fixed_target_derivatives": 32,
    "margin_derivatives": 368,
    "all_complete_derivatives": 768,
    "fits_since_V159": 9,
    "accepted_updates_since_V159": 170
  },
  "future_cumulative_caps": {
    "heads": 75506,
    "original_class_derivatives": 368,
    "fixed_target_derivatives": 536,
    "margin_derivatives": 29552,
    "all_complete_derivatives": 30456,
    "fits_since_V159": 15,
    "updates_since_V159": 290
  },
  "AST_recursive_training_module_paths": [
    "training/experiment_review.py",
    "training/experiment_review_v159_v3.py",
    "training/run_v75.py",
    "training/soc_v32_prepare.py",
    "training/soc_v3_prepare.py",
    "training/v101_prepare.py",
    "training/v101_select.py",
    "training/v104_phase_a.py",
    "training/v104_phase_b.py",
    "training/v107_matched_training.py",
    "training/v116_preflight.py",
    "training/v124_experiment_review.py",
    "training/v125_evaluate.py",
    "training/v125_experiment_review.py",
    "training/v125_model.py",
    "training/v130_verify_learning_review.py",
    "training/v131_common.py",
    "training/v131_evaluate.py",
    "training/v131_model.py",
    "training/v133_verify_targeted_plan.py",
    "training/v135_evaluate.py",
    "training/v135_model.py",
    "training/v135_runtime.py",
    "training/v136_verify_design.py",
    "training/v137_issue_guard.py",
    "training/v138_closeout.py",
    "training/v138_readout.py",
    "training/v138_retention_check.py",
    "training/v138_runtime.py",
    "training/v138_train.py",
    "training/v140_retention_check.py",
    "training/v142_retention_check.py",
    "training/v158_fusion_contract.py",
    "training/v158_nested_runtime_v2.py",
    "training/v159_boundary_evaluate_v4.py",
    "training/v159_boundary_runtime_v4.py",
    "training/v159_boundary_train_v4.py",
    "training/v159_class_direction.py",
    "training/v159_current_input_boundary.py",
    "training/v159_current_input_boundary_v3.py",
    "training/v159_float64_repeat_policy_v2.py",
    "training/v159_mgda_synthetic_qualification.py",
    "training/v160_active_margin_direction_v5.py",
    "training/v160_cached_polished_direction_finite_probe.py",
    "training/v160_fixed_endpoint_diagnostic_v3.py",
    "training/v160_independent_saved_direction_certificate.py",
    "training/v160_margin_normal.py",
    "training/v160_saved_vector_numeric_polish.py",
    "training/v161_fixed_error_endpoint_diagnostic_v2.py",
    "training/v161_fixed_pure_error_risk.py",
    "training/v163_fixed_endpoint_one_sided_restoration_v2.py",
    "training/v163_one_sided_joint_restoration.py",
    "training/v165_execution_review.py",
    "training/v165_fixed_endpoint_decision_floor_diagnostic.py",
    "training/v169_current_correct_context.py",
    "training/v169_dynamic_trial_restoration_v2.py",
    "training/v169_factorized_row_evidence.py",
    "training/v169_measurement_output_references_v2.py",
    "training/v169_pair_execution_review.py",
    "training/v169_pair_execution_review_v2.py",
    "training/v169_pair_execution_review_v3.py",
    "training/v169_pair_execution_review_v4.py",
    "training/v169_pair_lifecycle.py",
    "training/v169_pair_lifecycle_v3.py",
    "training/v169_prepare_full_preseal_bundle_v4.py",
    "training/v169_prior_pair_model.py",
    "training/v169_prior_pair_training_entry_v13.py",
    "training/v169_saved_state_quality.py",
    "training/v169_structural_vector_storage.py",
    "training/v169_working_joint_restoration_v2.py",
    "training/v32_features.py",
    "training/v32_finalize.py",
    "training/v331_prepare.py",
    "training/v351_safeguards.py",
    "training/v36_learning.py",
    "training/v36_representation.py",
    "training/v37_learning.py",
    "training/v37_representation.py",
    "training/v38_learning.py",
    "training/v38_representation.py",
    "training/v39_core.py",
    "training/v48_input.py",
    "training/v75_corrective.py",
    "training/v75_metadata.py",
    "training/v75_views.py",
    "training/v78_denial_adapter.py",
    "training/v78_denial_adapter_v4.py",
    "training/v79_execute.py",
    "training/v81_training_contract.py",
    "training/v82_capacity.py",
    "training/v85_protection.py",
    "training/v89_common.py",
    "training/v99_normalization_feasibility.py"
  ],
  "inherited_V168_physical_bindings_preserved": 17970,
  "physical_dependency_files": 18767,
  "python_version": "3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]",
  "package_versions": {
    "torch": "2.7.1+cu128",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "pyarrow": "19.0.1"
  },
  "no_mutable_publication_runtime_dependency": true,
  "actual_official_zero_step_and_full_runtime_storage_support_still_required": true,
  "RAM_workspace_estimate_and_whole_storage_accounting_require_independent_review": true,
  "supports_physical_seal": false,
  "execution_authority": false,
  "new_fit_permission": false,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "previous_immutable_bundle_path": "artifacts/v169_full_preseal_bundle_v4_20261002/bundle.json",
  "previous_immutable_bundle_sha256": "eab30233c54c087018050e6c31d77fb6636aba46ec1d81528d0f3a633219b715",
  "removed_independent_active_reviewer_binding": {
    "path": "training/v169_root_final_preseal_review.py",
    "previous_sha256": "a59f7f6a76791bf55ea64b92e5daadfc29e86e6dd931e8ec45c267454c26c90e",
    "stable_current_sha256": "204a289756c24f14c10fd9281baa5d688c3e0d1f3c266b05c6cbea5fa6340378",
    "final_independent_report_will_bind_its_own_source": true
  },
  "sealer_path": "training/v169_seal_prior_pair_training_v4.py",
  "sealer_sha256": "6cec63af110f225e4e3d121df8464d5b8a2c0f7d09befa776f5659558dd13206",
  "same_v13_training_entry_schedule_and_qualifications": true,
  "complete_physical_manifest_path": "artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json",
  "complete_physical_manifest_sha256": "e2375da9a3cff4fcd1dd1bd135a3c1d9f8a1f87922d17aeedea373ab322e1901"
}
```

## 前一方向与全部历史限制完整保留

# V168独立结果与下一轮训练决定

**建议开展下一轮有界、配对训练；不直接重复V164入口，也不将V168称为已经训练成功。** 本次用户要求先判断与优化方案，因此新拟合尚未启动。当前仍只解决“训练侧没有学会分类”，不把同类支持和独立来源迁移同时混入本轮。

## 1. 实际结果与继续依据

根端 `training/v168_independent_actual_decision_floor_review.py` 实际 exit0；从官方独立gold、全部原行、25个完整输入函数、50次配对全参数裕量导数、原点两类目标梯度、实际候选、逐行保护、完全恢复和调用账本复算，并独立重放1次CPU保存向量QP。本审查0新官方调用。

|角色|相对V164的M修复/新增错|相对V164的S修复/新增错|纯错误：V164→安全候选|本次新收益|
|---|---:|---:|---:|---|
|0|356/0|0/0|3278→2922|冻结V166/V167参考，没有新增|
|1|4/0|0/0|1952→1948|消除V167新增2条S退化，保留4M修复|
|2|8/0|0/0|1586→1578|冻结V166/V167参考，没有新增|

角色1候选M错误864/38886，召回97.78%；S错误1156/2071，召回44.18%。消除两条S退化只是恢复V164已经具有的正确判断，相对V164没有新S修复。不能把这两条称为原S学习缺口取得突破。

实际两条S的q差约1.776e-15，现有argmax确实判对；没有认证跨硬件/跨状态稳定距离。下一入口必须重新核对真实分类，不能仅凭概率close认定旧正确不变。

V168新增98头/特征、50完整裕量导数、1QP、1真实候选、0拟合/永久更新，全部恢复V164。累计19178头/特征、768完整导数；自V159以来9拟合/170永久更新不变。最新实际训练仍V164，最新完整三分类交付仍V159；第一问题及总体质量均未通过。

独立证据：`artifacts/v168_independent_actual_decision_floor_review_20261002/review.json`。三角色均存在已实际核验的安全候选，支持另登记训练；这项许可不等于训练掌握或模型晋升。

## 2. 进步慢的两个主要卡点

### 2.1 训练被计算数量限制截断

V164只接受1/1/3个更新，之后完整保护函数并集25/27/32超过当轮上限24，在新导数测量前停止。不是“充分训练后仍无法分类”，也没有局部/全局不可行证明。

旧正确原行的永久保护与临时QP矩阵中的函数管理应分开。修复计算管理不授权漏查保护行、删除困难记录、放宽每类错误门槛或在旧运行中追认扩大容量。

### 2.2 固定错误先验给纠错分支施加了很大负担

根端 `training/v168_root_saved_prior_burden_review.py` 实际 exit0，仅使用保存的基座概率、逐行结果、原gold和CPU checkpoint，没有新官方分类器/梯度调用。

|V164状态|S错误|其中真值基座均值概率≤1e-12|错例固定先验中位log差|已学纠正中位值|当前读出最大纠正幅度|
|---|---:|---:|---:|---:|---:|
|角色0|1414|831|-27.631|+0.176|0.894|
|角色1|1156|813|-27.631|+0.143|0.981|
|角色2|892|319|-2.708|+0.528|2.550|

当前函数为 `z=log(max(mean_p,1e-12))+delta(x,p)`。基座不是威胁真值，但它的log意见系数被固定为1；基座将S赋零时，纠错分支需跨过约27.63的差距。角色1若固定当前输出矩阵、只改变tanh隐藏表示，最多提供约0.981的M/S差修正，1082条现有S错误仍无法翻转。

这个界限只针对当前固定读出。现有输出矩阵本来可训练，不能由此推导模型整体容量不足、原数据缺信息或错误必然无解。它支持检查“固定先验过强+接受轨迹太短”，尚未证明可学习先验权重必然有效。

独立证据：`artifacts/v168_root_saved_prior_burden_review_20261002/review.json`。

## 3. 下一轮的唯一学习问题与配对设计

**问题：在保持已经判对原行的条件下，能否真正推翻错误基座意见，学会当前训练侧M/S分类？**

采用三角色、两臂，共最多6次拟合。两臂共同先修复执行管理，唯一模型变化为固定先验的系数是否可学习：

- A控制：`z=log(max(mean_p,1e-12))+delta(x,p)`。
- B预登记候选：`z=exp(beta)*log(max(mean_p,1e-12))+delta(x,p)`，唯一新增标量beta初始化0。所有类别共享这个标量，不按标签、来源、时间、端口或错例编号配置系数。beta和现有完整头共同训练。

两臂相同输入、16个合法基座意见、原频次、gold、固定纯错误队列、类别完整分母、已验收能力、种子、初始化和有限验收；同一角色从同一个安全候选开始。beta=0必须真实重现A的完整q/logq和逐行判决。基座概率不改成标签、不重新伪标，不更改1e-12的底层零概率表示。

这不是对最终输出做后处理温度校准。正温度对全部最终logits的缩放保持argmax，不能单独修复错类；B只让基座意见相对于原始信息纠错分支的贡献可学习，其是否修复错类必须实测。

不得把B改成全局类别偏置、单独S权重、按来源路由或多参数注意力搜索；这些都超出本次单因素。若共享标量不足，只记录失败，不在本轮追加更复杂门控。

## 4. 两臂共用的执行修复

采用“完整保护账本＋动态局部工作集”的约束管理：

1. 全部注册正确原行、旧技能及每次真实接受的新修复永久留在完整账本；每个候选仍全量扫描实际argmax、原M/S错误数、原风险和所有部署/旧技能。
2. 每个新接受参数点重新计算两类完整固定错误目标梯度并复测。失效的旧点Jacobian不得复用。
3. QP工作集随实际阻塞更新；可以移出当前不活动的局部函数，但不得移出其原行保护。移出后仍完整扫描，一旦再次阻塞必须重新纳入并在当前点测量。
4. 同时需要更多活动函数时，分块/精确稀疏存储和资源前置资格处理；不得把未测函数当安全，也不得靠丢行凑数量上限。真正达到事前物理预算即停止并恢复，不声称模型不可学。
5. V168已验证的判决地板规则、原单位数值复核和真实有限验收共同保留。局部地板仅辅助求方向；不能放宽实际判决。
6. 新入口首次真实重放三个安全候选。逐行判决与完整保护通过后才提交初始化为新训练状态并冻结其修复；诊断候选不是已永久掌握成果，不能跳过这一步。

工作集方法是针对既有容量截断的执行修复，不能将传统线性QP的收敛性质直接套到神经网络。工作集、局部修正和候选总次数均须另登记；没有自动循环到成功的权限。

## 5. 预算、关键记录与验收

前瞻设计上限为每臂每角色20个真实接受更新、每个新点最多2次修正；最多6拟合/120接受更新。采用固定注册日程，不因早期弱切片未翻类而提前结束。提前结束只允许真正掌握并完成末5个真实不同状态验收、技术失败、原保护无法满足或达到事前资源/调用限制；停止原因分别记录。

该数字不是执行授权。实施前依据实际调用图计算头/意见、完整梯度、QP、有限候选、磁盘/RAM/GPU的精确最坏上限，并与现有累计成本共同绑定。配对完整导数需能无损恢复全部参数；精确稀疏/压缩只能改变保存方式，不能改变数值、参数范围或独立可复算性。预算不足时事前减小两臂相同范围或使用平台，不在看到结果后调整。

必须每点保存：

- 实际参数身份、原点/线性化点、完整目标与裕量导数身份、接受/拒绝/恢复及真实成本。
- 每类原始分母、正确/错误、召回/精确率、修复/新增错误；混类原行仍完整计分。重叠角色不得当独立记录相加。
- 初始固定的低先验S切片及其他S切片：实际错数、修复原行/独立输入数、裕量分布、基座先验与学习残差分解。始终使用同一个初始切片，不用“剩余错例集合”偷换分母。
- B的beta/实际先验系数、输出矩阵纠正界限及两类实际修复；系数变化、loss下降不算分类成功。
- 所有累计保护回归；两条S微小正间隔在以后状态中仍必须真实判对。

预登记判定：

1. 安全：全部已验收范围与累计接受修复新增错0，原M/S逐类错误不增加；任何一项失败就拒绝候选。
2. 学习：各角色纯错误是否减少，尤其初始低先验S是否出现真实修复；完整混类错误独立报告。不只看总准确率或CE。
3. B机制支持：匹配终点B在三角色各类错误均不高于A，并且各角色S有实际额外修复；低先验S切片是否改善单列。否则只能称局部或无支持，不换种子挑折晋升。B是唯一候选，A结果更好时保留旧正式模型、记录对照发现，不事后更换候选。
4. 第一问题真正关闭：三个注册角色纯错误0，末5个真实不同接受参数状态均掌握，全部旧技能与累计修复保持。未达到就不称解决；有限诊断、少量修复不能替代。
5. 来源泛化与完整N/M/S质量：当前已反复查看的开发角色不能称盲测。第一问题掌握也不自动通过后两问题或2056871行完整质量验收；后续单独检验。

如果日程完成仍无深S分类收益，停止本配置，依据beta是否真正变化、读出幅度、有限保护阻塞及每类错误分解归因。不要再把“降低了损失但未翻类”写成有效方案，也不机械追加训练轮数。

## 6. 调研取舍与反证

- [NEOS的QP工作集方法](https://neos-guide.org/guide/algorithms/qp/)说明局部工作集可随活动约束更新。借鉴它管理计算，不借此取消完整神经网络逐行验收。
- [scikit-learn 1.3.2 StackingClassifier](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.StackingClassifier.html)将基模型输出作为最终学习器输入，默认最终学习器是逻辑回归。借鉴“基座意见应由最终监督学习器决定贡献”，本项目共享先验系数是最小可检验改法，不是该库效果的复现或保证。其关于同数据预拟合堆叠的过拟合限制仍适用，不能偷换来源外验收。
- [ICML 2017校准论文](https://proceedings.mlr.press/v70/guo17a.html)及[作者实现](https://github.com/gpleiss/temperature_scaling)支持概率校准。正标量缩放最终logits不改变排序，故不采纳它作为本轮错类修复的替代品。
- [SciPy 1.15.3 SLSQP](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-slsqp.html)中的精度/迭代参数只描述优化停止，不证明原神经网络分类安全。保持原单位复算与实际分类门槛。

本次审查保留四个反证：平手修复不等于S学习；容量截断不等于充分训练；当前固定读出界限不等于全模型不可学；学习先验权重不等于来源泛化。新方案须先验证真实零步、完整参数/预算、异常恢复、工作集移出后再阻塞、两臂同人口及历史反例，再封存执行。当前新方案0拟合，尚无效果声明。


## 预算适用范围纠正（当前解释）

旧审查把82174/2426称为继承累计上限；这些数值不能自动成为V169或整个项目的永久上限。2426最初是V159完整类别梯度2420加V160诊断类别梯度6，V160另列144间隔导数；V164后来明确采用2426全部导数技术上限，但各原契约只约束其自身入口和注册范围。旧上限若继续沿用，剩余1658及朴素满程12480的条件算术成立；它不证明beta方法失败。V169须由实际新入口前瞻计算并登记相称的新预算，历史19178头/768导数、9fit/170更新保留。当前不扩大预算，不赋训练权，0新官方调用。精确调用图、参数身份、保护账本、初始化更新计数和不匹配终点处理仍待落实。

下面完整保留旧审查与初版方向作为历史证据；其关于永久继承预算的解释以本节为准。

```json
{
  "status": "V169_original_budget_scope_checked_not_a_permanent_project_cap",
  "original_V159_scope": "six_fit_current_input_boundary_numeric_execution_contract_v4",
  "V159_cumulative_complete_class_gradient_cap": 2420,
  "V160_scope": "V160-three-fixed-B-endpoint-active-margin-finite-diagnostic-v1",
  "V160_allowed_entries": [
    "training/v160_fixed_endpoint_diagnostic_v3.py"
  ],
  "V160_head_cap_arithmetic": "78226+3948=82174",
  "V160_class_gradient_cap_arithmetic": "2420+6=2426",
  "V160_margin_gradient_cap_separately_registered": 144,
  "V164_later_explicit_all_derivative_cap": 2426,
  "original_2426_is_not_automatically_all_derivative_project_lifetime_cap": true,
  "old_trial_limits_remain_binding_for_their_own_entries": true,
  "previous_1658_remainder_is_only_conditional_on_reusing_old2426_ceiling": true,
  "conditional_remainder": 1658,
  "conditional_naive_dense_derivatives": 12480,
  "V169_exact_callgraph_not_implemented": true,
  "V169_new_budget_not_registered": true,
  "no_new_budget_or_training_authority": true,
  "no_beta_failure_or_effect_claim": true,
  "historical_costs_not_reset": {
    "heads": 19178,
    "complete_derivatives": 768,
    "fits_since_V159": 9,
    "updates_since_V159": 170
  },
  "still_required": [
    "Actual new entry callgraph including bootstrap, terminal repeats, refusal and restoration",
    "Prospectively register bounded identical arm schedules and proportionate resource/cumulative cost bounds before execution",
    "B complete1060833 parameter identity and derivative/repeat policy",
    "Permanent full row protection ledger and current-point working-set derivatives",
    "Define bootstrap update counting within20 per fit and120 total",
    "Predeclare paired endpoint inconclusive after unmatched technical termination",
    "Qualification, independent root review and physical seal"
  ],
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "execution_authority": false,
  "new_fit_permission": false,
  "source_sha256": {
    "training/v169_publish_budget_scope_correction.py": "7a90ef2699fb0add5f4791cd86efa299ace9dcfc9dc243a392fb348a25b840f2",
    "docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md": "6cb91a9b340f1b530ba14c224a4deb00de6769c06138da51e5c2b05d5c072f73",
    "training/review_policy/v169_learnable_prior_pair_draft.json": "109272638d24ff36eb9b9f8a0d301487d2b943c19a57a72947fba54c84cc18e5",
    "training/review_policy/v159_boundary_execution_contract_v4.json": "4269229687cf3f22ec139de23e7cf23b5ed25b690b9400cb9bd7abb81bfc0279",
    "training/review_policy/v160_fixed_endpoint_diagnostic_contract.json": "82aba35cc82ba0d473a71b6a71a5bd560c9b746e7766642cc0714ecbbc1c1752",
    "training/review_policy/v164_short_trajectory_prospective_budget.json": "3f95ce9a1452e853e6a180d5516b143b2412c050cbc370a46981e5eac13af11a",
    "training/v160_seal_fixed_endpoint_diagnostic.py": "c3cd3fe70d0e8425b2e19918faf49723c31935517f805fd4b876560ab4d913b6",
    "training/v168_decision_floor_execution_review.py": "9208df8ed1df68c40e2af23134f2f6381203af3a6f657a7a6208aac9796a407d",
    "docs/V159_FIXED_NUMERIC_POLICY_AND_TECHNICAL_BUDGET_CANDIDATE.md": "a7bb0f69a73ee0513ebf26669997867f52b162f41fefe056141d781903605581",
    "docs/V160_FIXED_ENDPOINT_FINITE_DIAGNOSTIC_PLAN_20261002.md": "79f09ef14976b669b2a304d74d09306ce7aeab3c3d49cc6d20a5ab96c1e3ae41",
    "docs/EXPERIMENT_REVIEW_RULES.md": "00de5f109bd5fbd5060bd8138c4fb5c4f9a3c59cb65cff9f8eeb09292520ab41",
    "artifacts/v169_saved_prospective_budget_review_20261002/review.json": "2f8ccf497e5ed8f1f98bc50bb33a1e3a2cf80ed123cd2d396c0b25477f79acc2",
    "artifacts/v168_results_20261002/actual_result_summary.json": "bc27e7273fb8612ef3221bb7507702c0a4ac6ce2f7bd69fad784a79b072a19b9",
    "artifacts/v168_direction_20261002/plan.md": "1d957e95f1bb887deb25f14a8d1de34e515cef036a5ff439075f757af8832b13",
    "artifacts/v168_direction_20261002/publication.json": "10b7d41bb2f702f21980c0868f1392d0f84361a67b1444d010999a938642fd70",
    "artifacts/v168_direction_20261002/validation.json": "65a2fa5c9d7e8ee212f9809b11531e99c78044f2b8c1622e8d4a7b092b1e0fec"
  }
}

```

## 初版方向与原始审查完整保留

# V168独立结果与下一轮训练决定

**建议开展下一轮有界、配对训练；不直接重复V164入口，也不将V168称为已经训练成功。** 本次用户要求先判断与优化方案，因此新拟合尚未启动。当前仍只解决“训练侧没有学会分类”，不把同类支持和独立来源迁移同时混入本轮。

## 1. 实际结果与继续依据

根端 `training/v168_independent_actual_decision_floor_review.py` 实际 exit0；从官方独立gold、全部原行、25个完整输入函数、50次配对全参数裕量导数、原点两类目标梯度、实际候选、逐行保护、完全恢复和调用账本复算，并独立重放1次CPU保存向量QP。本审查0新官方调用。

|角色|相对V164的M修复/新增错|相对V164的S修复/新增错|纯错误：V164→安全候选|本次新收益|
|---|---:|---:|---:|---|
|0|356/0|0/0|3278→2922|冻结V166/V167参考，没有新增|
|1|4/0|0/0|1952→1948|消除V167新增2条S退化，保留4M修复|
|2|8/0|0/0|1586→1578|冻结V166/V167参考，没有新增|

角色1候选M错误864/38886，召回97.78%；S错误1156/2071，召回44.18%。消除两条S退化只是恢复V164已经具有的正确判断，相对V164没有新S修复。不能把这两条称为原S学习缺口取得突破。

实际两条S的q差约1.776e-15，现有argmax确实判对；没有认证跨硬件/跨状态稳定距离。下一入口必须重新核对真实分类，不能仅凭概率close认定旧正确不变。

V168新增98头/特征、50完整裕量导数、1QP、1真实候选、0拟合/永久更新，全部恢复V164。累计19178头/特征、768完整导数；自V159以来9拟合/170永久更新不变。最新实际训练仍V164，最新完整三分类交付仍V159；第一问题及总体质量均未通过。

独立证据：`artifacts/v168_independent_actual_decision_floor_review_20261002/review.json`。三角色均存在已实际核验的安全候选，支持另登记训练；这项许可不等于训练掌握或模型晋升。

## 2. 进步慢的两个主要卡点

### 2.1 训练被计算数量限制截断

V164只接受1/1/3个更新，之后完整保护函数并集25/27/32超过当轮上限24，在新导数测量前停止。不是“充分训练后仍无法分类”，也没有局部/全局不可行证明。

旧正确原行的永久保护与临时QP矩阵中的函数管理应分开。修复计算管理不授权漏查保护行、删除困难记录、放宽每类错误门槛或在旧运行中追认扩大容量。

### 2.2 固定错误先验给纠错分支施加了很大负担

根端 `training/v168_root_saved_prior_burden_review.py` 实际 exit0，仅使用保存的基座概率、逐行结果、原gold和CPU checkpoint，没有新官方分类器/梯度调用。

|V164状态|S错误|其中真值基座均值概率≤1e-12|错例固定先验中位log差|已学纠正中位值|当前读出最大纠正幅度|
|---|---:|---:|---:|---:|---:|
|角色0|1414|831|-27.631|+0.176|0.894|
|角色1|1156|813|-27.631|+0.143|0.981|
|角色2|892|319|-2.708|+0.528|2.550|

当前函数为 `z=log(max(mean_p,1e-12))+delta(x,p)`。基座不是威胁真值，但它的log意见系数被固定为1；基座将S赋零时，纠错分支需跨过约27.63的差距。角色1若固定当前输出矩阵、只改变tanh隐藏表示，最多提供约0.981的M/S差修正，1082条现有S错误仍无法翻转。

这个界限只针对当前固定读出。现有输出矩阵本来可训练，不能由此推导模型整体容量不足、原数据缺信息或错误必然无解。它支持检查“固定先验过强+接受轨迹太短”，尚未证明可学习先验权重必然有效。

独立证据：`artifacts/v168_root_saved_prior_burden_review_20261002/review.json`。

## 3. 下一轮的唯一学习问题与配对设计

**问题：在保持已经判对原行的条件下，能否真正推翻错误基座意见，学会当前训练侧M/S分类？**

采用三角色、两臂，共最多6次拟合。两臂共同先修复执行管理，唯一模型变化为固定先验的系数是否可学习：

- A控制：`z=log(max(mean_p,1e-12))+delta(x,p)`。
- B预登记候选：`z=exp(beta)*log(max(mean_p,1e-12))+delta(x,p)`，唯一新增标量beta初始化0。所有类别共享这个标量，不按标签、来源、时间、端口或错例编号配置系数。beta和现有完整头共同训练。

两臂相同输入、16个合法基座意见、原频次、gold、固定纯错误队列、类别完整分母、已验收能力、种子、初始化和有限验收；同一角色从同一个安全候选开始。beta=0必须真实重现A的完整q/logq和逐行判决。基座概率不改成标签、不重新伪标，不更改1e-12的底层零概率表示。

这不是对最终输出做后处理温度校准。正温度对全部最终logits的缩放保持argmax，不能单独修复错类；B只让基座意见相对于原始信息纠错分支的贡献可学习，其是否修复错类必须实测。

不得把B改成全局类别偏置、单独S权重、按来源路由或多参数注意力搜索；这些都超出本次单因素。若共享标量不足，只记录失败，不在本轮追加更复杂门控。

## 4. 两臂共用的执行修复

采用“完整保护账本＋动态局部工作集”的约束管理：

1. 全部注册正确原行、旧技能及每次真实接受的新修复永久留在完整账本；每个候选仍全量扫描实际argmax、原M/S错误数、原风险和所有部署/旧技能。
2. 每个新接受参数点重新计算两类完整固定错误目标梯度并复测。失效的旧点Jacobian不得复用。
3. QP工作集随实际阻塞更新；可以移出当前不活动的局部函数，但不得移出其原行保护。移出后仍完整扫描，一旦再次阻塞必须重新纳入并在当前点测量。
4. 同时需要更多活动函数时，分块/精确稀疏存储和资源前置资格处理；不得把未测函数当安全，也不得靠丢行凑数量上限。真正达到事前物理预算即停止并恢复，不声称模型不可学。
5. V168已验证的判决地板规则、原单位数值复核和真实有限验收共同保留。局部地板仅辅助求方向；不能放宽实际判决。
6. 新入口首次真实重放三个安全候选。逐行判决与完整保护通过后才提交初始化为新训练状态并冻结其修复；诊断候选不是已永久掌握成果，不能跳过这一步。

工作集方法是针对既有容量截断的执行修复，不能将传统线性QP的收敛性质直接套到神经网络。工作集、局部修正和候选总次数均须另登记；没有自动循环到成功的权限。

## 5. 预算、关键记录与验收

前瞻设计上限为每臂每角色20个真实接受更新、每个新点最多2次修正；最多6拟合/120接受更新。采用固定注册日程，不因早期弱切片未翻类而提前结束。提前结束只允许真正掌握并完成末5个真实不同状态验收、技术失败、原保护无法满足或达到事前资源/调用限制；停止原因分别记录。

该数字不是执行授权。实施前依据实际调用图计算头/意见、完整梯度、QP、有限候选、磁盘/RAM/GPU的精确最坏上限，并与现有累计成本共同绑定。配对完整导数需能无损恢复全部参数；精确稀疏/压缩只能改变保存方式，不能改变数值、参数范围或独立可复算性。预算不足时事前减小两臂相同范围或使用平台，不在看到结果后调整。

必须每点保存：

- 实际参数身份、原点/线性化点、完整目标与裕量导数身份、接受/拒绝/恢复及真实成本。
- 每类原始分母、正确/错误、召回/精确率、修复/新增错误；混类原行仍完整计分。重叠角色不得当独立记录相加。
- 初始固定的低先验S切片及其他S切片：实际错数、修复原行/独立输入数、裕量分布、基座先验与学习残差分解。始终使用同一个初始切片，不用“剩余错例集合”偷换分母。
- B的beta/实际先验系数、输出矩阵纠正界限及两类实际修复；系数变化、loss下降不算分类成功。
- 所有累计保护回归；两条S微小正间隔在以后状态中仍必须真实判对。

预登记判定：

1. 安全：全部已验收范围与累计接受修复新增错0，原M/S逐类错误不增加；任何一项失败就拒绝候选。
2. 学习：各角色纯错误是否减少，尤其初始低先验S是否出现真实修复；完整混类错误独立报告。不只看总准确率或CE。
3. B机制支持：匹配终点B在三角色各类错误均不高于A，并且各角色S有实际额外修复；低先验S切片是否改善单列。否则只能称局部或无支持，不换种子挑折晋升。B是唯一候选，A结果更好时保留旧正式模型、记录对照发现，不事后更换候选。
4. 第一问题真正关闭：三个注册角色纯错误0，末5个真实不同接受参数状态均掌握，全部旧技能与累计修复保持。未达到就不称解决；有限诊断、少量修复不能替代。
5. 来源泛化与完整N/M/S质量：当前已反复查看的开发角色不能称盲测。第一问题掌握也不自动通过后两问题或2056871行完整质量验收；后续单独检验。

如果日程完成仍无深S分类收益，停止本配置，依据beta是否真正变化、读出幅度、有限保护阻塞及每类错误分解归因。不要再把“降低了损失但未翻类”写成有效方案，也不机械追加训练轮数。

## 6. 调研取舍与反证

- [NEOS的QP工作集方法](https://neos-guide.org/guide/algorithms/qp/)说明局部工作集可随活动约束更新。借鉴它管理计算，不借此取消完整神经网络逐行验收。
- [scikit-learn 1.3.2 StackingClassifier](https://scikit-learn.org/1.3/modules/generated/sklearn.ensemble.StackingClassifier.html)将基模型输出作为最终学习器输入，默认最终学习器是逻辑回归。借鉴“基座意见应由最终监督学习器决定贡献”，本项目共享先验系数是最小可检验改法，不是该库效果的复现或保证。其关于同数据预拟合堆叠的过拟合限制仍适用，不能偷换来源外验收。
- [ICML 2017校准论文](https://proceedings.mlr.press/v70/guo17a.html)及[作者实现](https://github.com/gpleiss/temperature_scaling)支持概率校准。正标量缩放最终logits不改变排序，故不采纳它作为本轮错类修复的替代品。
- [SciPy 1.15.3 SLSQP](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-slsqp.html)中的精度/迭代参数只描述优化停止，不证明原神经网络分类安全。保持原单位复算与实际分类门槛。

本次审查保留四个反证：平手修复不等于S学习；容量截断不等于充分训练；当前固定读出界限不等于全模型不可学；学习先验权重不等于来源泛化。新方案须先验证真实零步、完整参数/预算、异常恢复、工作集移出后再阻塞、两臂同人口及历史反例，再封存执行。当前新方案0拟合，尚无效果声明。


## 原始草案、实际边界与预算核查

V168已完成真实诊断及独立审查，但全部恢复V164，0fit/永久更新。V169只为下一训练设计，精确调用图、参数维度、工作集资格及根审查/物理封存尚未完成，没有新前向、梯度或拟合权限。

来源 `training/review_policy/v169_learnable_prior_pair_draft.json`；SHA256 `109272638d24ff36eb9b9f8a0d301487d2b943c19a57a72947fba54c84cc18e5`。

```json
{
  "status": "V169_prospective_paired_learning_design_not_execution_contract",
  "execution_authority": false,
  "new_fit_permission": false,
  "current_problem": "training_side_classification_not_mastered",
  "evidence": [
    "docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md",
    "artifacts/v168_independent_actual_decision_floor_review_20261002/review.json",
    "artifacts/v168_independent_actual_decision_floor_review_20261002/pre_review_bindings.json",
    "artifacts/v168_root_saved_prior_burden_review_20261002/review.json",
    "artifacts/v164_independent_capacity_stop_review_20261002/review.json"
  ],
  "proposed_arms": {
    "A": {
      "fixed_prior_coefficient": 1
    },
    "B": {
      "prior_coefficient": "exp(beta)",
      "beta_initial": 0,
      "additional_parameters": 1,
      "same_scalar_all_classes": true
    }
  },
  "candidate": "B",
  "proposed_fit_cap": 6,
  "proposed_roles": [
    0,
    1,
    2
  ],
  "proposed_accepted_updates_per_fit": 20,
  "proposed_permanent_update_cap": 120,
  "proposed_corrections_per_point": 2,
  "shared_execution_fix": "full_guard_ledger_and_dynamic_local_working_set",
  "same_initialization_and_input_labels_original_frequencies_and_population": true,
  "current_supervised_role_observations": 225614,
  "current_unique_original_ASA_rows": 112807,
  "overlapping_roles_not_independent": true,
  "initial_S_cohorts": [
    {
      "role": 0,
      "path": "artifacts/v169_learnable_prior_pair_plan_20261002/role0_all_S_initial_prior_cohort.parquet",
      "sha256": "e024febb7ccf5d9e0e98d0c42f3a8055554308f5c5796fc1732fe2628db100e7",
      "all_S_rows": 33497,
      "low_prior_rows": 831,
      "low_prior_errors": 831,
      "other_prior_correct_controls": 32083
    },
    {
      "role": 1,
      "path": "artifacts/v169_learnable_prior_pair_plan_20261002/role1_all_S_initial_prior_cohort.parquet",
      "sha256": "d468b8223d54395093f008efa4e7d4439af4a91bdef919e6669d895330ae6067",
      "all_S_rows": 2071,
      "low_prior_rows": 813,
      "low_prior_errors": 813,
      "other_prior_correct_controls": 915
    },
    {
      "role": 2,
      "path": "artifacts/v169_learnable_prior_pair_plan_20261002/role2_all_S_initial_prior_cohort.parquet",
      "sha256": "0fcddfc8285c9a5547a1a55ed57ecd75b89b447be09b6d3ff206a11081bd1463",
      "all_S_rows": 32550,
      "low_prior_rows": 319,
      "low_prior_errors": 319,
      "other_prior_correct_controls": 31658
    }
  ],
  "permanent_protection": "all_current_actual_mastered_ranges_and_all_new_actual_accepted_repairs",
  "quality_constraints": {
    "all_registered_protected_regressions": 0,
    "original_M_S_error_counts_must_not_increase": true,
    "classification_mastery_requires_all_three_pure_errors_zero_and_last_five_real_distinct_states": true,
    "full_task_and_independent_transfer_separate": true
  },
  "prior_actual_costs": {
    "heads": 19178,
    "complete_parameter_derivatives": 768,
    "fits_since_V159": 9,
    "permanent_updates_since_V159": 170
  },
  "actual_new_costs": {
    "heads": 0,
    "features": 0,
    "derivatives": 0,
    "fits": 0,
    "permanent_updates": 0
  },
  "pre_execution_requirements": [
    "actual A/B zero-step outputs and class decisions identical",
    "complete new parameter/function/counter/lifecycle and working-set historical-case qualification",
    "actual callgraph exact forward/derivative/QP/proposal/storage bounds and cumulative budget",
    "physical source data role and environment seal",
    "new root review"
  ],
  "model_training_not_started": true,
  "method_effect_not_claimed": true,
  "source_sha256": {
    "docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md": "6cb91a9b340f1b530ba14c224a4deb00de6769c06138da51e5c2b05d5c072f73",
    "artifacts/v168_independent_actual_decision_floor_review_20261002/review.json": "d7a65add7e256235f3961dfd526418463af20851ffc9bc802ac8e89dbc13d602",
    "artifacts/v168_independent_actual_decision_floor_review_20261002/pre_review_bindings.json": "41e3451116649ad04871f4464bb6f795f926d40e0c76df66fdde31738b5c4ba2",
    "artifacts/v168_root_saved_prior_burden_review_20261002/review.json": "2dcaf686f3ce73c655e31420473d1ab3d956019faef104f29f9970d336a24696",
    "artifacts/v164_independent_capacity_stop_review_20261002/review.json": "ef79550909f051f0cd747df697206a4a9d522c76843dd76a88afa95ace7beaa6",
    "data/official/train.parquet": "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/endpoint/OOF_original_rows.parquet": "e8d38d97f78a49e00e7365e16e30e6b33d00060b18d52ad02a172f8a64f303a3",
    "artifacts/v158_legal_fusion_bank_v2_20261001/fold0/OOF_probabilities.npy": "1a158fb881748593c16c5b3a90f42b33acb37ce2ea35969a6737f6747eef3e64",
    "artifacts/v169_learnable_prior_pair_plan_20261002/role0_all_S_initial_prior_cohort.parquet": "e024febb7ccf5d9e0e98d0c42f3a8055554308f5c5796fc1732fe2628db100e7",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/endpoint/OOF_original_rows.parquet": "4f948c69359aac4ab62c3c7e82bdf7f2e48203556ac98585f0353f53a9ce6771",
    "artifacts/v158_legal_fusion_bank_v2_20261001/fold1/OOF_probabilities.npy": "a39938b15214da7e14dc25438776fb8ac9878ca2f1a0a940fcd77725eb622089",
    "artifacts/v169_learnable_prior_pair_plan_20261002/role1_all_S_initial_prior_cohort.parquet": "d468b8223d54395093f008efa4e7d4439af4a91bdef919e6669d895330ae6067",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/endpoint/OOF_original_rows.parquet": "fd10479c37471b5f728b444243853c52c31cefe75133c4e6a2f635320d232a79",
    "artifacts/v158_legal_fusion_bank_v2_20261001/fold2/OOF_probabilities.npy": "eed286ab24f2ca8ab338076319804bd77f2d572a4ab5163915a363129bcb9c65",
    "artifacts/v169_learnable_prior_pair_plan_20261002/role2_all_S_initial_prior_cohort.parquet": "0fcddfc8285c9a5547a1a55ed57ecd75b89b447be09b6d3ff206a11081bd1463"
  }
}

```

来源 `training/review_policy/v168_observed_runtime_boundaries.json`；SHA256 `bd4345f14dbfaf4da2b3fdbc1a6bfd2919453ead2e6fa0e38ef1576e88148e5e`。

```json
{
  "schema_version": 1,
  "actual_diagnostic": "V168",
  "inherit_applicable_constraints": "training/review_policy/v167_observed_runtime_boundaries.json",
  "cases": [
    {
      "id": "V168-ACTUAL-SAFE-TINY-MARGIN-NOT-STABILITY",
      "evidence": "artifacts/v168_saved_actual_floor_geometry_review_20261002/review.json",
      "actual_fact": "The two previous S ties are now correct with log margin3.552713678800501e-15 and q gap1.7763568394002505e-15; one actual candidate passed",
      "required_action": "Keep exact argmax and all original guards. Do not claim independent transfer, multiple-state repeat stability or mastery from this single tiny decision margin"
    },
    {
      "id": "V168-S-RESTORATION-NOT-NEW-S-ERROR-REPAIR",
      "evidence": "artifacts/v168_saved_decision_floor_quality_review_20261002/review.json",
      "actual_fact": "Versus V167 the two S regressions disappear; versus V164 there are4M repairs and0S old-error repairs, with0new M/S errors and1948 remaining pure role1 errors",
      "required_action": "Report both baselines and remaining per-class errors. Do not count restoring original correctness as repairing V164 S errors or hide serious remaining errors with total loss"
    },
    {
      "id": "V168-FINITE-DIAGNOSTIC-NOT-NEW-TRAINED-MODEL",
      "evidence": "artifacts/v168_decision_floor_diagnostic_20261002/role1/diagnostic.json",
      "actual_fact": "98heads/50derivatives/1QP/1candidate,0fits/0permanent updates and complete V164 restore; safe roles0/2 use immutable old references",
      "required_action": "Keep cumulative19178heads/768derivatives, latest real trainingV164 and full deliveryV159. No second correction or fit under this contract; unchanged safe controls have no new gains"
    },
    {
      "id": "V168-FROZEN-READOUT-BURDEN-NOT-GLOBAL-INFEASIBILITY",
      "evidence": "artifacts/v168_root_saved_prior_burden_review_20261002/review.json",
      "actual_fact": "At V164 role1,813of1156 S errors have truth prior at or below1e-12;1082 cannot be repaired while current readout is frozen, but all model parameters remain trainable",
      "required_action": "Track deep prior margins, learned residuals, readout scale and actual repairs. Do not claim model-wide impossibility, missing information or solve classification using temperature calibration"
    },
    {
      "id": "V168-ELIGIBLE-FOR-REGISTRATION-NOT-AUTOMATIC-TRAINING",
      "evidence": "artifacts/v168_independent_actual_decision_floor_review_20261002/review.json",
      "actual_fact": "Independent original-gold audit passes the single finite diagnostic and frozen safe controls, but training_issue_mastered and quality_acceptance remain false",
      "required_action": "Register a new single-issue bounded training contract with complete protection, real accepted trajectory and deep-error progress. Preserve V164 capacity stops and failed budgets; do not rerun the old entry or substitute this result for pure-error-zero and five distinct mastered states"
    }
  ],
  "no_new_execution_authority": true,
  "executable_constraint_replay": "training/v168_record_observed_boundaries.py",
  "scope": "Actual supervised development finite diagnostic and state-dependent saved readout bounds only; no full task or independent-source acceptance"
}

```

来源 `artifacts/v168_saved_accepted_learning_trajectory_review_20261002/review.json`；SHA256 `c947fd6fdf415dea4ecf09b1c4945947f0588dc4118e809ecede683b73c43d47`。

```json
{
  "status": "V164_actual_committed_trajectories_and_V168_uncommitted_deep_error_burden_reviewed",
  "roles": [
    {
      "role": 0,
      "actual_accepted_updates": 1,
      "actual_stop": "margin_normal_cap_stop",
      "actual_committed_trajectory": [
        {
          "kind": "actual_training_origin",
          "state_index": 0,
          "parameter_sha256": "cb06249c27330c89cf6bfd7359391c969c75d81ee7697bf1f60c9f7707b83b24",
          "classes": {
            "M": {
              "original_class_mass": 58840,
              "errors": 1886,
              "pure_errors": 1886,
              "mixed_errors": 0,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.14303964302612093,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4483,
              "zero_recall_root_proxy_groups": 24,
              "equal_root_proxy_recall": 0.9928464523161628,
              "remaining_fixed_errors": 1886,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.630944231945914,
                "0.25": -1.9444026231436318,
                "0.5": -0.7883820012234901,
                "0.75": -0.2511526879019299,
                "1.0": -0.0031628571639973346
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1.0": 0.0
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 33497,
              "errors": 1414,
              "pure_errors": 1392,
              "mixed_errors": 22,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.758562250497899,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 158,
              "zero_recall_root_proxy_groups": 124,
              "equal_root_proxy_recall": 0.19377912857696564,
              "remaining_fixed_errors": 1414,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.630982076333346,
                "0.25": -27.63098141974405,
                "0.5": -27.63098141974405,
                "0.75": -2.7078193054089423,
                "1.0": -2.56527224248293e-05
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1.0": 0.0
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 188,
              "fixed_deep_remaining_errors": 188,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.630944231945914,
                "0.25": -27.630944231945914,
                "0.5": -27.630944231945914,
                "0.75": -27.630944231945914,
                "1": -27.630943886796512
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 7.688398263283602e-05,
                "0.25": 7.688398263283602e-05,
                "0.5": 7.688398263283602e-05,
                "0.75": 7.688398263283602e-05,
                "1": 7.722913203522808e-05
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.0002644898709127981,
              "all_current_errors": 1886,
              "all_current_pure_errors": 1886,
              "full_original_class_mass": 58840
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 831,
              "fixed_deep_remaining_errors": 831,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.630982076333346,
                "0.25": -27.63098141974405,
                "0.5": -27.63098141974405,
                "0.75": -27.63098141974405,
                "1": -27.630981137632304
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 3.9039595201018074e-05,
                "0.25": 3.969618449772838e-05,
                "0.5": 3.969618449772838e-05,
                "0.75": 3.969618449772838e-05,
                "1": 3.9978296243248224e-05
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.0002644898709127981,
              "all_current_errors": 1414,
              "all_current_pure_errors": 1392,
              "full_original_class_mass": 33497
            }
          ]
        },
        {
          "kind": "actual_committed_training_state",
          "state_index": 1,
          "parameter_sha256": "200408346991be4071b2dd1406226f21f421ebad6829e9050c83b7e5454ae52f",
          "newly_repaired_original_rows": 0,
          "classes": {
            "M": {
              "original_class_mass": 58840,
              "errors": 1886,
              "pure_errors": 1886,
              "mixed_errors": 0,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.14217594026757943,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4483,
              "zero_recall_root_proxy_groups": 24,
              "equal_root_proxy_recall": 0.9928464523161628,
              "remaining_fixed_errors": 1886,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.41767554319595,
                "0.25": -1.7865504870856883,
                "0.5": -0.7121646095260881,
                "0.75": -0.22650592883534848,
                "1.0": -0.002293273350627456
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0008695838133698786,
                "0.25": 0.02383801757178594,
                "0.5": 0.07605244485996332,
                "0.75": 0.1564900455076803,
                "1.0": 0.21456775177889398
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 33497,
              "errors": 1414,
              "pure_errors": 1392,
              "mixed_errors": 22,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.7528555339201842,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 158,
              "zero_recall_root_proxy_groups": 124,
              "equal_root_proxy_recall": 0.19377912857696564,
              "remaining_fixed_errors": 1414,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.457074402115058,
                "0.25": -27.45542759133048,
                "0.5": -27.45542759133048,
                "0.75": -2.5512224723412196,
                "1.0": -0.000501090663636683
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": -0.0009230925650383082,
                "0.25": 0.1549626242779394,
                "0.5": 0.1755538284135696,
                "0.75": 0.1755538284135696,
                "1.0": 0.17687345654241327
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 188,
              "fixed_deep_remaining_errors": 188,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.41767554319595,
                "0.25": -27.41767554319595,
                "0.5": -27.41767554319595,
                "0.75": -27.41767554319595,
                "1": -27.416376135017618
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.2133455727325959,
                "0.25": 0.2133455727325959,
                "0.5": 0.2133455727325959,
                "0.75": 0.2133455727325959,
                "1": 0.2146449809109292
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.8936808195813299,
              "all_current_errors": 1886,
              "all_current_pure_errors": 1886,
              "full_original_class_mass": 58840
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 831,
              "fixed_deep_remaining_errors": 831,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.457074402115058,
                "0.25": -27.45542759133048,
                "0.5": -27.45542759133048,
                "0.75": -27.45542759133048,
                "1": -27.454107796908946
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.17394671381348914,
                "0.25": 0.17559352459806732,
                "0.5": 0.17559352459806732,
                "0.75": 0.17559352459806732,
                "1": 0.17691331901960083
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.8936808195813299,
              "all_current_errors": 1414,
              "all_current_pure_errors": 1392,
              "full_original_class_mass": 33497
            }
          ]
        }
      ],
      "temporary_V168_candidate": null,
      "mastery": false
    },
    {
      "role": 1,
      "actual_accepted_updates": 1,
      "actual_stop": "margin_normal_cap_stop",
      "actual_committed_trajectory": [
        {
          "kind": "actual_training_origin",
          "state_index": 0,
          "parameter_sha256": "a9f562f610e9c391d2329cddd6eea5622749dddf80152469ecac46d1bccd0cb8",
          "classes": {
            "M": {
              "original_class_mass": 38886,
              "errors": 872,
              "pure_errors": 800,
              "mixed_errors": 72,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.28540780656264153,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4466,
              "zero_recall_root_proxy_groups": 53,
              "equal_root_proxy_recall": 0.9848944378362883,
              "remaining_fixed_errors": 872,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -1.400129565949642,
                "0.75": -0.5108256237659906,
                "1.0": -4.076694163135386e-06
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1.0": 0.0
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 2071,
              "errors": 1160,
              "pure_errors": 1160,
              "mixed_errors": 0,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 11.428015395544154,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 160,
              "zero_recall_root_proxy_groups": 119,
              "equal_root_proxy_recall": 0.19005952380952382,
              "remaining_fixed_errors": 1160,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -2.762583980470788,
                "1.0": -2.7024971149813837e-10
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1.0": 0.0
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 288,
              "fixed_deep_remaining_errors": 288,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1": 0.0
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.0,
              "all_current_errors": 872,
              "all_current_pure_errors": 800,
              "full_original_class_mass": 38886
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 813,
              "fixed_deep_remaining_errors": 813,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928487
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1": 0.0
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928487
              },
              "current_readout_M_S_margin_bound": 0.0,
              "all_current_errors": 1160,
              "all_current_pure_errors": 1160,
              "full_original_class_mass": 2071
            }
          ]
        },
        {
          "kind": "actual_committed_training_state",
          "state_index": 1,
          "parameter_sha256": "56d9600e98ff480de24147b1e5a632e9c85ff2fad2f3dac65f492008bf76c9fd",
          "newly_repaired_original_rows": 8,
          "classes": {
            "M": {
              "original_class_mass": 38886,
              "errors": 868,
              "pure_errors": 796,
              "mixed_errors": 72,
              "repairs_vs_previous_accepted": 4,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 4,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.2835112661159661,
              "fixed_endpoint_repair_distinct_locals": 1,
              "fixed_endpoint_repair_distinct_roots": 1,
              "largest_root_share_of_fixed_endpoint_repairs": 1.0,
              "root_proxy_support_groups": 4466,
              "zero_recall_root_proxy_groups": 53,
              "equal_root_proxy_recall": 0.9849840034430953,
              "remaining_fixed_errors": 868,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.408252717276994,
                "0.25": -27.40750496921824,
                "0.5": -1.301762582182624,
                "0.75": -0.4621360888108827,
                "1.0": -8.335279624238634e-05
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": -0.0017001922379495937,
                "0.25": 0.04850616559208765,
                "0.5": 0.1304789721402514,
                "0.75": 0.22276839865155296,
                "1.0": 0.22510802944426445
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 2071,
              "errors": 1156,
              "pure_errors": 1156,
              "mixed_errors": 0,
              "repairs_vs_previous_accepted": 4,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 4,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 11.358216536757709,
              "fixed_endpoint_repair_distinct_locals": 2,
              "fixed_endpoint_repair_distinct_roots": 2,
              "largest_root_share_of_fixed_endpoint_repairs": 0.5,
              "root_proxy_support_groups": 160,
              "zero_recall_root_proxy_groups": 117,
              "equal_root_proxy_recall": 0.19220238095238096,
              "remaining_fixed_errors": 1156,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.49028317559476,
                "0.25": -27.4878005580534,
                "0.5": -27.4878005580534,
                "0.75": -2.7198732863062345,
                "1.0": -0.0005167074442525843
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": -0.0037411978198839124,
                "0.25": 0.13028615459898585,
                "0.5": 0.14322055787514643,
                "0.75": 0.14322055787514643,
                "1.0": 0.14422482901981937
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 288,
              "fixed_deep_remaining_errors": 288,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.408252717276994,
                "0.25": -27.408252717276994,
                "0.5": -27.408252717276994,
                "0.75": -27.40750496921824,
                "1": -27.405913086484283
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.22276839865155296,
                "0.25": 0.22276839865155296,
                "0.5": 0.22276839865155296,
                "0.75": 0.22351614671030617,
                "1": 0.22510802944426445
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.9810286087283117,
              "all_current_errors": 868,
              "all_current_pure_errors": 796,
              "full_original_class_mass": 38886
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 813,
              "fixed_deep_remaining_errors": 813,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.49028317559476,
                "0.25": -27.4878005580534,
                "0.5": -27.4878005580534,
                "0.75": -27.4878005580534,
                "1": -27.486796286908728
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.14073794033378562,
                "0.25": 0.14322055787514643,
                "0.5": 0.14322055787514643,
                "0.75": 0.14322055787514643,
                "1": 0.14422482901981937
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928487
              },
              "current_readout_M_S_margin_bound": 0.9810286087283117,
              "all_current_errors": 1156,
              "all_current_pure_errors": 1156,
              "full_original_class_mass": 2071
            }
          ]
        }
      ],
      "temporary_V168_candidate": {
        "kind": "temporary_finite_diagnostic_not_committed",
        "parameter_sha256": "4e922d215c58546536f35771969fa2a778a9b3ec25022045b5d3fa57e1878108",
        "classes_vs_last_accepted": {
          "M": {
            "original_class_mass": 38886,
            "errors": 864,
            "pure_errors": 792,
            "mixed_errors": 72,
            "repairs_vs_previous_accepted": 4,
            "new_errors_vs_previous_accepted": 0,
            "repairs_vs_fixed_endpoint": 8,
            "new_errors_vs_fixed_endpoint_all_original_rows": 0,
            "new_mixed_errors_vs_fixed_endpoint": 0,
            "new_pure_errors_vs_fixed_endpoint": 0,
            "protected_regressions": 0,
            "stable_CE_mean": 0.2820234553630577,
            "fixed_endpoint_repair_distinct_locals": 2,
            "fixed_endpoint_repair_distinct_roots": 2,
            "largest_root_share_of_fixed_endpoint_repairs": 0.5,
            "root_proxy_support_groups": 4466,
            "zero_recall_root_proxy_groups": 53,
            "equal_root_proxy_recall": 0.985006394844797,
            "remaining_fixed_errors": 864,
            "remaining_fixed_error_margin_quantiles": {
              "0.0": -27.190421402737545,
              "0.25": -27.188313032604672,
              "0.5": -1.1966133262089773,
              "0.75": -0.4088945838266833,
              "1.0": -0.0003826114030592853
            },
            "remaining_fixed_error_margin_change_quantiles": {
              "0.0": -0.0026522654711602645,
              "0.25": 0.1019310399393073,
              "0.5": 0.2650083203201197,
              "0.75": 0.44143715961887864,
              "1.0": 0.44585713006561534
            },
            "root_is_source_proxy_not_independent_attack_behavior": true
          },
          "S": {
            "original_class_mass": 2071,
            "errors": 1156,
            "pure_errors": 1156,
            "mixed_errors": 0,
            "repairs_vs_previous_accepted": 0,
            "new_errors_vs_previous_accepted": 0,
            "repairs_vs_fixed_endpoint": 4,
            "new_errors_vs_fixed_endpoint_all_original_rows": 0,
            "new_mixed_errors_vs_fixed_endpoint": 0,
            "new_pure_errors_vs_fixed_endpoint": 0,
            "protected_regressions": 0,
            "stable_CE_mean": 11.266201260814745,
            "fixed_endpoint_repair_distinct_locals": 2,
            "fixed_endpoint_repair_distinct_roots": 2,
            "largest_root_share_of_fixed_endpoint_repairs": 0.5,
            "root_proxy_support_groups": 160,
            "zero_recall_root_proxy_groups": 117,
            "equal_root_proxy_recall": 0.19220238095238096,
            "remaining_fixed_errors": 1156,
            "remaining_fixed_error_margin_quantiles": {
              "0.0": -27.302777279340678,
              "0.25": -27.298299899078568,
              "0.5": -27.298299899078568,
              "0.75": -2.5495891355288203,
              "1.0": -0.0015718297428468642
            },
            "remaining_fixed_error_margin_change_quantiles": {
              "0.0": -0.0015718102487529695,
              "0.25": 0.3005177556543779,
              "0.5": 0.33272121684997913,
              "0.75": 0.33272121684997913,
              "1.0": 0.33553453960126944
            },
            "root_is_source_proxy_not_independent_attack_behavior": true
          }
        },
        "deep_error_burden": [
          {
            "class_name": "M",
            "fixed_deep_initial_error_rows": 288,
            "fixed_deep_remaining_errors": 288,
            "fixed_deep_actual_margin_quantiles": {
              "0": -27.190421402737545,
              "0.25": -27.18958395630967,
              "0.5": -27.18958395630967,
              "0.75": -27.188313032604672,
              "1": -27.185163985862932
            },
            "fixed_deep_learned_residual_margin_quantiles": {
              "0": 0.440599713191002,
              "0.25": 0.44143715961887864,
              "0.5": 0.44143715961887864,
              "0.75": 0.44270808332387546,
              "1": 0.44585713006561534
            },
            "fixed_deep_frozen_prior_margin_quantiles": {
              "0": -27.631021115928547,
              "0.25": -27.631021115928547,
              "0.5": -27.631021115928547,
              "0.75": -27.631021115928547,
              "1": -27.631021115928547
            },
            "current_readout_M_S_margin_bound": 1.7903409155737087,
            "all_current_errors": 864,
            "all_current_pure_errors": 792,
            "full_original_class_mass": 38886
          },
          {
            "class_name": "S",
            "fixed_deep_initial_error_rows": 813,
            "fixed_deep_remaining_errors": 813,
            "fixed_deep_actual_margin_quantiles": {
              "0": -27.302777279340678,
              "0.25": -27.298299899078568,
              "0.5": -27.298299899078568,
              "0.75": -27.298299899078568,
              "1": -27.295486576327278
            },
            "fixed_deep_learned_residual_margin_quantiles": {
              "0": 0.3282438365878697,
              "0.25": 0.33272121684997913,
              "0.5": 0.33272121684997913,
              "0.75": 0.33272121684997913,
              "1": 0.33553453960126944
            },
            "fixed_deep_frozen_prior_margin_quantiles": {
              "0": -27.631021115928547,
              "0.25": -27.631021115928547,
              "0.5": -27.631021115928547,
              "0.75": -27.631021115928547,
              "1": -27.631021115928487
            },
            "current_readout_M_S_margin_bound": 1.7903409155737087,
            "all_current_errors": 1156,
            "all_current_pure_errors": 1156,
            "full_original_class_mass": 2071
          }
        ],
        "counts_as_additional_accepted_training_state": false
      },
      "mastery": false
    },
    {
      "role": 2,
      "actual_accepted_updates": 3,
      "actual_stop": "margin_normal_cap_stop",
      "actual_committed_trajectory": [
        {
          "kind": "actual_training_origin",
          "state_index": 0,
          "parameter_sha256": "98d40d9533caf91834488beec8dd0855f3c535cf34147db51935104bdaf626c9",
          "classes": {
            "M": {
              "original_class_mass": 59770,
              "errors": 806,
              "pure_errors": 700,
              "mixed_errors": 106,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.10447075203961108,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4503,
              "zero_recall_root_proxy_groups": 29,
              "equal_root_proxy_recall": 0.992133743584767,
              "remaining_fixed_errors": 806,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.648289829723257,
                "0.25": -1.9527197821831,
                "0.5": -0.8012836181131344,
                "0.75": -0.503695744327914,
                "1.0": -0.005638072668805116
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1.0": 0.0
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 32550,
              "errors": 892,
              "pure_errors": 886,
              "mixed_errors": 6,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.35553799754356386,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 148,
              "zero_recall_root_proxy_groups": 119,
              "equal_root_proxy_recall": 0.15102672954614701,
              "remaining_fixed_errors": 892,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.629678439040216,
                "0.25": -27.629323136563933,
                "0.5": -2.7103020603482326,
                "0.75": -1.108727439454151,
                "1.0": -0.011102726409339514
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0,
                "0.25": 0.0,
                "0.5": 0.0,
                "0.75": 0.0,
                "1.0": 0.0
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 152,
              "fixed_deep_remaining_errors": 152,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.648289829723257,
                "0.25": -27.648289829723257,
                "0.5": -27.648289829723257,
                "0.75": -27.648289829723257,
                "1": -27.64804299438859
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": -0.017268713794710067,
                "0.25": -0.017268713794710067,
                "0.5": -0.017268713794710067,
                "0.75": -0.017268713794710067,
                "1": -0.01702187846031933
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.63102111592827
              },
              "current_readout_M_S_margin_bound": 0.18977206594302368,
              "all_current_errors": 806,
              "all_current_pure_errors": 700,
              "full_original_class_mass": 59770
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 319,
              "fixed_deep_remaining_errors": 319,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.629678439040216,
                "0.25": -27.6294633007783,
                "0.5": -27.62939018048189,
                "0.75": -27.62929010235376,
                "1": -27.629001559585927
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.00134267688833134,
                "0.25": 0.0015578151502460003,
                "0.5": 0.0016309354466557124,
                "0.75": 0.0017310135747869992,
                "1": 0.002019556342620632
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.18977206594302368,
              "all_current_errors": 892,
              "all_current_pure_errors": 886,
              "full_original_class_mass": 32550
            }
          ]
        },
        {
          "kind": "actual_committed_training_state",
          "state_index": 1,
          "parameter_sha256": "cfc84d093192d1f96a520aed24326df046f1010cba90d8f63d488005750f16ca",
          "newly_repaired_original_rows": 0,
          "classes": {
            "M": {
              "original_class_mass": 59770,
              "errors": 806,
              "pure_errors": 700,
              "mixed_errors": 106,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.10462189144239485,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4503,
              "zero_recall_root_proxy_groups": 29,
              "equal_root_proxy_recall": 0.992133743584767,
              "remaining_fixed_errors": 806,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.403890169662265,
                "0.25": -1.775273124840935,
                "0.5": -0.7161001586517176,
                "0.75": -0.44949903680947156,
                "1.0": -0.005146251177293948
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0004918214915111685,
                "0.25": 0.05419670751844241,
                "0.5": 0.08582272978677308,
                "0.75": 0.17744665734216492,
                "1.0": 0.2449893708484563
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 32550,
              "errors": 892,
              "pure_errors": 886,
              "mixed_errors": 6,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.35631912593763704,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 148,
              "zero_recall_root_proxy_groups": 119,
              "equal_root_proxy_recall": 0.15102672954614701,
              "remaining_fixed_errors": 892,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.45716793715112,
                "0.25": -27.455289671417933,
                "0.5": -2.5538711596758157,
                "0.75": -1.0126467678521838,
                "1.0": -0.009845248275161644
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0007224104108927154,
                "0.25": 0.09736964559262273,
                "0.5": 0.15669183476897963,
                "0.75": 0.1734715369149038,
                "1.0": 0.17510320223858855
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 152,
              "fixed_deep_remaining_errors": 152,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.403890169662265,
                "0.25": -27.403890169662265,
                "0.5": -27.403890169662265,
                "0.75": -27.403890169662265,
                "1": -27.403701652487918
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.22713094626628205,
                "0.25": 0.22713094626628205,
                "0.5": 0.22713094626628205,
                "0.75": 0.22713094626628205,
                "1": 0.22731946344035237
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.63102111592827
              },
              "current_readout_M_S_margin_bound": 0.9142105556722068,
              "all_current_errors": 806,
              "all_current_pure_errors": 700,
              "full_original_class_mass": 59770
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 319,
              "fixed_deep_remaining_errors": 319,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.45716793715112,
                "0.25": -27.456228154160968,
                "0.5": -27.455829525538714,
                "0.75": -27.455202041247958,
                "1": -27.454288643008493
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.17385317877742779,
                "0.25": 0.1747929617675812,
                "0.5": 0.1751915903898329,
                "0.75": 0.17581907468058766,
                "1": 0.17673247292005456
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 0.9142105556722068,
              "all_current_errors": 892,
              "all_current_pure_errors": 886,
              "full_original_class_mass": 32550
            }
          ]
        },
        {
          "kind": "actual_committed_training_state",
          "state_index": 2,
          "parameter_sha256": "51697a3f8a5b4fbce5610cd3b870620668bfff96c77a3eb0d65ac5d08244edff",
          "newly_repaired_original_rows": 0,
          "classes": {
            "M": {
              "original_class_mass": 59770,
              "errors": 806,
              "pure_errors": 700,
              "mixed_errors": 106,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.10551152359837096,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4503,
              "zero_recall_root_proxy_groups": 29,
              "equal_root_proxy_recall": 0.992133743584767,
              "remaining_fixed_errors": 806,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.200505835397337,
                "0.25": -1.620497212844961,
                "0.5": -0.6429688484010352,
                "0.75": -0.4028089551575479,
                "1.0": -0.013836022683470506
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": -0.00819795001466539,
                "0.25": 0.10088678917036609,
                "0.5": 0.1592849037292079,
                "0.75": 0.3322225693381389,
                "1.0": 0.4511356795172965
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 32550,
              "errors": 892,
              "pure_errors": 886,
              "mixed_errors": 6,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.35635563573327267,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 148,
              "zero_recall_root_proxy_groups": 119,
              "equal_root_proxy_recall": 0.15102672954614701,
              "remaining_fixed_errors": 892,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.24696092202191,
                "0.25": -27.242820030211814,
                "0.5": -2.3665167791478594,
                "0.75": -0.9031302498112785,
                "1.0": -0.0020986366225033137
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.00778043547237206,
                "0.25": 0.20893875222000036,
                "0.5": 0.3450017731569912,
                "0.75": 0.38545526187090307,
                "1.0": 0.389145539954324
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 152,
              "fixed_deep_remaining_errors": 152,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.200505835397337,
                "0.25": -27.19715415020596,
                "0.5": -27.19715415020596,
                "0.75": -27.19715415020596,
                "1": -27.19715415020596
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.43051528053093335,
                "0.25": 0.43386696572258643,
                "0.5": 0.43386696572258643,
                "0.75": 0.43386696572258643,
                "1": 0.43386696572258643
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.63102111592827
              },
              "current_readout_M_S_margin_bound": 1.7683965131719026,
              "all_current_errors": 806,
              "all_current_pure_errors": 700,
              "full_original_class_mass": 59770
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 319,
              "fixed_deep_remaining_errors": 319,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.24696092202191,
                "0.25": -27.24459331019374,
                "0.5": -27.243523604053486,
                "0.75": -27.24271423423987,
                "1": -27.240170551874844
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.38406019390663815,
                "0.25": 0.38642780573480806,
                "0.5": 0.3874975118750612,
                "0.75": 0.38830688168867766,
                "1": 0.39085056405370366
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 1.7683965131719026,
              "all_current_errors": 892,
              "all_current_pure_errors": 886,
              "full_original_class_mass": 32550
            }
          ]
        },
        {
          "kind": "actual_committed_training_state",
          "state_index": 3,
          "parameter_sha256": "2590bb46548aa280b91c0bc9efb59210480873a5f14eb3c995185861baabdde4",
          "newly_repaired_original_rows": 0,
          "classes": {
            "M": {
              "original_class_mass": 59770,
              "errors": 806,
              "pure_errors": 700,
              "mixed_errors": 106,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.10620724215318583,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 4503,
              "zero_recall_root_proxy_groups": 29,
              "equal_root_proxy_recall": 0.992133743584767,
              "remaining_fixed_errors": 806,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -26.964489091251078,
                "0.25": -1.4405594047076486,
                "0.5": -0.5401477520559654,
                "0.75": -0.3265278336745575,
                "1.0": -0.001603392005312232
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": 0.0040346806634928845,
                "0.25": 0.17716791065335646,
                "0.5": 0.261135866057169,
                "0.75": 0.5121603774754513,
                "1.0": 0.6865156244358417
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            },
            "S": {
              "original_class_mass": 32550,
              "errors": 892,
              "pure_errors": 886,
              "mixed_errors": 6,
              "repairs_vs_previous_accepted": 0,
              "new_errors_vs_previous_accepted": 0,
              "repairs_vs_fixed_endpoint": 0,
              "new_errors_vs_fixed_endpoint_all_original_rows": 0,
              "new_mixed_errors_vs_fixed_endpoint": 0,
              "new_pure_errors_vs_fixed_endpoint": 0,
              "protected_regressions": 0,
              "stable_CE_mean": 0.35802858951142585,
              "fixed_endpoint_repair_distinct_locals": 0,
              "fixed_endpoint_repair_distinct_roots": 0,
              "largest_root_share_of_fixed_endpoint_repairs": null,
              "root_proxy_support_groups": 148,
              "zero_recall_root_proxy_groups": 119,
              "equal_root_proxy_recall": 0.15102672954614701,
              "remaining_fixed_errors": 892,
              "remaining_fixed_error_margin_quantiles": {
                "0.0": -27.041935606369513,
                "0.25": -27.030996126780025,
                "0.5": -2.1821165824080984,
                "0.75": -0.8042419464897115,
                "1.0": -0.010208014184078795
              },
              "remaining_fixed_error_margin_change_quantiles": {
                "0.0": -0.0021949479121932125,
                "0.25": 0.3121488776731385,
                "0.5": 0.5300169906700374,
                "0.75": 0.5956687119669581,
                "1.0": 0.607229888961907
              },
              "root_is_source_proxy_not_independent_attack_behavior": true
            }
          },
          "deep_error_burden": [
            {
              "class_name": "M",
              "fixed_deep_initial_error_rows": 152,
              "fixed_deep_remaining_errors": 152,
              "fixed_deep_actual_margin_quantiles": {
                "0": -26.964489091251078,
                "0.25": -26.961774205287416,
                "0.5": -26.961774205287416,
                "0.75": -26.961774205287416,
                "1": -26.961774205287416
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.666532024677192,
                "0.25": 0.6692469106411316,
                "0.5": 0.6692469106411316,
                "0.75": 0.6692469106411316,
                "1": 0.6692469106411316
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.63102111592827
              },
              "current_readout_M_S_margin_bound": 2.550195198630832,
              "all_current_errors": 806,
              "all_current_pure_errors": 700,
              "full_original_class_mass": 59770
            },
            {
              "class_name": "S",
              "fixed_deep_initial_error_rows": 319,
              "fixed_deep_remaining_errors": 319,
              "fixed_deep_actual_margin_quantiles": {
                "0": -27.041935606369513,
                "0.25": -27.035568556499108,
                "0.5": -27.03235969765967,
                "0.75": -27.03074093185203,
                "1": -27.02291422856394
              },
              "fixed_deep_learned_residual_margin_quantiles": {
                "0": 0.5890855095590339,
                "0.25": 0.5954525594294378,
                "0.5": 0.598661418268879,
                "0.75": 0.6002801840765191,
                "1": 0.608106887364606
              },
              "fixed_deep_frozen_prior_margin_quantiles": {
                "0": -27.631021115928547,
                "0.25": -27.631021115928547,
                "0.5": -27.631021115928547,
                "0.75": -27.631021115928547,
                "1": -27.631021115928547
              },
              "current_readout_M_S_margin_bound": 2.550195198630832,
              "all_current_errors": 892,
              "all_current_pure_errors": 886,
              "full_original_class_mass": 32550
            }
          ]
        }
      ],
      "temporary_V168_candidate": null,
      "mastery": false
    }
  ],
  "actual_committed_updates_total": 5,
  "V168_candidate_not_added_to_training_trajectory": true,
  "role1_frozen_deep_S_error_rows_still_wrong": 813,
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "no_new_training_authority": true,
  "classification_mastery": false,
  "full_quality_acceptance": false,
  "scope": "Saved supervised development trajectory; fixed NumPy prior means are diagnostic, not a new forward, acceptance tolerance or global model bound",
  "source_sha256": {
    "artifacts/v158_legal_fusion_bank_v2_20261001/fold0/OOF_probabilities.npy": "1a158fb881748593c16c5b3a90f42b33acb37ce2ea35969a6737f6747eef3e64",
    "artifacts/v158_legal_fusion_bank_v2_20261001/fold1/OOF_probabilities.npy": "a39938b15214da7e14dc25438776fb8ac9878ca2f1a0a940fcd77725eb622089",
    "artifacts/v158_legal_fusion_bank_v2_20261001/fold2/OOF_probabilities.npy": "eed286ab24f2ca8ab338076319804bd77f2d572a4ab5163915a363129bcb9c65",
    "artifacts/v159_class_boundary_numeric_trial_20261002/fold0_B/endpoint.pt": "49607ceb0aa6a586411f29f9e722fdd895d03a0ca58d54f6dcc7c64a5a178284",
    "artifacts/v159_class_boundary_numeric_trial_20261002/fold1_B/endpoint.pt": "25f856507c1a75277f60872c38e331646f3ace02a0b9b4497ed11571467f2d0f",
    "artifacts/v159_class_boundary_numeric_trial_20261002/fold2_B/endpoint.pt": "9382c79857d2d69af2fefbc0fdfdb8ee28c4e20e1c511efd6a6f67b53915f30f",
    "artifacts/v164_saved_training_quality_review_20261002/review.json": "f5738110eecebe9a113dc37f33e3c5c7fc4d25318aa5cbc16c26358df86c765c",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/accepted1/checkpoint.pt": "8a779d83fe67c35fb8398aad11ad42a6e800f115a101d7af0cd929c267ecde65",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/accepted1/commit.json": "eec258137535f05455747081a3c7a946dba0861240c4c893271394032154e3a0",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/accepted1/OOF_original_rows.parquet": "804329e9ee3ab4446981541d1a7a4056f106c6ec04fb19123e8dd5fc6afd707a",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/baseline/OOF_original_rows.parquet": "8e32d6a29bad9599c79adc4cd157aa28e854be8eceaee85927893d2386dfd971",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/fit.json": "ac66768be70d2a5bbd998ce4db10334f6bea574aa26a2242372db9c36971a442",
    "artifacts/v164_short_supervised_trajectory_20261002/role0/started.json": "2bdf25a69377724ed557cf2ccafcf7f082c9bad53fdab8b9f25464ec2c30d033",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/accepted1/checkpoint.pt": "2d25421fa934a6ce1f9e72d65e75c8483f63c0b57dbe4fd42d5633b9b208b600",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/accepted1/commit.json": "b327bb187b40446ca239f7b453a66b23665f5cb60f94ad8b3053969a3fc79553",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/accepted1/OOF_original_rows.parquet": "43c6e22d1de38732ba23647f4e4ec5411392d536975be7471f038d64591fb0c6",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/baseline/OOF_original_rows.parquet": "85fb3a0162c47437730e9f22666fd06b4b7cb0da837314a5288ffa0414be758e",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/endpoint.pt": "19c5f0061ca2d844695c15dd78afef37f615e9c8e075a60a1ea73ee9deb5a417",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/fit.json": "daa566091f79d7629240db5c1b7bb3be5118845d95a1ec3552678ae80d543911",
    "artifacts/v164_short_supervised_trajectory_20261002/role1/started.json": "64ca702b12af442582e4b20c8f164920aca3ae898575906b4eed2876359207e9",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted1/checkpoint.pt": "7c95239c2149a6a2c9d2c3cd8ba82a311d8bf1dd56cc9435bb6b4910276cfb53",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted1/commit.json": "13c033a492585fc20d3876bd735f0e8d789e14b2f474909cda1a5ab96043ae44",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted1/OOF_original_rows.parquet": "79555ace00cde3939bf6f5e0e9e1835e291ff39a3bab859b0dd863d848419388",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted2/checkpoint.pt": "15e42434ae038e61d20ff8eec97b1139d9ad6f1d4daf8bb091686beb3bb7998d",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted2/commit.json": "0f17a1cc4a3453c78461bf091f3c9585920a52c23d978354463191263d03f309",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted2/OOF_original_rows.parquet": "875945ac088b9487c037e9ecf6eceae075c6c41e79fd5f01e114560b7c8beb81",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted3/checkpoint.pt": "ff08ea90124a2ee669ce0e3379d144a14048f884195f874d48bec31828949440",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted3/commit.json": "3a991b4b7a7e1715fab49729b0bbea9ff48c9df7d6f861a23f67056561b671a4",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/accepted3/OOF_original_rows.parquet": "f244bec88946af4287307f3ec8dc6b124629db2293289002a7ed19ef5c2cf513",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/baseline/OOF_original_rows.parquet": "42681ad558cd488598251bce5cbcd7311d4e7cc3663ede8d307e2bbcfa1c6606",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/fit.json": "8d64df7636048219607275a8718bf65035bf0f3885522d63c2e94656e3337630",
    "artifacts/v164_short_supervised_trajectory_20261002/role2/started.json": "e224b6f5cceb5559b0fcffc99c5819c383bc89a950c437da6396b3ddfdd3f1fb",
    "artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0/direction.npy": "1543376edc5c333dae868c05380b13e87abfc9997b9b2530b3d24cb1aa8511d3",
    "artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0/OOF_original_rows.parquet": "aab660ceb93b866131d1eab651f72e2a78f1f277e4df2e62d353aaa94306b054",
    "artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0/v168_complete_probe_review.json": "63ad27bf09ae7e1646536d4458fb77049c95c5872801ed8afcba75fc0c6d78ef",
    "artifacts/v168_independent_actual_decision_floor_review_20261002/pre_review_bindings.json": "41e3451116649ad04871f4464bb6f795f926d40e0c76df66fdde31738b5c4ba2",
    "artifacts/v168_independent_actual_decision_floor_review_20261002/review.json": "d7a65add7e256235f3961dfd526418463af20851ffc9bc802ac8e89dbc13d602",
    "data/official/train.parquet": "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742",
    "training/v161_independent_all_finite_results_review.py": "2ca3ae6874c1ed743d854150e91cfbf2d0f5ddd23146baf6a3a7d8259642f79b",
    "training/v164_saved_training_quality_review.py": "e91f354704db5cd0a17c1332dc46124a87c0f9df77a93aa40d4e70b000a23aa5",
    "training/v168_root_saved_prior_burden_review.py": "dda5cecf9d1ce94f70972a0d85bb83215ca564361d3aad31f340e171a56066e1",
    "training/v168_saved_accepted_learning_trajectory_review.py": "6e4cd1c1a88c69d5fb03ba766f3d141a4466752054158b0b1da9096c1ae213bd"
  }
}

```

来源 `artifacts/v169_saved_prospective_budget_review_20261002/review.json`；SHA256 `2f8ccf497e5ed8f1f98bc50bb33a1e3a2cf80ed123cd2d396c0b25477f79acc2`。

```json
{
  "status": "V169_prospective_call_budget_and_parameter_identity_requirements_saved_reviewed",
  "execution_authority": false,
  "new_fit_permission": false,
  "current_cumulative_heads": 19178,
  "current_cumulative_complete_derivatives": 768,
  "inherited_cumulative_head_ceiling": 82174,
  "inherited_cumulative_derivative_ceiling": 2426,
  "remaining_heads": 62996,
  "remaining_complete_derivatives": 1658,
  "design_max_fits": 6,
  "design_max_accepted_updates": 120,
  "conditional_all_120_points_two_targets_twice_derivatives": 480,
  "conditional_remaining_margin_derivatives_after480_targets": 1178,
  "naive_25_functions_twice_two_corrections_all120_points_margin_derivatives": 12000,
  "naive_all120_points_total_derivatives": 12480,
  "naive_dense_remeasure_schedule_exceeds_existing_remaining_budget": true,
  "actual_callgraph_not_yet_implemented_or_registered": true,
  "arithmetic_is_conditional_not_a_lower_bound_for_every_legal_bootstrap_schedule": true,
  "required_before_execution": [
    "Define whether safe initialization submission counts within20 accepted updates; preserve actual commit ledger and120 total cap",
    "Implement both arms with exact counted calls including bootstrap replay, gradient repeats, dynamic working set, finite scans, refusals and restoration",
    "B adds beta: parameter width1060833 versusA1060832; old fixed-width gradient/repeat helpers must not omit beta or certify stale parameter points",
    "Dynamic working set must keep complete permanent protection ledger; all discarded constraints remain checked on every actual finite candidate",
    "Budget projection must fit inherited cumulative ceilings without hidden derivatives or reset; otherwise prospectively revise equal-arm schedule or supported resources",
    "If one arm stops technically before its matched fixed terminal, predeclare an inconclusive paired result; do not choose a historical checkpoint or claim B benefit from more exposure"
  ],
  "official_heads": 0,
  "official_features": 0,
  "official_derivatives": 0,
  "fits": 0,
  "permanent_updates": 0,
  "no_claim_beta_method_is_infeasible": true,
  "no_method_effect_claim": true,
  "source_sha256": {
    "training/v169_saved_prospective_budget_review.py": "5cb2692aba9b7e716c3bde24bd7f1d4995531f6c76363d616c4f7df31f8fa907",
    "training/review_policy/v169_learnable_prior_pair_draft.json": "109272638d24ff36eb9b9f8a0d301487d2b943c19a57a72947fba54c84cc18e5",
    "artifacts/v168_results_20261002/actual_result_summary.json": "bc27e7273fb8612ef3221bb7507702c0a4ac6ce2f7bd69fad784a79b072a19b9",
    "training/v168_decision_floor_execution_review.py": "9208df8ed1df68c40e2af23134f2f6381203af3a6f657a7a6208aac9796a407d",
    "artifacts/v168_saved_accepted_learning_trajectory_review_20261002/review.json": "c947fd6fdf415dea4ecf09b1c4945947f0588dc4118e809ecede683b73c43d47"
  }
}

```

五项V168实际边界回放：`artifacts/v168_observed_runtime_boundaries_20261002/replay.json`。预算核查为条件性算术，不是证明方法不可行；必须先明确安全初始化提交的更新计数、B新增beta的完整梯度/重复策略、逐候选完整保护以及配对技术中止的解释，不能复用旧固定宽度检查遗漏新参数。

## 发布前当前限制全部保留

- 三个目标及完整质量未完成；当前只通过一次有限诊断，最新训练V164、完整交付V159；全部旧限制见当前附录。

## 既有证据限制完整保留

只读目录接近既有256KiB限制；本次将项目级旧版本限制原文移入本附录，并在当前状态保留直接影响下一步的限制。所有旧文档ID、路径、标题、sha和元数据原样保留；原目录另存快照，不增加工具上限。以下每条来自更新前项目known_limits，均保留：

- V159前3评价未存全q：仅FIT重放行/source及全q精确argmax断言控制流；冻结完整q来源另列。V158/当前已看开发来源，非盲验/外部部署；N无新监督。
- 训练分类保留和总目标下降不替代来源外分类。
- 原578困难S登记辅助配对直接覆盖0；本轮无新同类标签。
- 正常/其他格式冻结107错，无晋升；预登记A0质量门槛保持。
- 原末层12、V142三、V146六拟合均不自动追加。
- 解析facts不是原始正文或模型全部输入；最近字段差异不授权删除判别信息。
- V148共同观察字段零距离不等于全输入或威胁类别相同。
- V150实际解码拟合4053504系数，未更新分类器；基座见内层TRAIN/留出标签，受限线性不证明独立泛化、威胁判据或信息不可恢复。
- V151产品/facility/匿名实体/属性重复只诊断来源，尚无新可信M/S上下文。
- V151首版key过宽，另版复核当前路径；50->116含包装规范变化，非全因日期。
- V152两次合法TRAIN非新增独立样本；historical outer正确标记不认证对应TRAIN角色。
- V152根是隔离代理，多根计数/集中度不证明真实攻击条件；精确缺支持不排除组成泛化。
- V153父本轮只绑定保存概率，不新前向；TRAIN与historical outer概率角色分开。
- 225202 pure TRAIN角色记录与225558正确保护记录不同；ACL标签不是实际策略绑定。
- V154一半径三个固定方向不是最大球内尖锐性、曲率或因果证明；不替代随机SAM实训。
- V154全部ASA为已看开发折；无其他格式全量新模型验收，损失变化不等于分类掌握变化。
- V155是full-original-frequency一阶邻域适配，非随机SAM作者benchmark复现，B额外计算不称等算力。
- V155实际6拟合/1800梯度，9零步梯度另计；完整质量失败，无新独立同类支持/新环境泛化确认。
- V156仅固定表示条件几何；H2余弦未额外独立重算，完整分类还含head及facts通路。
- V157仅复现固定V146 A完整函数；不证明跨来源因果、不增标签/模型收益。初次CPU ICMP空切片另版纠正，原结果保存。
- V158正文32维CountSketch有损，A/B容量208/8640不同，旧OVA softmax不宣称校准。
- V158 N1没有每接受状态完整类别/来源轨迹；当前22/6/28保护不是追加N1分数后22/4/26完整可观测下限的学习证明。
- V158部署FIT掌握不证明合法OOF各类已学会；未来独立元验证不能复用已见验证根标签的基模型/守卫。
- V159质量失败；V161/162有限资格不证明分类掌握，无新拟合。





