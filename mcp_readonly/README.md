# SF02 项目只读 MCP Server

这个服务供 ChatGPT / GPT-6 Pro 查询 SF02 项目的当前进度、实验结论、证据和下一步方向。安全边界由服务端实施：调用者只能使用预定义资料 ID，不能提交文件路径、执行命令、读取原始数据、下载模型权重或修改项目。

当前进度已更新至 2026-10-04，最新方向 ID 为 `v169-platform-package-ready`，实际完整交付 ID 仍为 `v159-delivery`。GitHub 只保存源码、报告与必要摘要；历史白名单中的 `artifacts/` 文件依赖本地完整产物。克隆仓库后须先恢复这些文件并核验哈希，才能把所有历史查询作为可用服务；完整副本的现有只读接口与安全限制保持不变。

## 暴露的工具

- `search(query)`：搜索 `catalog.json` 明确列出的资料。
- `fetch(id)`：读取 `search` 返回的稳定资料 ID。
- `get_project_status()`：读取最新交付证据中的状态，并返回经审查的当前方向。

三个工具都声明 `readOnlyHint=true`、`destructiveHint=false`、`idempotentHint=true`、`openWorldHint=false`。这些标记帮助 ChatGPT 正确展示工具；真正的权限限制来自服务器没有写工具、没有路径参数，以及每次读取都执行白名单、路径、类型、大小和 SHA-256 校验。

## 本机运行

在项目根目录执行：

```powershell
py -3.12 -m venv .venv-mcp
.\.venv-mcp\Scripts\python.exe -m pip install -r .\mcp_readonly\requirements.txt
.\.venv-mcp\Scripts\python.exe -m mcp_readonly.server
```

默认地址：

- 浏览器说明页：`http://127.0.0.1:8765/`
- MCP：`http://127.0.0.1:8765/mcp`
- 健康检查：`http://127.0.0.1:8765/health`

浏览器地址栏访问 `/mcp` 不能验证 MCP 调用，因为它是给 MCP 客户端使用的协议端点。浏览器检查请使用 `/` 或 `/health`。

默认只监听回环地址。本机以外的绑定必须设置 `MCP_READONLY_TOKEN`，否则服务拒绝启动：

```powershell
$env:MCP_HOST = "0.0.0.0"
$env:MCP_PORT = "8765"
$env:MCP_PUBLIC_BASE_URL = "https://your-mcp.example.com"
$env:MCP_READONLY_TOKEN = "在密码管理器中生成并保存的长随机值"
.\.venv-mcp\Scripts\python.exe -m mcp_readonly.server
```

静态 Bearer token 适合受控开发链路的额外保护。公网正式部署应在反向代理或托管平台上提供 TLS，并按 OpenAI/MCP 的当前要求配置 OAuth 2.1 或 OpenAI 管理的 mTLS；不要把裸 HTTP 端口直接暴露到公网。

## 连接 ChatGPT Pro

当前采用官方私有 Secure MCP Tunnel。完整步骤和一键启动入口见 [CONNECT_CHATGPT.md](CONNECT_CHATGPT.md)。先在 Platform 创建 Tunnel 并关联工作区，再运行 `mcp_readonly\connect_tunnel.cmd`，最后在 ChatGPT 创建并选中应用。输入 `/mcp` 不会自动连接本机服务。

不要假设私有 Tunnel 同时转发普通 `/source` 网页；ChatGPT 通过 `fetch` 读取完整证据。本机证据 URL 仅用于本机浏览。如果以后正式部署网站和 OAuth，再设置其实际 HTTPS 根地址为 `MCP_PUBLIC_BASE_URL`。

## 更新白名单

当项目出现新的正式审查或交付证据时：

1. 只把需要向 ChatGPT 公开的 UTF-8 `.md`、`.txt` 或 `.json` 文件加入 `catalog.json`。
2. 使用 `Get-FileHash -Algorithm SHA256 <file>` 更新对应哈希。
3. 更新 `project.as_of`、摘要和方向，并运行测试。
4. 不加入 `data/`、模型权重、环境变量、凭据、临时结果或任意目录通配规则。

目录不是自动扫描的；未在 `catalog.json` 中逐项登记的文件不可读取。

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv-mcp\Scripts\python.exe -m unittest discover -s mcp_readonly\tests -v
```

测试覆盖 MCP 工具清单和只读标记、正常检索、未知 ID 与路径穿越拒绝、哈希失配拒绝、非本机无密钥拒绝，以及调用前后的项目文件哈希不变。


历史目录现在保存在 `catalog_archive/v167_prior.json`，保留原447条 ID、路径、哈希和元数据。`catalog.json`通过一个固定哈希引用它；读取器合并后提供相同search/fetch/status。每个目录和资料文件仍限制256KiB，不允许递归归档、越界路径或重复ID。API操作、只读边界与资料白名单不变。
