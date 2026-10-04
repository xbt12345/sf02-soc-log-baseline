# 把 SF02 只读 MCP 接入 ChatGPT

2026-09-19。当前本机 MCP 已通过真实 HTTP 客户端检查；账号侧 Tunnel 和 ChatGPT 工具调用尚未验证。

## 原因

`127.0.0.1:8765` 只在这台电脑上可达。在 ChatGPT 消息里输入 `/mcp` 不会安装连接器。本机 `/health` 返回 200 也不代表 ChatGPT 已经挂载工具。完整链路是：ChatGPT 中选择已创建的应用 → OpenAI Secure MCP Tunnel → 本机 tunnel-client → 本机只读 MCP。

上次 README 把“创建 Tunnel”和“填写 HTTPS 地址”讲得过于简略。官方 Tunnel 需要账号侧 Tunnel ID、运行密钥和工作区关联；不能凭空由 localhost 地址获得。静态 Bearer 密钥也不能直接当作 OAuth 配置。

## 你现在只需要完成账号侧设置

1. 打开 https://platform.openai.com/settings/organization/tunnels ，创建名为 `SF02 read-only` 的 Tunnel。把它关联到你要使用的 ChatGPT 工作区；个人账号使用相应个人 Platform organization。复制 `tunnel_` 开头的 ID。
2. 在 https://platform.openai.com/settings/organization/api-keys 创建运行用 API key。运行身份需要目标 Tunnel 的 **Read + Use** 权限；创建/管理 Tunnel 的身份另需 **Read + Manage**。不要把 Admin API key 当长期运行密钥，不需要向聊天窗口发送密钥。
3. 在项目根目录的 CMD 或 PowerShell 执行：

   ```text
   .\mcp_readonly\connect_tunnel.cmd
   ```

   它会检查/启动本机服务、要求输入 Tunnel ID、隐藏输入 API key、运行官方 `doctor`，然后在当前窗口运行 Tunnel。密钥仅进入该进程环境，不写到脚本、配置或命令行参数。保持这个窗口打开；关闭/按 Ctrl+C 会断开 Tunnel，本机 MCP 仍可能运行。
4. 在另一个终端执行 `mcp_readonly\check_tunnel.cmd`，检查 `/readyz`。也可打开 http://127.0.0.1:8766/ui 查看官方客户端状态。ready 只证明隧道状态，还需完成下一步。
5. ChatGPT 网页 Settings → Apps → Advanced settings 开启 Developer mode，Create app。按页面的 Secure MCP Tunnel 连接选项选择刚创建的 Tunnel；若页面要求 endpoint，使用 Platform 提供的地址，不要填本机 `127.0.0.1`，也不要自己拼 Tunnel URL。此流程本机 MCP 没有 OAuth，认证选项选择 No authentication；私有通道的访问由 OpenAI Tunnel 账号权限管理。不得将该配置改成无认证的公开隧道。
6. Scan tools 应能看到 `search`、`fetch`、`get_project_status`，创建完成后，在新消息中用应用选择菜单或 `@应用名` 明确选中它。输入：

   > 请实际调用 SF02 read-only 的 get_project_status，再 fetch 返回的证据 ID。报告工具返回的 as_of、experiment_status、quality_acceptance 和证据 SHA-256；如果没挂载工具，请明确说明，不要用历史对话代替实时查询。

只有第 6 步产生真实工具调用结果，才算 ChatGPT 端到端接入成功。每次需要新数据时重新选择/提及该应用。Pro 的 read/fetch 支持与模型名称是不同层面的能力；以你的账户界面和实际工具调用为准。

## 失败定位

| 现象 | 优先检查 |
|---|---|
| 8765 拒绝连接 | `powershell.exe -NoProfile -File mcp_readonly\start_local.ps1`；日志在 `.runtime` |
| doctor 提示 401/403 | 运行密钥、对应 organization、Tunnels Read/Use；勿发送密钥给聊天 |
| ChatGPT 列表看不到 Tunnel | 工作区关联是否正确、当前账号是否有 Use 权限、Tunnel 是否 ready |
| 本机 ready，但 Scan tools 失败 | 保留实际错误码，核对账号侧 discovery；不能靠关闭鉴权或公开端口“修复” |
| 会话说没有工具 | 应用是否创建成功、是否在这条消息选中、当前模式是否支持应用 |

官方 Tunnel 是 MCP 通道，不能假设会把 `/source/...` 当普通网站发布。完整证据应通过 `fetch(id)` 读取；本机 source URL 仅用于本机浏览，不能当作已验证的公网引用链接。

服务不会执行训练或改文件。Catalog 仍只允许明确的报告；本机 Python 进程并非操作系统沙箱，如需防御服务进程本身被攻破，另需专用只读账号/容器挂载，这不应与工具层只读混淆。

## 已准备的软件和验证边界

- 本地目录 `.runtime/bin`：OpenAI 官方 `tunnel-client` v0.0.14；包含的 cloudflared 没有启动。
- 官方发布包：`https://github.com/openai/tunnel-client/releases/download/v0.0.14/tunnel-client-v0.0.14-windows-amd64.zip`
- SHA-256：`784ab8da7b5a88f0109f1fd8aaf0a1c86067430b896dddf307ef7e3cc49fa1a5`，已与发布方 `SHA256SUMS.txt` 核对。
- 已验证版本、CLI 参数、PowerShell 语法、真实本机 search/fetch/status 调用及返回的 JSON text/structuredContent 一致性。
- 没有账号 Tunnel ID/运行密钥时，无法执行真实 Tunnel 鉴权与 ChatGPT Scan tools。这一步仍待账号侧完成。

官方依据：[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)、[客户端 onboarding](https://github.com/openai/tunnel-client/blob/master/docs/onboarding.md)、[ChatGPT 开发者模式](https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-apps-in-chatgpt)。界面入口可能随账号和工作区不同。
