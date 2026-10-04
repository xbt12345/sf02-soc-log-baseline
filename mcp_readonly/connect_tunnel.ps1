param([string]$TunnelId)
$ErrorActionPreference = 'Stop'
$runtime = Join-Path $PSScriptRoot '.runtime'
$client = Join-Path $runtime 'bin\tunnel-client.exe'
if (-not (Test-Path -LiteralPath $client)) { throw 'Official tunnel-client is missing. See CONNECT_CHATGPT.md.' }
if ((Get-FileHash -LiteralPath $client -Algorithm SHA256).Hash.ToLower() -ne 'fcc85a69ec0ad82518e4f8964f60c45e31787957782a0fc9c1b0c44e82d61b9b') { throw 'tunnel-client differs from the verified v0.0.14 binary. Re-verify the official release before running.' }
& (Join-Path $PSScriptRoot 'start_local.ps1')
if (-not $TunnelId) { $TunnelId = Read-Host 'Tunnel ID from OpenAI Platform (tunnel_...)' }
if ($TunnelId -notmatch '^tunnel_[0-9a-f]{32}$') { throw 'Invalid Tunnel ID. Use the ID shown in Platform.' }
$previousKey = $env:CONTROL_PLANE_API_KEY
$keyBuffer = [IntPtr]::Zero
try {
    $secret = Read-Host 'Runtime API key (Tunnels Read + Use; hidden input; not saved)' -AsSecureString
    $keyBuffer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    $env:CONTROL_PLANE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyBuffer)
    if ([string]::IsNullOrWhiteSpace($env:CONTROL_PLANE_API_KEY)) { throw 'API key cannot be empty.' }
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyBuffer)
    $keyBuffer = [IntPtr]::Zero
    $common = @('--control-plane.api-key','env:CONTROL_PLANE_API_KEY','--control-plane.tunnel-id',$TunnelId,'--mcp.server-url','http://127.0.0.1:8765/mcp')
    & $client doctor @common --explain
    if ($LASTEXITCODE -ne 0) { throw 'Tunnel doctor failed. Fix the reported account/network/permission issue first.' }
    Write-Host 'Keep this window open. In another terminal run mcp_readonly\check_tunnel.cmd, then add the Tunnel in ChatGPT Settings > Apps.'
    & $client run @common --health.listen-addr '127.0.0.1:8766' --health.url-file (Join-Path $runtime 'tunnel-health.url')
    if ($LASTEXITCODE -ne 0) { throw 'Tunnel process exited unsuccessfully. Read its diagnostic output.' }
} finally {
    if ($keyBuffer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyBuffer) }
    $env:CONTROL_PLANE_API_KEY = $previousKey
    if ($secret) { $secret.Dispose() }
}
