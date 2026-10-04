$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-mcp\Scripts\python.exe'
$runtime = Join-Path $PSScriptRoot '.runtime'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv-mcp. Follow README installation steps.' }
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MCP_HOST = '127.0.0.1'
$env:MCP_PORT = '8765'
if ($env:MCP_READONLY_TOKEN) { throw 'This private Tunnel workflow expects a loopback MCP without a separate bearer token. Use a clean PowerShell window.' }
$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
if ($listener -and ($listener | Where-Object { $_.LocalAddress -ne '127.0.0.1' })) { throw 'Port 8765 is not restricted to IPv4 loopback. Inspect it before continuing.' }
if (-not $listener) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $process = Start-Process -FilePath $python -ArgumentList '-B','-m','mcp_readonly.server' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtime "server_$stamp.out.log") -RedirectStandardError (Join-Path $runtime "server_$stamp.err.log")
    $process.Id | Set-Content -LiteralPath (Join-Path $runtime 'server.pid') -Encoding ascii
    $ready = $false
    for ($i=0; $i -lt 40; $i++) {
        if ($process.HasExited) { throw "MCP exited. Check logs in $runtime" }
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 1
            if ($health.project -eq 'SF02' -and $health.mode -eq 'read-only') { $ready=$true; break }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) { throw 'Local MCP did not become ready; inspect runtime logs.' }
}
Push-Location $projectRoot
try {
    & $python -B -m mcp_readonly.check_connection
    if ($LASTEXITCODE -ne 0) { throw 'Local MCP contract check failed.' }
} finally { Pop-Location }
Write-Host 'Local MCP ready: http://127.0.0.1:8765/mcp. This alone does not connect ChatGPT.'
