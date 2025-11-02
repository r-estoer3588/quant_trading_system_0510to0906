# PowerShell strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# タスク名
[string]$taskName = 'n8n Docker Startup'

# タスクの説明
[string]$taskDescription = 'Auto-start n8n Docker container at Windows logon'

# 起動用スクリプトをファイルとして配置（引用符/展開の問題を避けるため）
$scriptDir = Join-Path $env:ProgramData 'n8n'
$scriptPath = Join-Path $scriptDir 'n8n-start.ps1'
New-Item -ItemType Directory -Force -Path $scriptDir | Out-Null

$scriptContent = @'
docker start n8n
if ($LASTEXITCODE -ne 0) {
	docker run -d --name n8n -p 5678:5678 -v $env:USERPROFILE\.n8n:/home/node/.n8n -e DB_SQLITE_POOL_SIZE=5 -e N8N_RUNNERS_ENABLED=true -e N8N_BLOCK_ENV_ACCESS_IN_NODE=false -e N8N_GIT_NODE_DISABLE_BARE_REPOS=true n8nio/n8n
}
'@

Set-Content -Path $scriptPath -Value $scriptContent -Encoding UTF8

# アクション（-File により堅牢な実行に）
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`""

# 管理者権限チェック（管理者必須に変更）
$isAdmin = try {
    $wi = [Security.Principal.WindowsIdentity]::GetCurrent()
    $wp = New-Object Security.Principal.WindowsPrincipal($wi)
    $wp.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
catch { $false }

if (-not $isAdmin) {
    Write-Error 'このタスクをシステム起動で実行するには管理者権限が必要です。PowerShellを管理者で再実行してください。'
    exit 1
}

# 起動タイミング: システム起動時（AtStartup）
$delaySeconds = 30
$trigger = New-ScheduledTaskTrigger -AtStartup
try { $trigger.Delay = "PT${delaySeconds}S" } catch { }

# 設定
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# 既存タスクがあれば置き換え
try {
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
    }
}
catch {
    # 取得失敗は無視（存在しない等）
}

# タスク作成（SYSTEM アカウントで最高特権）
try {
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal
    Register-ScheduledTask -TaskName $taskName -InputObject $task -Description $taskDescription -ErrorAction Stop
    Write-Host 'Task created successfully. n8n will auto-start at system startup.'
}
catch {
    Write-Error ("Failed to create task: {0}. Ensure you are running PowerShell as Administrator." -f $_.Exception.Message)
    exit 1
}
