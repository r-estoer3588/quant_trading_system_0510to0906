param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
    param()
    if ($PSScriptRoot) {
        return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    }
    $scriptPath = $MyInvocation.MyCommand.Path
    if ($scriptPath) {
        $scriptDir = Split-Path -Parent $scriptPath
        return (Resolve-Path (Join-Path $scriptDir '..')).Path
    }
    # Fallback: current working directory
    return (Resolve-Path (Get-Location)).Path
}

function Get-Policy {
    param([string]$Root)
    $policyPath = Join-Path $Root 'tools/command_policy.json'
    if (!(Test-Path $policyPath)) { throw "Policy file not found: $policyPath" }
    $json = Get-Content -Raw -Path $policyPath | ConvertFrom-Json
    return $json
}

function New-LogPath {
    param([string]$Root, [string]$RelPath)
    $logPath = Join-Path $Root $RelPath
    $logDir = Split-Path -Parent $logPath
    if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    return $logPath
}

function Write-AuditLog {
    param([string]$LogPath, [string]$Cmd, [string]$Decision, [string]$Reason)
    $entry = [ordered]@{
        timestamp = (Get-Date).ToString('o')
        user      = $env:USERNAME
        host      = $env:COMPUTERNAME
        cwd       = (Get-Location).Path
        command   = $Cmd
        decision  = $Decision
        reason    = $Reason
    } | ConvertTo-Json -Compress
    Add-Content -LiteralPath $LogPath -Value $entry
}

$root = Get-RepoRoot
$policy = Get-Policy -Root $root
$cmdLine = ($Command -join ' ').Trim()
$logPath = New-LogPath -Root $root -RelPath $policy.logPath

# Evaluate deny patterns (regex, case-insensitive)
$blocked = $false
$matchedPattern = $null
foreach ($pattern in $policy.denyPatterns) {
    if ($cmdLine -match $pattern) {
        $blocked = $true
        $matchedPattern = $pattern
        break
    }
}

if ($blocked) {
    Write-AuditLog -LogPath $logPath -Cmd $cmdLine -Decision 'blocked' -Reason ($policy.denyReason + " (pattern: $matchedPattern)")
    Write-Host "[safe_exec] BLOCKED: $($policy.denyReason)" -ForegroundColor Red
    Write-Host "Matched pattern: $matchedPattern" -ForegroundColor Yellow
    exit 3
}

Write-AuditLog -LogPath $logPath -Cmd $cmdLine -Decision 'allowed' -Reason 'policy allow (non-destructive)'
Write-Host "[safe_exec] Running: $cmdLine" -ForegroundColor Cyan

# Execute directly with call operator to preserve argument boundaries
$exe = $Command[0]
$argv = @()
if ($Command.Count -gt 1) { $argv = $Command[1..($Command.Count - 1)] }

& $exe @argv
$code = $LASTEXITCODE
exit $code
