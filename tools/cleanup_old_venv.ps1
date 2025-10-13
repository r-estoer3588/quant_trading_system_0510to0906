param(
    [string]$Path = "c:\Repos\quant_trading_system\venv_old_20251013"
)

Write-Host "[cleanup_old_venv] Target: $Path" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $Path)) {
    Write-Host "Already absent: $Path" -ForegroundColor Green
    exit 0
}

try {
    # 1) Kill processes locking files under the path (by process Path prefix)
    Write-Host "Stopping processes that may lock files..." -ForegroundColor Yellow
    $procs = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($Path, [System.StringComparison]::OrdinalIgnoreCase) }
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
    }

    Start-Sleep -Seconds 1

    # 2) Clear attributes (read-only/hidden) to reduce deletion errors
    Write-Host "Clearing file attributes..." -ForegroundColor Yellow
    Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue |
    ForEach-Object { try { $_.Attributes = 'Normal' } catch {} }

    # 3) Try delete
    Write-Host "Removing folder..." -ForegroundColor Yellow
    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue

    if (Test-Path -LiteralPath $Path) {
        # 4) Fallback: rename for pending deletion
        $suffix = (Get-Date -Format 'yyyyMMdd_HHmmss')
        $parent = Split-Path -Parent $Path
        $name = Split-Path -Leaf $Path
        $renamed = Join-Path $parent ("${name}_pending_delete_${suffix}")
        try {
            Rename-Item -LiteralPath $Path -NewName (Split-Path -Leaf $renamed) -ErrorAction SilentlyContinue
        }
        catch {}

        if (Test-Path -LiteralPath $renamed) {
            Write-Host "Renamed to: $renamed (delete later after closing handles)" -ForegroundColor Yellow
            exit 0
        }
        else {
            Write-Host "Folder still present and rename failed. Close open handles and retry." -ForegroundColor Yellow
            exit 1
        }
    }
    else {
        Write-Host "Deleted: $Path" -ForegroundColor Green
        exit 0
    }
}
catch {
    Write-Host "Unexpected error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
