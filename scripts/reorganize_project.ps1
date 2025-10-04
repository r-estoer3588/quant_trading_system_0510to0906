# プロジェクト再編成スクリプト
# quant_trading_system 整理・最適化スクリプト

param(
    [switch]$DryRun = $false,
    [switch]$Verbose = $false,
    [switch]$CreateBackup = $true
)

$ErrorActionPreference = "Stop"
$startTime = Get-Date

function Write-Info {
    param($Message)
    if ($Verbose) {
        Write-Host "[INFO] $Message" -ForegroundColor Cyan
    }
}

function Write-Progress {
    param($Phase, $Message)
    Write-Host "`n=== $Phase ===" -ForegroundColor Green
    Write-Host $Message -ForegroundColor White
}

function Move-SafelyTo {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path $Source)) {
        Write-Info "Source not found: $Source"
        return
    }

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would move: $Source -> $Destination" -ForegroundColor Yellow
        return
    }

    $destDir = Split-Path -Parent $Destination
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        Write-Info "Created directory: $destDir"
    }

    try {
        Move-Item -Path $Source -Destination $Destination -Force
        Write-Info "Moved: $Source -> $Destination"
    }
    catch {
        Write-Warning "Failed to move $Source to $Destination : $_"
    }
}

function Remove-SafelyFile {
    param([string]$FilePath)

    if (-not (Test-Path $FilePath)) {
        Write-Info "File not found for removal: $FilePath"
        return
    }

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would remove: $FilePath" -ForegroundColor Yellow
        return
    }

    try {
        Remove-Item -Path $FilePath -Force
        Write-Info "Removed: $FilePath"
    }
    catch {
        Write-Warning "Failed to remove $FilePath : $_"
    }
}

# ==========================================
# Phase 1: バックアップ作成
# ==========================================
if ($CreateBackup -and -not $DryRun) {
    Write-Progress "Phase 1" "Creating backup before reorganization"

    $backupName = "quant_trading_system_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    $backupPath = "C:\Backup\$backupName"

    if (-not (Test-Path "C:\Backup")) {
        New-Item -ItemType Directory -Path "C:\Backup" -Force | Out-Null
    }

    Write-Host "Creating backup at: $backupPath" -ForegroundColor Cyan
    robocopy . $backupPath /E /XD .git __pycache__ .venv node_modules /XF *.tmp *.log > $null

    if ($LASTEXITCODE -le 8) {  # robocopy success codes
        Write-Host "✓ Backup created successfully" -ForegroundColor Green
    } else {
        Write-Error "Backup creation failed"
    }
}

# ==========================================
# Phase 2: 整理ディレクトリの作成
# ==========================================
Write-Progress "Phase 2" "Creating organizational directories"

$newDirs = @(
    "backup/scripts",
    "backup/core",
    "backup/common",
    "backup/tests",
    "test_coverage",
    "tools/debug",
    "tools/maintenance",
    "tools/analysis",
    "docs/architecture",
    "docs/guides",
    "docs/api",
    "docs/internal/deprecated"
)

foreach ($dir in $newDirs) {
    if (-not (Test-Path $dir) -and -not $DryRun) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Info "Created directory: $dir"
    }
}

# ==========================================
# Phase 3: バックアップファイルの移動
# ==========================================
Write-Progress "Phase 3" "Moving backup and deprecated files"

$backupMoves = @(
    @("scripts/run_all_systems_today.py.backup", "backup/scripts/"),
    @("core/system5_backup.py", "backup/core/"),
    @("core/system3_backup.py", "backup/core/"),
    @("core/system4_backup.py", "backup/core/"),
    @("core/system6_backup.py", "backup/core/"),
    @("core/system7_backup.py", "backup/core/"),
    @("common/cache_manager_old.py", "backup/common/"),
    @("test_fixed_function.py", "backup/tests/"),
    @("test_new_cache_format.py", "backup/tests/"),
    @("debug_duplicate_columns.py", "tools/debug/"),
    @("debug_prepare_rolling.py", "tools/debug/")
)

foreach ($move in $backupMoves) {
    if (Test-Path $move[0]) {
        $fileName = Split-Path -Leaf $move[0]
        Move-SafelyTo -Source $move[0] -Destination "$($move[1])$fileName"
    }
}

# ==========================================
# Phase 4: テストカバレッジファイルの移動
# ==========================================
Write-Progress "Phase 4" "Moving test coverage files"

$coveragePatterns = @("htmlcov*", ".coverage*", "coverage.xml")
foreach ($pattern in $coveragePatterns) {
    $files = Get-ChildItem -Path . -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Move-SafelyTo -Source $file.FullName -Destination "test_coverage/$($file.Name)"
    }
}

# ==========================================
# Phase 5: 不要ファイルの削除（慎重に）
# ==========================================
Write-Progress "Phase 5" "Removing unnecessary files"

$unnecessaryFiles = @(
    "*.tmp",
    "*.temp",
    "*.bak",
    "Thumbs.db",
    ".DS_Store"
)

foreach ($pattern in $unnecessaryFiles) {
    $files = Get-ChildItem -Path . -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Remove-SafelyFile -FilePath $file.FullName
    }
}

# ==========================================
# Phase 6: ログファイルの整理
# ==========================================
Write-Progress "Phase 6" "Organizing log files"

if (Test-Path "logs") {
    $oldLogs = Get-ChildItem -Path "logs" -Filter "*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }

    if ($oldLogs.Count -gt 0) {
        $archiveDir = "logs/archive"
        if (-not (Test-Path $archiveDir) -and -not $DryRun) {
            New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
        }

        foreach ($log in $oldLogs) {
            Move-SafelyTo -Source $log.FullName -Destination "$archiveDir/$($log.Name)"
        }
        Write-Info "Archived $($oldLogs.Count) old log files"
    }
}

# ==========================================
# Phase 7: .gitignore の更新
# ==========================================
Write-Progress "Phase 7" "Updating .gitignore"

$gitignoreAdditions = @"

# === Reorganization Additions ===

# Backup directories (contains old/deprecated code)
backup/
*.backup
*.bak
*.old

# Test coverage reports
test_coverage/
htmlcov*/
.coverage*
coverage.xml
*.cover

# Temporary experimental files
tests/experimental/disabled/
**/experimental_*.py

# Performance profiling
*.prof
*.stats
profile_results/

# Temporary data
temp/
tmp/
*.tmp

# OS specific
Thumbs.db
.DS_Store

# Development tools output
logs/archive/
"@

if (-not $DryRun) {
    Add-Content -Path ".gitignore" -Value $gitignoreAdditions -Encoding UTF8
    Write-Info "Updated .gitignore with new patterns"
}

# ==========================================
# Phase 8: 統計とサマリー
# ==========================================
Write-Progress "Phase 8" "Generating completion summary"

$elapsed = (Get-Date) - $startTime
$summary = @"

🎉 Project Reorganization Complete!

Time Elapsed: $($elapsed.TotalMinutes.ToString("F1")) minutes
Mode: $(if ($DryRun) { "DRY RUN (no changes made)" } else { "EXECUTION" })

Organized Directories:
- ✓ backup/ - Deprecated and backup files
- ✓ test_coverage/ - Coverage reports and analysis
- ✓ tools/debug/ - Debug utilities
- ✓ docs/architecture/ - System documentation
- ✓ logs/archive/ - Old log files

Next Steps:
1. Review moved files in backup/ directories
2. Run tests to ensure system integrity: pytest -q
3. Check git status: git status
4. Commit changes: git add -A && git commit -m "Project reorganization"
5. Create new tag: git tag -a "v1.1-organized" -m "Post-cleanup organized state"

$(if ($DryRun) { "Re-run this script without -DryRun flag to apply changes." } else { "" })
"@

Write-Host $summary -ForegroundColor Green

# ==========================================
# Phase 9: 最終検証（DryRunでない場合）
# ==========================================
if (-not $DryRun) {
    Write-Progress "Phase 9" "Final verification"

    # 重要なファイルが残っているか確認
    $criticalFiles = @(
        "core/final_allocation.py",
        "scripts/run_all_systems_today.py",
        "common/cache_manager.py",
        "config/settings.py",
        "requirements.txt"
    )

    $allGood = $true
    foreach ($file in $criticalFiles) {
        if (-not (Test-Path $file)) {
            Write-Warning "Critical file missing: $file"
            $allGood = $false
        }
    }

    if ($allGood) {
        Write-Host "`n✓ All critical files verified" -ForegroundColor Green

        # Git status display
        Write-Host "`n=== Git Status ===" -ForegroundColor Magenta
        git status --short

        Write-Host "`nReorganization completed successfully! 🚀" -ForegroundColor Green
    } else {
        Write-Error "Some critical files are missing. Please review the changes."
    }
}

Write-Host "`nScript completed at $(Get-Date)" -ForegroundColor Gray
