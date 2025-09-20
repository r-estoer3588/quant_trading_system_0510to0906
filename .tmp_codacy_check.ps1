Set-Content -Path 'c:\Repos\quant_trading_system\codacy_analyze.log' -Value 'CLI on Windows is not supported without WSL.' -Encoding UTF8
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine.ToLower() -match 'codacy' -or $_.Name -match 'codacy') } | Select-Object ProcessId,@{Name='Cmd';Expression={$_.CommandLine}}
if ($procs) { $procs | Format-Table -AutoSize } else { Write-Output 'NO_CODACY_PROCS' }
Write-Output '---codacy_report files---'
$files = Get-ChildItem -Path 'c:\Repos\quant_trading_system\codacy_report' -File -ErrorAction SilentlyContinue
if ($files) { $files | Select-Object @{Name='Name';Expression={$_.Name}},@{Name='Size';Expression={$_.Length}} | Format-Table -AutoSize; $total = ($files | Measure-Object -Property Length -Sum).Sum; Write-Output "TOTAL_BYTES=$total" } else { Write-Output 'codacy_report not found or empty' }
