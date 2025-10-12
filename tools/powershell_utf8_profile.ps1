# Enforce UTF-8 output for PowerShell sessions to avoid mojibake in logs
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$env:NO_EMOJI = '1'  # optional: strip emoji from app logs
Write-Host "UTF-8 output configured. NO_EMOJI=$($env:NO_EMOJI)"
