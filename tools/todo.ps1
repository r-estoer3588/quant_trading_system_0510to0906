<#!
.SYNOPSIS
  Lightweight repo-scoped todo CLI for PowerShell.
.DESCRIPTION
  Commands:
    todo list                # show todos
    todo add "text"         # add new todo
    todo done <id>           # mark as completed
    todo remove <id>         # delete a todo
    todo clear               # delete all todos (with prompt)
    todo help                # show help

  Data is stored in ./.todo/todos.json relative to repo root.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('list', 'add', 'done', 'remove', 'clear', 'help')]
    [string]$Command = 'list',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

function Get-StoragePath {
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot ".."))
    $dir = Join-Path $repoRoot ".todo"
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    return (Join-Path $dir "todos.json")
}

function Get-Todos {
    $path = Get-StoragePath
    if (-not (Test-Path -LiteralPath $path)) { return @() }
    try {
        $json = Get-Content -LiteralPath $path -Raw -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($json)) { return @() }
        $obj = ($json | ConvertFrom-Json -ErrorAction Stop)
        # Ensure array semantics
        if ($null -eq $obj) { return @() }
        if ($obj -is [System.Array]) { return $obj }
        return @($obj)
    }
    catch {
        Write-Warning "Failed to read todos.json; starting fresh. $_"
        return @()
    }
}

function Save-Todos($items) {
    $path = Get-StoragePath
    $json = ConvertTo-Json -InputObject $items -Depth 5
    Set-Content -LiteralPath $path -Value $json -Encoding UTF8
}

function Get-NextId($items) {
    if (-not $items -or $items.Count -eq 0) { return 1 }
    return ([int]($items | ForEach-Object { $_.id } | Measure-Object -Maximum).Maximum + 1)
}

function Show-List($items) {
    if (-not $items -or $items.Count -eq 0) { Write-Host "No todos."; return }
    $items | Sort-Object status, id | ForEach-Object {
        $status = if ($_.status -eq 'done') { '[x]' } else { '[ ]' }
        Write-Host ("{0} {1,3}: {2}" -f $status, $_.id, $_.text)
    }
}

$items = @(Get-Todos)

switch ($Command) {
    'help' { Get-Help -Detailed -ErrorAction SilentlyContinue; return }
    'list' { Show-List $items; return }
    'add' {
        $text = ($Rest -join ' ').Trim()
        if ([string]::IsNullOrWhiteSpace($text)) { Write-Error "Usage: todo add \"text\""; exit 1 }
        $new = [pscustomobject]@{ id = (Get-NextId $items); text = $text; status = 'todo'; created = (Get-Date).ToString('s') }
        $items += $new
        Save-Todos $items
        Write-Host "Added #$($new.id): $($new.text)"
        return
    }
    'done' {
        if (-not $Rest -or -not ($Rest[0] -as [int])) { Write-Error "Usage: todo done <id>"; exit 1 }
        $id = [int]$Rest[0]
        $item = $items | Where-Object { $_.id -eq $id } | Select-Object -First 1
        if (-not $item) { Write-Error "No such id: $id"; exit 1 }
        $item.status = 'done'
        $completedAt = (Get-Date).ToString('s')
        if ($item.PSObject.Properties.Name -contains 'completed') {
            $item.completed = $completedAt
        }
        else {
            $item | Add-Member -NotePropertyName completed -NotePropertyValue $completedAt -Force
        }
        Save-Todos $items
        Write-Host "Done #$id"
        return
    }
    'remove' {
        if (-not $Rest -or -not ($Rest[0] -as [int])) { Write-Error "Usage: todo remove <id>"; exit 1 }
        $id = [int]$Rest[0]
        $remaining = @()
        $removed = $false
        foreach ($it in $items) { if ($it.id -ne $id) { $remaining += $it } else { $removed = $true } }
        if (-not $removed) { Write-Error "No such id: $id"; exit 1 }
        Save-Todos $remaining
        Write-Host "Removed #$id"
        return
    }
    'clear' {
        $confirm = Read-Host "Clear ALL todos? (y/N)"
        if ($confirm -ne 'y' -and $confirm -ne 'Y') { Write-Host 'Canceled.'; return }
        Save-Todos @()
        Write-Host 'Cleared.'
        return
    }
}
