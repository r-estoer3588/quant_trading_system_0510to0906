<!-- docs/operations/safe_exec.md -->

# 安全実行ラッパ（safe_exec）とコマンドポリシー

目的: 破壊操作だけを自動ブロックし、それ以外のコマンドは自動承認で通す。実行ログは `logs/command_audit.jsonl` に残します。

## 構成

- ポリシー: `tools/command_policy.json`
  - denyPatterns: 破壊的操作を正規表現で定義（rm -rf, del /s, Remove-Item -Recurse/-Force, format/diskpart/cipher /w, robocopy /MIR など）
  - defaultPolicy: allow（上記以外は許可）
- ラッパ: `tools/safe_exec.ps1`
  - 引数のコマンドラインを受け取り、deny にマッチすればブロック。そうでなければ実行。
  - すべての実行を JSON Lines で監査ログに記録。

## 使い方

PowerShell からラッパ経由でコマンドを実行します。

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/safe_exec.ps1 -- "venv\\Scripts\\python.exe scripts\\run_all_systems_today.py --parallel --save-csv"
```

deny に該当する例（ブロックされます）:

```powershell
pwsh -NoProfile -File tools/safe_exec.ps1 -- "Remove-Item -Recurse -Force .\\data_cache"
```

## VS Code タスクへの組み込み（任意）

`.vscode/tasks.json` の command を `tools/safe_exec.ps1` に変更し、args に元のコマンドを 1 つの文字列として渡します。

```jsonc
{
  "label": "Safe: Run All Systems Today",
  "type": "shell",
  "command": "pwsh",
  "args": [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    "tools/safe_exec.ps1",
    "--",
    "venv\\Scripts\\python.exe scripts\\run_all_systems_today.py --parallel --save-csv"
  ],
  "group": "build"
}
```

## ポリシーの拡張

`tools/command_policy.json` の `denyPatterns` に正規表現を追加してください。PowerShell の `-match` は .NET 正規表現（既定で大文字小文字を区別しない）です。

## 注意事項

- このラッパ自体は「破壊操作の自動抑止」だけを行います。ネットワークや資格情報の扱いは別途運用ルールに従ってください。
- 実運用では CI からの一括実行よりも、VS Code タスク経由の利用を推奨します。
