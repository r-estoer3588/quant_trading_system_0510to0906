# 文字化け（mojibake）対策ガイド（Windows）

本プロジェクトは UTF-8 でログや CSV を出力します。Windows の既定（cp932）と混在すると文字化けが発生します。以下の手順で恒久対処します。

## 推奨設定（PowerShell）

1. プロファイルに UTF-8 を設定
   - `tools/powershell_utf8_profile.ps1` を実行
   - もしくは `Microsoft.PowerShell_profile.ps1` に以下を追記:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$env:NO_EMOJI = '1'  # 任意: 絵文字をログから除去
```

2. 一時的に適用（セッションごと）

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:NO_EMOJI = '1'
```

## アプリ側の対策

- scripts/run_all_systems_today.py は Windows で stdout/stderr を UTF-8 に再構成
- common/logging_utils.SystemLogger は `NO_EMOJI=1` で絵文字を除去
- ログ/CSV/JSON は UTF-8 で書き出し

## よくある症状と対処

- 表示だけが化ける: ターミナルのエンコーディングを UTF-8 に
- CSV が化ける: Excel ではなくエディタ（UTF-8）で開くか、Excel のインポートで UTF-8 指定
- 一部の記号だけが化ける: `NO_EMOJI=1` を併用

---

参考: docs/technical/environment_variables.md（NO_EMOJI, COMPACT_TODAY_LOGS など）
