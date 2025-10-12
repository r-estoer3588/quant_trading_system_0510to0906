"""
UI進捗分析マスタースクリプト

3つの分析ツールを順次実行し、統合レポートを生成:
1. スクショ自動解析（OCR + ピクセル解析）
2. JSONL同期検証
3. ビジュアル差分検出

Usage:
    python tools/run_complete_ui_analysis.py

出力:
    - screenshots/progress_tracking/analysis_results.json
    - screenshots/progress_tracking/sync_verification.json
    - screenshots/progress_tracking/visual_diff_report.json
    - screenshots/progress_tracking/COMPLETE_ANALYSIS_REPORT.md
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """コマンド実行ヘルパー"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} 完了\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} 失敗: {e}\n")
        return False


def generate_markdown_report(
    analysis_file: Path,
    sync_file: Path,
    diff_file: Path,
    output_file: Path,
) -> None:
    """統合Markdownレポートを生成"""

    # データ読み込み
    with open(analysis_file, encoding="utf-8") as f:
        analysis_data = json.load(f)

    with open(sync_file, encoding="utf-8") as f:
        sync_data = json.load(f)

    with open(diff_file, encoding="utf-8") as f:
        diff_data = json.load(f)

    # レポート生成
    report_lines = [
        "# UI進捗分析 完全レポート",
        "",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 📊 サマリー",
        "",
        "### スクショ解析",
        f"- 総スクショ数: {len(analysis_data)} 枚",
        f"- 進捗バー抽出成功: {sum(1 for r in analysis_data if r.get('progress_percentage') is not None)} 枚",
        "",
        "### UI/JSONL同期検証",
        f"- 検証済みスクショ: {sync_data['summary']['verified_screenshots']} 枚",
        f"- **問題検出: {sync_data['summary']['issues_found']} 件**",
        f"- **進捗後退: {sync_data['summary']['progress_regressions']} 件**",
        "",
        "### ビジュアル差分検出",
        f"- 比較ペア数: {diff_data['summary']['total_pairs']} ペア",
        f"- **進捗後退検出: {diff_data['summary']['progress_regressions']} 件**",
        f"- 大きな変化: {diff_data['summary']['major_changes']} 件",
        "",
    ]

    # 進捗後退の詳細
    if sync_data["progress_regressions"]:
        report_lines.extend(
            [
                "## 🔴 進捗後退の詳細（JSONL照合）",
                "",
            ]
        )

        for reg in sync_data["progress_regressions"]:
            report_lines.extend(
                [
                    f"### {reg['prev_file']} → {reg['curr_file']}",
                    f"- 前: {reg['prev_progress']}%",
                    f"- 後: {reg['curr_progress']}%",
                    f"- **後退量: {reg['regression_amount']}%**",
                    f"- タイムスタンプ: {reg['prev_timestamp']} → {reg['curr_timestamp']}",
                    "",
                ]
            )

    # ビジュアル差分による進捗後退
    if diff_data["regression_details"]:
        report_lines.extend(
            [
                "## 🔴 進捗後退の詳細（ビジュアル差分）",
                "",
            ]
        )

        for reg in diff_data["regression_details"]:
            pb_diff = reg.get("progress_bar_diff", {})
            report_lines.extend(
                [
                    f"### {reg['prev_file']} → {reg['curr_file']}",
                    f"- 緑ピクセル: {pb_diff.get('green_pixels_prev')} → {pb_diff.get('green_pixels_curr')}",
                    f"- 変化量: {pb_diff.get('green_pixel_change')}",
                    "",
                ]
            )

    # UI同期問題
    issues = [r for r in sync_data["verification_results"] if r["status"] == "issue"]
    if issues:
        report_lines.extend(
            [
                "## ⚠️ UI同期問題",
                "",
            ]
        )

        for issue in issues[:10]:  # 最初の10件のみ
            report_lines.extend(
                [
                    f"### {issue['file']}",
                    f"- タイムスタンプ: {issue['timestamp']}",
                    "- 問題:",
                ]
            )
            for problem in issue["issues"]:
                report_lines.append(f"  - {problem}")
            report_lines.append("")

    # 進捗バー推移グラフ（テキストベース）
    valid_progress = [
        (r.get("timestamp"), r.get("progress_percentage"))
        for r in analysis_data
        if r.get("progress_percentage") is not None
    ]

    if valid_progress:
        report_lines.extend(
            [
                "## 📈 進捗バー推移",
                "",
                "| タイムスタンプ | 進捗% |",
                "|--------------|-------|",
            ]
        )

        # サンプリング（20件程度）
        step = max(1, len(valid_progress) // 20)
        for ts, progress in valid_progress[::step]:
            report_lines.append(f"| {ts} | {progress}% |")

        report_lines.append("")

    # 結論
    total_issues = (
        sync_data["summary"]["issues_found"]
        + sync_data["summary"]["progress_regressions"]
        + diff_data["summary"]["progress_regressions"]
    )

    report_lines.extend(
        [
            "## 🎯 結論",
            "",
        ]
    )

    if total_issues == 0:
        report_lines.extend(
            [
                "✅ **UI進捗バーとJSONL記録は完全に同期しています。**",
                "",
                "問題は検出されませんでした。",
            ]
        )
    else:
        report_lines.extend(
            [
                f"🔴 **合計 {total_issues} 件の問題が検出されました。**",
                "",
                "### 推奨アクション",
                "",
                "1. `apps/app_today_signals.py` の `StageTracker` クラスを確認",
                "2. 進捗バー更新ロジック（`update_progress()`）を検証",
                "3. JSONL読み込みと UI 反映の同期タイミングを調整",
                "",
            ]
        )

    # ファイル出力
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📄 統合レポート生成: {output_file}\n")


def main():
    print("=" * 60)
    print("🚀 UI進捗分析 完全実行")
    print("=" * 60)
    print("")
    print("以下の3つの分析を順次実行します:")
    print("  1. スクショ自動解析（OCR + ピクセル解析）")
    print("  2. JSONL同期検証")
    print("  3. ビジュアル差分検出")
    print("")

    python_exe = sys.executable

    # 1. スクショ自動解析
    analysis_success = run_command(
        [python_exe, "tools/analyze_ui_screenshots.py"],
        "スクショ自動解析",
    )

    if not analysis_success:
        print("❌ スクショ解析が失敗したため、処理を中断します。")
        return 1

    # 2. JSONL同期検証
    sync_success = run_command(
        [python_exe, "tools/verify_ui_jsonl_sync.py"],
        "JSONL同期検証",
    )

    # 3. ビジュアル差分検出
    diff_success = run_command(
        [python_exe, "tools/detect_visual_diff.py"],
        "ビジュアル差分検出",
    )

    # 4. 統合レポート生成
    if analysis_success and sync_success and diff_success:
        print("\n" + "=" * 60)
        print("📊 統合レポート生成")
        print("=" * 60 + "\n")

        analysis_file = Path("screenshots/progress_tracking/analysis_results.json")
        sync_file = Path("screenshots/progress_tracking/sync_verification.json")
        diff_file = Path("screenshots/progress_tracking/visual_diff_report.json")
        output_file = Path("screenshots/progress_tracking/COMPLETE_ANALYSIS_REPORT.md")

        try:
            generate_markdown_report(analysis_file, sync_file, diff_file, output_file)

            print("\n" + "=" * 60)
            print("✅ 全分析完了！")
            print("=" * 60)
            print(f"\n📄 完全レポート: {output_file}")
            print("\nレポートを確認してください。")

            return 0

        except Exception as e:
            print(f"❌ レポート生成エラー: {e}")
            return 1
    else:
        print("\n❌ 一部の分析が失敗したため、統合レポートは生成されませんでした。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
