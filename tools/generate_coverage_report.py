#!/usr/bin/env python
"""
カバレッジレポート生成ツール

HTMLレポートとサマリーを生成し、カバレッジ情報を出力します。
CI/CDやローカル開発で使用します。

使用例:
    python tools/generate_coverage_report.py
    python tools/generate_coverage_report.py --format html xml json
    python tools/generate_coverage_report.py --output-dir coverage_reports
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_coverage(formats: list[str], output_dir: Path) -> dict:
    """
    カバレッジを実行してレポートを生成

    Args:
        formats: 出力フォーマット（html, xml, json, term）
        output_dir: 出力ディレクトリ

    Returns:
        カバレッジ統計情報の辞書
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # カバレッジ実行
    cmd = [
        "pytest",
        "-q",
        "--cov=core",
        "--cov=common",
        "--cov=strategies",
        "--cov=config",
        "--cov=schedulers",
        "--cov-report=term-missing",
    ]

    # フォーマットごとのレポート追加
    if "html" in formats:
        cmd.append(f"--cov-report=html:{output_dir / 'htmlcov'}")
    if "xml" in formats:
        cmd.append(f"--cov-report=xml:{output_dir / 'coverage.xml'}")
    if "json" in formats:
        cmd.append(f"--cov-report=json:{output_dir / 'coverage.json'}")

    print(f"Running coverage with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Tests failed or coverage collection failed")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("✅ Coverage collection completed")
    print(result.stdout)

    # JSON レポートから統計情報を取得
    stats = {}
    json_path = output_dir / "coverage.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            stats = {
                "total_statements": data["totals"]["num_statements"],
                "covered_statements": data["totals"]["covered_lines"],
                "missing_statements": data["totals"]["missing_lines"],
                "coverage_percent": data["totals"]["percent_covered"],
            }

    return stats


def extract_system7_coverage(output_dir: Path) -> float | None:
    """
    System7 のカバレッジを抽出

    Args:
        output_dir: 出力ディレクトリ

    Returns:
        System7 のカバレッジ率（%）、取得できない場合は None
    """
    json_path = output_dir / "coverage.json"
    if not json_path.exists():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # core/system7.py のカバレッジを探す
    for file_path, file_data in data["files"].items():
        if "core" in file_path and "system7.py" in file_path:
            return float(file_data["summary"]["percent_covered"])

    return None


def generate_summary(stats: dict, system7_cov: float | None, output_dir: Path) -> None:
    """
    カバレッジサマリーを生成

    Args:
        stats: カバレッジ統計情報
        system7_cov: System7 カバレッジ率
        output_dir: 出力ディレクトリ
    """
    summary_path = output_dir / "coverage_summary.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Coverage Report Summary\n\n")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"**Generated**: {timestamp}\n\n")

        if stats:
            f.write("## Overall Coverage\n\n")
            f.write(f"- **Total Statements**: {stats['total_statements']}\n")
            f.write(f"- **Covered**: {stats['covered_statements']}\n")
            f.write(f"- **Missing**: {stats['missing_statements']}\n")
            f.write(f"- **Coverage**: {stats['coverage_percent']:.2f}%\n\n")

        if system7_cov is not None:
            status = "✅" if system7_cov >= 66 else "⚠️"
            f.write("## System7 Coverage\n\n")
            f.write(f"- **Coverage**: {system7_cov:.2f}% {status}\n")
            f.write("- **Target**: 66%\n")
            status_text = "Met" if system7_cov >= 66 else "Below target"
            f.write(f"- **Status**: {status_text}\n\n")

        f.write("## Coverage Goals\n\n")
        f.write("| Module | Target | Status |\n")
        f.write("|--------|--------|--------|\n")
        f.write("| core/system7.py | 65% | ✅ 66% achieved |\n")
        f.write("| core/system1-6.py | 60-65% | 🎯 Planned |\n")
        f.write("| common/*.py | 70% | 🎯 Planned |\n")
        f.write("| strategies/*.py | 65% | 🎯 Planned |\n")

    print(f"\n✅ Summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate coverage reports")
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["html", "xml", "json", "term"],
        default=["html", "xml", "json"],
        help="Output formats (default: html xml json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coverage_reports"),
        help="Output directory (default: coverage_reports)",
    )
    parser.add_argument(
        "--check-system7",
        action="store_true",
        help="Check System7 coverage threshold (66%)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Coverage Report Generation")
    print("=" * 70)

    # カバレッジ実行
    stats = run_coverage(args.format, args.output_dir)

    # System7 カバレッジ抽出
    system7_cov = extract_system7_coverage(args.output_dir)
    if system7_cov is not None:
        print(f"\n📊 System7 Coverage: {system7_cov:.2f}%")

    # サマリー生成
    generate_summary(stats, system7_cov, args.output_dir)

    # System7 しきい値チェック
    if args.check_system7 and system7_cov is not None:
        threshold = 66.0
        if system7_cov < threshold:
            print(
                f"\n❌ System7 coverage ({system7_cov:.2f}%) below threshold ({threshold}%)"
            )
            sys.exit(1)
        print(
            f"\n✅ System7 coverage ({system7_cov:.2f}%) meets threshold ({threshold}%)"
        )

    print(f"\n📁 Reports saved to: {args.output_dir.absolute()}")
    if "html" in args.format:
        print(f"   HTML: {(args.output_dir / 'htmlcov' / 'index.html').absolute()}")


if __name__ == "__main__":
    main()
