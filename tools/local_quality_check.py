"""Local lightweight quality aggregator.

目的:
    Docker や Codacy CLI を使わずに、既存ローカル環境で素早く品質状況を一括把握する。

実行内容(存在するツールのみ):
    - ruff (lint) : 問題一覧収集
    - mypy (型)   : エラー収集
    - pytest (短縮) : -q で失敗数のみ把握 (環境で pytest 利用可能な場合)
    - bandit (任意) : セキュリティ簡易チェック (インストールされていれば)
    - radon (任意)  : 複雑度 (インストールされていれば)

出力:
    JSON を stdout (人が読む簡易サマリは stderr) 。CI パイプライン等でパース可能。

使用例:
    python tools/local_quality_check.py --paths apps common strategies

終了コード:
    0  : 重大な失敗なし (テスト失敗や型/lintersエラーが 0)
    1  : 何かしらのカテゴリで検出 (lint, type, test fail, bandit high, etc.)

制限:
    - 依存ツールが無い場合はスキップ (status="skipped")。
    - パフォーマンス重視で詳細ログは最初の N 件に抑制。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_LIMIT = 50  # 各ツールの最大エントリ数


@dataclass
class ToolResult:
    name: str
    status: str  # success | fail | skipped | error
    issues: int = 0
    details: List[str] | None = None
    extra: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # None を落とす
        return {k: v for k, v in d.items() if v is not None}


def run_cmd(cmd: List[str], timeout: int = 600) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(cmd)}"


def detect_paths(user_paths: List[str] | None) -> List[str]:
    if user_paths:
        return user_paths
    # プロジェクト主要ディレクトリ推測
    candidates = [
        "apps",
        "common",
        "core",
        "strategies",
        "scripts",
        "utils",
        "tests",
    ]
    return [p for p in candidates if Path(p).exists()]


def run_ruff(paths: List[str], limit: int) -> ToolResult:
    if not shutil.which("ruff"):
        return ToolResult(name="ruff", status="skipped")
    cmd = ["ruff", "check", *paths, "--format", "json"]
    code, out, err = run_cmd(cmd)
    if code in (0, 1):  # 1 = 何らかの違反
        try:
            data = json.loads(out or "[]")
        except json.JSONDecodeError:
            return ToolResult(
                name="ruff",
                status="error",
                issues=0,
                details=["JSON parse error"],
                extra={"raw": out[:500]},
            )
        issues = len(data)
        sample = []
        for item in data[:limit]:
            sample.append(
                f"{item.get('filename')}:{item.get('location', {}).get('row')}:{item.get('code')} {item.get('message')}"
            )
        status = "fail" if issues > 0 else "success"
        return ToolResult(name="ruff", status=status, issues=issues, details=sample)
    if code == 127:
        return ToolResult(name="ruff", status="skipped")
    return ToolResult(name="ruff", status="error", details=[err[:500]])


def run_mypy(paths: List[str], limit: int) -> ToolResult:
    if not shutil.which("mypy"):
        return ToolResult(name="mypy", status="skipped")
    cmd = ["mypy", "--hide-error-context", "--no-color-output", *paths]
    code, out, err = run_cmd(cmd)
    text = out + err
    lines = [l for l in text.splitlines() if l.strip()]
    # 最終行に summary があるケース: "Found X errors" 等
    error_lines = [l for l in lines if ": error:" in l or l.endswith(" error")]
    issues = len(error_lines)
    if code == 0:
        return ToolResult(name="mypy", status="success", issues=0)
    return ToolResult(name="mypy", status="fail", issues=issues, details=error_lines[:limit])


def run_pytest(limit: int) -> ToolResult:
    if not shutil.which("pytest"):
        return ToolResult(name="pytest", status="skipped")
    cmd = ["pytest", "-q", "--maxfail=1"]
    code, out, err = run_cmd(cmd, timeout=1800)
    text = (out + err).strip()
    # pytest は失敗で非 0 だが、短い summary 行を抽出
    summary = None
    for l in reversed(text.splitlines()):
        if " failed" in l or " passed" in l:
            summary = l
            break
    if code == 0:
        return ToolResult(
            name="pytest", status="success", issues=0, details=[summary] if summary else None
        )

    # 失敗: 1件目の失敗部分の冒頭抜粋
    lines = text.splitlines()
    snippet = lines[-limit:]
    return ToolResult(
        name="pytest", status="fail", issues=1, details=snippet, extra={"summary": summary}
    )


def run_bandit(paths: List[str], limit: int) -> ToolResult:
    if not shutil.which("bandit"):
        return ToolResult(name="bandit", status="skipped")
    cmd = ["bandit", "-q", "-f", "json", "-r", *paths]
    code, out, err = run_cmd(cmd)
    try:
        data = json.loads(out or "{}")
    except json.JSONDecodeError:
        return ToolResult(
            name="bandit", status="error", details=["JSON parse error"], extra={"raw": out[:500]}
        )
    results = data.get("results", [])
    issues = len(results)
    sample = []
    for r in results[:limit]:
        sample.append(
            f"{r.get('filename')}:{r.get('line_number')} {r.get('issue_severity')} {r.get('issue_text')[:120]}"
        )
    status = "fail" if issues > 0 else "success"
    return ToolResult(name="bandit", status=status, issues=issues, details=sample)


def run_radon(paths: List[str], limit: int) -> ToolResult:
    if not shutil.which("radon"):
        return ToolResult(name="radon", status="skipped")
    cmd = ["radon", "cc", "-s", "-n", "C", *paths]
    code, out, err = run_cmd(cmd)
    text = out.strip()
    if code == 0 and not text:
        return ToolResult(name="radon", status="success", issues=0)
    lines = text.splitlines()
    issues = len(lines)
    return ToolResult(
        name="radon", status="fail" if issues else "success", issues=issues, details=lines[:limit]
    )


def aggregate(paths: List[str], limit: int) -> Dict[str, Any]:
    results = []
    results.append(run_ruff(paths, limit))
    results.append(run_mypy(paths, limit))
    results.append(run_pytest(limit))
    results.append(run_bandit(paths, limit))
    results.append(run_radon(paths, limit))

    overall_fail = any(r.status == "fail" or r.status == "error" for r in results)
    payload = {
        "overall_status": "fail" if overall_fail else "success",
        "results": [r.to_dict() for r in results],
        "paths": paths,
        "limit": limit,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Local quality aggregator (no Docker)")
    parser.add_argument("--paths", nargs="*", help="Target directories (default: auto-detect)")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="Per tool issue sample limit"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    paths = detect_paths(args.paths)
    if not paths:
        print(json.dumps({"error": "No target paths found"}), file=sys.stderr)
        return 1

    payload = aggregate(paths, args.limit)

    # stderr に簡易サマリ
    summary_lines = [
        f"[{p['name']}] {p['status']} issues={p.get('issues',0)}" for p in payload["results"]
    ]
    print("\n".join(summary_lines), file=sys.stderr)

    if args.pretty:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False))

    return 1 if payload["overall_status"] == "fail" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
