"""半自動のピクセル差分検証ループ。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def run_tests(python_exe: str) -> bool:
    result = run_command([python_exe, "-m", "pytest", "-q"], check=False)
    if result.returncode != 0:
        print("[error] テスト失敗")
        print(result.stdout)
        print(result.stderr)
        return False
    print("[ok] テスト成功")
    return True


def capture_snapshot(python_exe: str, sources: List[str]) -> None:
    cmd = [python_exe, "tools/snapshot.py"]
    for item in sources:
        cmd.extend(["--source", item])
    run_command(cmd)


def check_image_diff(python_exe: str, src_dir: str) -> tuple[bool, str]:
    result = run_command(
        [python_exe, "tools/imgdiff.py", "--src-dir", src_dir],
        check=False,
    )
    output = result.stdout + result.stderr
    if "mismatches=0" in output:
        return False, "no diff"
    for line in output.splitlines():
        if "HTML report:" in line:
            return True, line.split(":", 1)[1].strip()
    return True, "diff found"


def build_prompt(report_path: str, iteration: int, max_iterations: int) -> str:
    return f"""
## Pixel diff review (iteration {iteration}/{max_iterations})

Please inspect the diff report below and propose changes that restore pixel parity.

- Diff report: `{report_path}`

### Provide the following
1. Root cause of the diff
2. Files to update and the plan
3. Updated code snippets (context + change)

### Constraints
- Keep existing tests passing
- Follow the validation loop described in docs/README.md
- Respect repository guidelines in .github/copilot-instructions.md
"""


def show_prompt(prompt: str) -> None:
    border = "=" * 80
    print(f"\n{border}\nPaste the following prompt into Copilot Chat\n{border}\n")
    print(prompt.strip())
    print(f"\n{border}\n")


def wait_user_choice() -> str:
    print("Review the AI proposal.")
    print(" a: apply changes and continue")
    print(" s: skip this iteration")
    print(" q: stop the loop")
    while True:
        choice = input("選択 [a/s/q]: ").strip().lower()
        if choice in {"a", "s", "q"}:
            return choice
        print("入力が正しくありません。")


def apply_fixes() -> bool:
    print("Apply the changes, save files, and press Enter when ready.")
    input()
    diff = run_command(["git", "diff", "--name-only"], check=False)
    changed = [line for line in diff.stdout.splitlines() if line.strip()]
    if changed:
        print(f"Modified files: {len(changed)}")
        for name in changed[:5]:
            print(f" - {name}")
        if len(changed) > 5:
            print(f" - ... {len(changed) - 5} more")
        return True
    print("No changes detected. Continue anyway? [y/N]")
    answer = input().strip().lower()
    return answer == "y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semi-automatic pixel diff loop")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum loop count",
    )
    parser.add_argument(
        "--snapshot-source",
        action="append",
        help="Directory to include in snapshots (repeatable)",
    )
    parser.add_argument(
        "--src-dir",
        default="results_images",
        help="Relative directory that contains generated images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = args.python
    sources = args.snapshot_source or [
        "results_csv",
        "results_csv_test",
        "logs",
        "results_images",
    ]
    max_iterations = max(1, args.max_iterations)

    print("[info] Starting semi-automatic pixel diff loop.")
    project_root = Path(__file__).resolve().parents[1]
    print(f"作業ディレクトリ: {project_root}")

    for iteration in range(1, max_iterations + 1):
        print("=" * 80)
        print(f"Iteration {iteration}/{max_iterations}")
        print("=" * 80)

        if not run_tests(python_exe):
            print("Tests failed. Stopping loop.")
            sys.exit(1)

        capture_snapshot(python_exe, sources)
        has_diff, info = check_image_diff(python_exe, args.src_dir)
        if not has_diff:
            print("[ok] No differences detected. Loop finished.")
            sys.exit(0)

        prompt = build_prompt(info, iteration, max_iterations)
        show_prompt(prompt)

        choice = wait_user_choice()
        if choice == "q":
            print("Loop stopped by user.")
            sys.exit(1)
        if choice == "s":
            print("Skipping this iteration.")
            continue
        if not apply_fixes():
            print("No changes applied. Skipping.")
            continue
        time.sleep(0.5)

    print("Maximum iteration count reached. Manual follow-up required.")
    sys.exit(1)


if __name__ == "__main__":
    main()
