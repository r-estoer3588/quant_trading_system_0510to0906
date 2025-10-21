import os
import subprocess
import sys
import time
from pathlib import Path


def test_cli_writes_to_results_subdir(tmp_path: Path) -> None:
    """Run the CLI in mini mode with RESULTS_DIR override and assert
    outputs are written under results_dir/run_<namespace>/ without error.
    """
    results_dir = tmp_path / "results"
    ns = f"pytest_cli_{int(time.time())}"

    env = os.environ.copy()
    env.update(
        {
            "RESULTS_DIR": str(results_dir),
            "PIPELINE_USE_RUN_SUBDIR": "1",
            "PIPELINE_USE_RUN_LOCK": "1",
            "RUN_NAMESPACE": ns,
            "FILTER_DEBUG": "1",
            "EXPORT_DIAGNOSTICS_SNAPSHOT_ALWAYS": "1",
        }
    )

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_all_systems_today.py"
    assert script.exists(), f"Expected script at {script}"

    cmd = [
        sys.executable,
        str(script),
        "--save-csv",
        "--skip-external",
        "--test-mode",
        "mini",
        "--run-namespace",
        ns,
    ]

    # Run CLI (may be a few seconds in mini mode)
    # Explicitly set text encoding and error handling so that decoding
    # of subprocess output is stable across CI/OS environments (Windows
    # CP932 locale can raise UnicodeDecodeError). Use UTF-8 and replace
    # undecodable bytes to avoid flakes caused by locale encoding.
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "CLI failed (rc={}). stdout:\n{}".format(proc.returncode, proc.stdout)
        )

    run_dir = results_dir / f"run_{ns}"
    assert run_dir.exists(), f"Expected run dir {run_dir} to exist"

    csvs = list(run_dir.glob('signals_final_*.csv'))
    assert csvs, f"No signals_final CSV found under {run_dir}"
    assert any(p.stat().st_size > 0 for p in csvs), "signals_final CSVs seem empty"

    val_dir = run_dir / "validation"
    assert val_dir.exists(), f"Expected validation dir {val_dir}"
    val_files = list(val_dir.glob('validation_report_*.json'))
    assert val_files, f"No validation report JSON found under {val_dir}"
