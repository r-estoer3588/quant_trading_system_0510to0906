"""Compare per-system persisted candidates, diagnostics snapshot, and final CSV.

Produces per-system lists of symbols that were present in the persisted per-system
inputs but not in the final signals CSV, and prints the diagnostics exclude_reasons
summary for each system when available.

Usage: run from repository root in the project's venv:
  python scripts/show_excluded_vs_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_diagnostics(snapshot_dir: Path) -> dict:
    # pick latest snapshot json by name
    if not snapshot_dir.exists():
        return {}
    files = sorted(snapshot_dir.glob('diagnostics_snapshot_*.json'))
    if not files:
        return {}
    latest = files[-1]
    try:
        return json.loads(latest.read_text(encoding='utf8'))
    except Exception:
        return {}


def find_per_system_feathers(results_dir: Path) -> dict[str, Path]:
    res = {}
    if not results_dir.exists():
        return res
    for p in results_dir.glob('per_system_*.feather'):
        name = p.stem.replace('per_system_', '')
        res[name] = p
    return res


def load_final_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding='utf-8')
        except Exception:
            return None


def main() -> int:
    snapshot_dir = ROOT / 'results_csv_test' / 'diagnostics_test'
    results_dir = ROOT / 'results_csv_test'
    today_name = pd.Timestamp.today().strftime('%Y-%m-%d')
    final_csv = ROOT / 'data_cache' / 'signals' / (
        f"signals_final_{today_name}.csv"
    )

    # load diagnostics
    diag = load_diagnostics(snapshot_dir)

    # find per-system feathers
    per_system = find_per_system_feathers(results_dir)

    # load final csv (fallback to latest file in directory if today-named missing)
    final_df = load_final_csv(final_csv)
    if final_df is None:
        # try to pick latest signals_final_*.csv
        sfiles = sorted((ROOT / 'data_cache' / 'signals').glob('signals_final_*.csv'))
        if sfiles:
            final_df = load_final_csv(sfiles[-1])
            final_csv = sfiles[-1]

    if final_df is None:
        print('ERROR: final signals CSV not found')
        return 2

    # normalize symbol column
    final_symbols_by_system: dict[str, set[str]] = {}
    if 'system' in final_df.columns and 'symbol' in final_df.columns:
        for _, r in final_df.iterrows():
            sysn = str(r['system'])
            sym = str(r['symbol'])
            final_symbols_by_system.setdefault(sysn, set()).add(sym)

    # For each per-system persisted file, show symbols present and which were dropped
    summary = {}
    for sysname, path in sorted(per_system.items()):
        try:
            df = pd.read_feather(path)
        except Exception as e:
            print(f'Could not read {path}: {e}')
            continue
        # attempt to find symbol column name
        if 'symbol' in df.columns:
            syms = set(df['symbol'].astype(str).tolist())
        else:
            # try lowercase
            possible = [c for c in df.columns if c.lower() == 'symbol']
            if possible:
                syms = set(df[possible[0]].astype(str).tolist())
            else:
                syms = set()

        final_syms = final_symbols_by_system.get(sysname, set())
        dropped = sorted(list(syms - final_syms))
        kept = sorted(list(syms & final_syms))
        extra = sorted(list(final_syms - syms))

        diag_summary = None
        if isinstance(diag, dict) and sysname in diag:
            diag_summary = diag.get(sysname, {})

        summary[sysname] = {
            'persisted_count': len(syms),
            'persisted_symbols': sorted(list(syms)),
            'kept_in_final': kept,
            'dropped_in_final': dropped,
            'extra_in_final': extra,
            'diagnostics': diag_summary,
        }

    # print concise report
    print(f'Final CSV used: {final_csv}')
    for sysname, info in sorted(summary.items()):
        print('\n' + '=' * 40)
        print(f'System: {sysname}')
        print(f"Persisted candidates: {info['persisted_count']}")
        print('Persisted symbols: ' + ', '.join(info['persisted_symbols']))
        print('Kept in final: ' + (', '.join(info['kept_in_final']) or '(none)'))
        dropped_txt = ', '.join(info['dropped_in_final']) or '(none)'
        print('Dropped before final: ' + dropped_txt)
        if info['extra_in_final']:
            extra_txt = ', '.join(info['extra_in_final'])
            print(
                'Present in final but not in persisted '
                '(likely from another source): ' + extra_txt
            )
        if info['diagnostics']:
            try:
                ex = (
                    info['diagnostics'].get('exclude_reasons')
                    if isinstance(info['diagnostics'], dict)
                    else None
                )
                if isinstance(ex, dict):
                    pairs = [f"{k}={v}" for k, v in ex.items()]
                    print('Diagnostics exclude_reasons: ' + ', '.join(pairs))
            except Exception:
                pass

    # also print systems that had final rows but no persisted file
    final_only_systems = set(final_symbols_by_system.keys()) - set(per_system.keys())
    if final_only_systems:
        systems_txt = ', '.join(sorted(final_only_systems))
        print(
            '\nSystems present in final CSV but with no persisted per_system file: '
            + systems_txt
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
