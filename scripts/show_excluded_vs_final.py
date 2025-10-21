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

    # normalize diagnostics mapping:
    # - older snapshots stored per-system at top-level dict
    # - newer snapshots use a 'systems' list with entries containing 'system_id'
    #   and 'diagnostics_extra'
    # Build a map: sysname -> diagnostics_extra
    diag_map: dict[str, dict] = {}
    try:
        if isinstance(diag, dict) and 'systems' in diag:
            if isinstance(diag['systems'], list):
                for s in diag['systems']:
                    sid = s.get('system_id')
                    if not sid:
                        continue
                    extra = s.get('diagnostics_extra') or s.get('diagnostics') or {}
                    diag_map[str(sid)] = dict(extra)
        elif isinstance(diag, dict):
            # older format: keys are system names
            for k, v in diag.items():
                if isinstance(v, dict):
                    diag_map[str(k)] = dict(v)
    except Exception:
        diag_map = {}

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

        diag_summary = diag_map.get(sysname) if isinstance(diag_map, dict) else None

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
                # prefer structured exclude_symbols if available (reason -> [symbols])
                d = info['diagnostics']
                ex_syms = None
                if isinstance(d, dict):
                    ex_syms = (
                        d.get('exclude_symbols')
                        or d.get('exclude_reasons_symbols')
                    )
                    # fallback to simple counts if no symbol lists present
                    ex_counts = d.get('exclude_reasons')
                else:
                    ex_syms = None
                    ex_counts = None

                if isinstance(ex_syms, dict) and ex_syms:
                    print('Diagnostics exclude_symbols:')
                    for reason, syms in ex_syms.items():
                        try:
                            symlist = sorted(map(str, syms))
                        except Exception:
                            if isinstance(syms, (list, set)):
                                symlist = list(syms)
                            else:
                                symlist = [str(syms)]
                        print(f'  - {reason}: ' + ', '.join(symlist))
                elif isinstance(ex_counts, dict) and ex_counts:
                    pairs = [f"{k}={v}" for k, v in ex_counts.items()]
                    print('Diagnostics exclude_reasons: ' + ', '.join(pairs))
            except Exception:
                pass

    # write machine-readable summary to results_csv_test
    try:
        out_dir = ROOT / 'results_csv_test'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'excluded_vs_final_{ts}.json'
        # summary contains only JSON-serializable types (lists/strings)
        with out_path.open('w', encoding='utf8') as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"\nWrote machine-readable report: {out_path}")
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
