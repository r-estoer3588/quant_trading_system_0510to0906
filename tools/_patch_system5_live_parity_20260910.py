from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8-sig")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly 1 match, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_exit_check() -> None:
    path = ROOT / "scripts" / "paper_exit_check.py"
    replace_once(
        path,
        "    fetch_position_snapshots,\n    parse_entry_date_from_client_order_id,",
        "    fetch_position_snapshots,\n    hydrate_system_tags,\n    parse_entry_date_from_client_order_id,",
    )
    replace_once(
        path,
        "from common.trade_management import SYSTEM_TRADE_RULES  # noqa: E402\n",
        "from common.trade_management import SYSTEM_TRADE_RULES  # noqa: E402\n"
        "from common.system5_live_exit import (  # noqa: E402\n"
        "    SYSTEM5,\n"
        "    SYSTEM5_TARGET_NEXT_OPEN,\n"
        "    build_system5_exit_orders,\n"
        "    load_history as load_system5_history,\n"
        ")\n",
    )

    old_build = '''    exits = build_exit_orders_from_positions(
        snapshots,
        today=date_str,
        unassigned_out=unassigned,
        protection_coverage_out=protection_coverage,
        tracker=tracker,
        symbol_map=symbol_map,
        symbol_aliases=rename_aliases,
        entry_orders_index=entry_orders_index,
        existing_protect_coids=existing_protect_coids,
        existing_exit_coids=existing_exit_coids,
        spy_high=spy_high,
        spy_max70=spy_max70,
        atr_by_symbol=atr_by_symbol,
        price_by_symbol=price_by_symbol,
    )
'''
    new_build = '''    # Resolve tags before splitting: System5 now has a dedicated live adapter whose
    # state machine mirrors strategies/system5_strategy.py exactly.  All other systems
    # stay on the existing generic builder unchanged.
    hydrate_system_tags(
        snapshots,
        tracker=tracker,
        entry_orders_index=entry_orders_index,
        symbol_map=symbol_map,
        symbol_aliases=rename_aliases,
    )
    generic_snapshots = [s for s in snapshots if str(s.system or "").lower() != SYSTEM5]
    system5_snapshots = [s for s in snapshots if str(s.system or "").lower() == SYSTEM5]

    exits = build_exit_orders_from_positions(
        generic_snapshots,
        today=date_str,
        unassigned_out=unassigned,
        protection_coverage_out=protection_coverage,
        tracker=tracker,
        symbol_map=symbol_map,
        symbol_aliases=rename_aliases,
        entry_orders_index=entry_orders_index,
        existing_protect_coids=existing_protect_coids,
        existing_exit_coids=existing_exit_coids,
        spy_high=spy_high,
        spy_max70=spy_max70,
        atr_by_symbol=atr_by_symbol,
        price_by_symbol=price_by_symbol,
    )

    # System5 contract (Bensdorp): target touch is a trigger, NOT an immediate
    # take-profit fill.  The target and stop both use ATR10 frozen from the bar before
    # entry; target -> next-session OPEN market close; timeout -> seventh-session OPEN.
    rolling_dir = ROOT / "data_cache" / "rolling"
    for snap in system5_snapshots:
        history = load_system5_history(rolling_dir, snap.symbol)
        exits.extend(
            build_system5_exit_orders(
                snap,
                today=date_str,
                history=history,
                existing_protect_coids=existing_protect_coids,
                existing_exit_coids=existing_exit_coids,
                coverage_out=protection_coverage,
            )
        )
'''
    replace_once(path, old_build, new_build)

    replace_once(
        path,
        "                if po.reason in (ExitReasonCode.TIME, ExitReasonCode.BREAKOUT)\n",
        "                if po.reason\n"
        "                in (\n"
        "                    ExitReasonCode.TIME,\n"
        "                    ExitReasonCode.BREAKOUT,\n"
        "                    SYSTEM5_TARGET_NEXT_OPEN,\n"
        "                )\n",
    )

    replace_once(
        path,
        "    breakout_cnt = sum(1 for e in exits if e.reason == \"spy_breakout\")\n"
        "    protect_cnt = sum(1 for e in exits if e.reason.startswith(\"protect_\"))\n",
        "    breakout_cnt = sum(1 for e in exits if e.reason == \"spy_breakout\")\n"
        "    target_next_open_cnt = sum(\n"
        "        1 for e in exits if e.reason == SYSTEM5_TARGET_NEXT_OPEN\n"
        "    )\n"
        "    protect_cnt = sum(1 for e in exits if e.reason.startswith(\"protect_\"))\n",
    )
    replace_once(
        path,
        "        f\"(time={time_cnt}, breakout={breakout_cnt}, protect={protect_cnt}) \"\n",
        "        f\"(time={time_cnt}, target_next_open={target_next_open_cnt}, \"\n"
        "        f\"breakout={breakout_cnt}, protect={protect_cnt}) \"\n",
    )


def patch_exporter_wrapper() -> None:
    path = ROOT / "scripts" / "export_alpaca_snapshot.py"
    replace_once(
        path,
        "from common.trade_management import SYSTEM_TRADE_RULES\n",
        "from common.alpaca_trading import PositionSnapshot, protective_stop_price\n"
        "from common.system5_live_exit import (\n"
        "    SYSTEM5,\n"
        "    SYSTEM5_TARGET_NEXT_OPEN,\n"
        "    evaluate_system5_state,\n"
        "    load_history as load_system5_history,\n"
        ")\n"
        "from common.trade_management import SYSTEM_TRADE_RULES\n",
    )

    marker = '''def build_snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Delegate all legacy measurement, then replace only protection truth."""
'''
    helper = '''def _apply_system5_truth(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Overlay only System5 timing/ATR fields with canonical strategy semantics."""
    today = str(snapshot.get("date") or "")[:10]
    rolling_dir = ROOT / "data_cache" / "rolling"
    rules = SYSTEM_TRADE_RULES[SYSTEM5]
    for position in snapshot.get("positions") or []:
        if str(position.get("system") or "").lower() != SYSTEM5:
            continue
        snap = PositionSnapshot(
            symbol=str(position.get("symbol") or "").upper(),
            qty=float(position.get("qty") or 0.0),
            side=str(position.get("side") or "long"),
            avg_entry_price=float(position.get("avg_entry_price") or 0.0),
            market_value=_f(position.get("market_value")),
            unrealized_pl=_f(position.get("unrealized_pl")),
            system=SYSTEM5,
            entry_date=str(position.get("entry_date") or "")[:10] or None,
        )
        history = load_system5_history(rolling_dir, snap.symbol)
        state = evaluate_system5_state(snap, today=today, history=history)

        if snap.entry_date:
            # Six full post-entry sessions are observations; the seventh open is the
            # timeout.  Therefore holding=6 means "あと1日", not "本日手仕舞い".
            position["days_remaining"] = (
                int(rules.max_holding_days) + 1 - state.holding_days
            )
            position["exit_date"] = state.timeout_exit_date

        position["target_price_est"] = (
            round(state.target_price, 4) if state.target_price is not None else None
        )
        position["system5_entry_atr10"] = state.entry_atr10
        position["system5_target_hit_date"] = state.target_hit_date

        stop = None
        if state.entry_atr10 is not None and snap.avg_entry_price > 0:
            stop = protective_stop_price(
                side=snap.side,
                avg_entry_price=snap.avg_entry_price,
                rules=rules,
                atr_value=state.entry_atr10,
                symbol=snap.symbol,
            )
        position["stop_price_est"] = round(stop, 4) if stop is not None else None
        current = _f(position.get("current_price"))
        position["distance_to_stop_pct"] = (
            round((stop - current) / current * 100.0, 3)
            if stop is not None and current is not None and current > 0
            else None
        )
        target = state.target_price
        position["distance_to_target_pct"] = (
            round((target - current) / current * 100.0, 3)
            if target is not None and current is not None and current > 0
            else None
        )

        if state.target_exit_due:
            position["exit_expected"] = SYSTEM5_TARGET_NEXT_OPEN
        elif state.timeout_exit_due:
            position["exit_expected"] = "time_based"
        else:
            position["exit_expected"] = None
            position["exit_execution_state"] = None
    return snapshot


''' + marker
    replace_once(path, marker, helper)

    replace_once(
        path,
        "        snapshot = _LEGACY_BUILD_SNAPSHOT(*args, **kwargs)\n",
        "        snapshot = _LEGACY_BUILD_SNAPSHOT(*args, **kwargs)\n"
        "        snapshot = _apply_system5_truth(snapshot)\n",
    )


def patch_dashboard() -> None:
    path = ROOT / "apps" / "dashboards" / "alpaca-next" / "components" / "AlpacaSectionLegacy.tsx"
    replace_once(
        path,
        "function exitBadge(p: AlpacaPosition): { text: string; cls: string; sub?: string } {\n"
        "  if (p.exit_expected === 'time_based') {",
        "function exitBadge(p: AlpacaPosition): { text: string; cls: string; sub?: string } {\n"
        "  if (p.exit_expected === 'system5_target_next_open') {\n"
        "    return {\n"
        "      text: '利益目標達成 · 次寄り手仕舞い',\n"
        "      cls: 'bg-ok/20 text-ok',\n"
        "      sub: 'S5: +1ATR到達 → 翌セッション寄付成行',\n"
        "    };\n"
        "  }\n"
        "  if (p.exit_expected === 'time_based') {",
    )


def main() -> None:
    patch_exit_check()
    patch_exporter_wrapper()
    patch_dashboard()
    print("System5 live parity integration patch applied")


if __name__ == "__main__":
    main()
