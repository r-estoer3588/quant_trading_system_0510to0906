"""``aggregate_round_trip_rows`` の regression test (表示用の集約)。

守りたい不変条件:

1. **1 ポジション = 1 行** — 同一注文 (同じ entry_order_id + exit_order_id) の
   部分約定でバラけた fragment が 1 行に畳まれること。株数は合計、価格は
   数量加重平均、実現損益は fragment の和。
2. **会計は不変** — 集約前後で実現損益の総和が 1 円も動かないこと。
3. **誤結合しない** — 決済を 2 回に分けた (exit_order_id が違う) round-trip や、
   order_id が欠けた古い行は畳まれず別行のまま残ること。
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.exit_ledger import aggregate_round_trip_rows  # noqa: E402


def _row(**kw):
    base = {
        "symbol": "DAIC",
        "side": "long",
        "qty": 10.0,
        "system": "system3",
        "entry_time": "2026-09-02T16:08:51.000000Z",
        "entry_session": "2026-09-02",
        "entry_price": 3.22,
        "exit_time": "2026-09-03T13:58:42.000000Z",
        "exit_session": "2026-09-03",
        "exit_price": 3.35,
        "holding_days": 1,
        "realized_pl": 1.30,
        "realized_pl_pct": 4.0,
        "exit_reason": "protect_oco",
        "exit_order_id": "X1",
        "entry_order_id": "E1",
        "system_source": "entry_order",
        "system_unknown_reason": None,
        "symbol_aliases": [],
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# 1. 1 ポジション = 1 行
# ---------------------------------------------------------------------------


def test_daic_5_fragments_collapse_to_one_position():
    """実データ (exit_ledger_20260904, DAIC-S3) の 5 分割を再現。"""
    frags = [
        _row(qty=47.0, entry_price=3.23, realized_pl=5.64),
        _row(qty=102.0, entry_price=3.22, realized_pl=13.26),
        _row(qty=29.0, entry_price=3.22, realized_pl=3.77),
        _row(qty=25.0, entry_price=3.22, realized_pl=3.25),
        _row(qty=31.0, entry_price=3.22, realized_pl=4.03),
    ]
    out = aggregate_round_trip_rows(frags)

    assert len(out) == 1
    pos = out[0]
    assert pos["qty"] == 234.0
    assert pos["n_fills"] == 5
    # realized_pl = fragment の和
    assert pos["realized_pl"] == round(5.64 + 13.26 + 3.77 + 3.25 + 4.03, 2)
    assert pos["realized_pl"] == 29.95
    # 数量加重平均 entry: (3.23*47 + 3.22*187) / 234  (payload は 4 桁丸め)
    assert abs(pos["entry_price"] - (3.23 * 47 + 3.22 * 187) / 234) < 5e-4
    assert pos["exit_price"] == 3.35  # 全 fragment 同値
    # 内訳は保持
    assert [f["qty"] for f in pos["fills"]] == [47.0, 102.0, 29.0, 25.0, 31.0]
    assert sum(f["realized_pl"] for f in pos["fills"]) == pos["realized_pl"]


def test_weighted_prices_and_pct_recomputed_for_short():
    frags = [
        _row(
            symbol="GTLB",
            side="short",
            exit_order_id="G",
            entry_order_id="GE",
            qty=22.0,
            entry_price=52.24,
            exit_price=49.58,
            realized_pl=58.52,
        ),
        _row(
            symbol="GTLB",
            side="short",
            exit_order_id="G",
            entry_order_id="GE",
            qty=11.0,
            entry_price=52.24,
            exit_price=49.56,
            realized_pl=29.48,
        ),
    ]
    pos = aggregate_round_trip_rows(frags)[0]
    assert pos["qty"] == 33.0
    assert pos["side"] == "short"
    assert abs(pos["exit_price"] - (49.58 * 22 + 49.56 * 11) / 33) < 5e-4
    # pct = realized / (avg_entry * qty) * 100  (符号は realized_pl が持つ)
    expected_pct = pos["realized_pl"] / (pos["entry_price"] * pos["qty"]) * 100
    assert abs(pos["realized_pl_pct"] - round(expected_pct, 3)) < 1e-2


def test_single_fragment_row_is_passed_through_with_n_fills_1():
    out = aggregate_round_trip_rows([_row(qty=12.0, realized_pl=1.0)])
    assert len(out) == 1
    assert out[0]["n_fills"] == 1
    assert "fills" not in out[0]
    assert out[0]["qty"] == 12.0


# ---------------------------------------------------------------------------
# 2. 会計は不変
# ---------------------------------------------------------------------------


def test_total_realized_pl_is_unchanged_by_aggregation():
    frags = [
        _row(
            symbol="DAIC",
            exit_order_id="X1",
            entry_order_id="E1",
            qty=47,
            realized_pl=5.64,
        ),
        _row(
            symbol="DAIC",
            exit_order_id="X1",
            entry_order_id="E1",
            qty=102,
            realized_pl=13.26,
        ),
        _row(
            symbol="REAX",
            exit_order_id="R1",
            entry_order_id="RE",
            qty=4,
            realized_pl=2.92,
        ),
        _row(
            symbol="REAX",
            exit_order_id="R1",
            entry_order_id="RE",
            qty=16,
            realized_pl=11.68,
        ),
        _row(
            symbol="XP",
            exit_order_id="P1",
            entry_order_id="PE",
            qty=1,
            realized_pl=-0.5,
        ),
    ]
    before = round(sum(f["realized_pl"] for f in frags), 2)
    out = aggregate_round_trip_rows(frags)
    after = round(sum(r["realized_pl"] for r in out), 2)
    assert before == after
    assert len(out) == 3  # DAIC, REAX, XP


def test_row_order_is_preserved():
    frags = [
        _row(
            symbol="AAA", exit_order_id="a", entry_order_id="ae", qty=1, realized_pl=1
        ),
        _row(
            symbol="BBB", exit_order_id="b", entry_order_id="be", qty=1, realized_pl=1
        ),
        _row(
            symbol="AAA", exit_order_id="a", entry_order_id="ae", qty=1, realized_pl=1
        ),
    ]
    out = aggregate_round_trip_rows(frags)
    assert [r["symbol"] for r in out] == ["AAA", "BBB"]
    assert out[0]["n_fills"] == 2


# ---------------------------------------------------------------------------
# 3. 誤結合しない
# ---------------------------------------------------------------------------


def test_two_exit_events_of_one_symbol_stay_separate():
    """REAX-S3 を 09-03 に一部、09-04 に残りを決済 -> exit_order_id が違う -> 別行。"""
    frags = [
        _row(
            symbol="REAX",
            entry_order_id="RE",
            exit_order_id="R_0903",
            qty=4,
            realized_pl=2.92,
            exit_session="2026-09-03",
        ),
        _row(
            symbol="REAX",
            entry_order_id="RE",
            exit_order_id="R_0903",
            qty=16,
            realized_pl=11.68,
            exit_session="2026-09-03",
        ),
        _row(
            symbol="REAX",
            entry_order_id="RE",
            exit_order_id="R_0904",
            qty=3,
            realized_pl=-0.24,
            exit_session="2026-09-04",
        ),
        _row(
            symbol="REAX",
            entry_order_id="RE",
            exit_order_id="R_0904",
            qty=8,
            realized_pl=-0.64,
            exit_session="2026-09-04",
        ),
    ]
    out = aggregate_round_trip_rows(frags)
    assert len(out) == 2
    assert {r["qty"] for r in out} == {20.0, 11.0}


def test_rows_missing_order_ids_are_not_merged():
    frags = [
        _row(
            symbol="OLD",
            entry_order_id=None,
            exit_order_id=None,
            qty=5,
            realized_pl=1.0,
        ),
        _row(
            symbol="OLD",
            entry_order_id=None,
            exit_order_id=None,
            qty=5,
            realized_pl=1.0,
        ),
    ]
    out = aggregate_round_trip_rows(frags)
    assert len(out) == 2
    assert all(r["n_fills"] == 1 for r in out)


def test_different_entry_orders_same_exit_are_not_merged():
    frags = [
        _row(
            symbol="ZZ", entry_order_id="E1", exit_order_id="X", qty=5, realized_pl=1.0
        ),
        _row(
            symbol="ZZ", entry_order_id="E2", exit_order_id="X", qty=5, realized_pl=1.0
        ),
    ]
    out = aggregate_round_trip_rows(frags)
    assert len(out) == 2


def test_empty_input():
    assert aggregate_round_trip_rows([]) == []
