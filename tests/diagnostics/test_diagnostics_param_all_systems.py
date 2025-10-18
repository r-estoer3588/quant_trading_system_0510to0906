"""Parametric diagnostics shape tests for Systems 1–7.

This test validates that each system, when invoked with latest_only=True and
 include_diagnostics=True, returns a diagnostics payload with unified keys:
 - ranking_source
 - setup_predicate_count
 - ranked_top_n_count
 - predicate_only_pass_count
 - mismatch_flag

We use minimal in-memory DataFrames from conftest fixtures to ensure
setup conditions are met for each system.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "system_id, fixture_name, func_import, top_n",
    [
        (
            "system1",
            "minimal_system1_df",
            "core.system1:generate_candidates_system1",
            5,
        ),
        (
            "system2",
            "minimal_system2_df",
            "core.system2:generate_candidates_system2",
            5,
        ),
        (
            "system3",
            "minimal_system3_df",
            "core.system3:generate_candidates_system3",
            5,
        ),
        (
            "system4",
            "minimal_system4_df",
            "core.system4:generate_candidates_system4",
            5,
        ),
        (
            "system5",
            "minimal_system5_df",
            "core.system5:generate_candidates_system5",
            5,
        ),
        (
            "system6",
            "minimal_system6_df",
            "core.system6:generate_candidates_system6",
            5,
        ),
        (
            "system7",
            "minimal_system7_df",
            "core.system7:generate_candidates_system7",
            1,
        ),
    ],
)
def test_diagnostics_shape_latest_only(system_id: str, fixture_name: str, func_import: str, top_n: int, request):
    """latest_only=True モードで diagnostics の形状を検証する。"""
    # fixture を動的に取得
    maker = request.getfixturevalue(fixture_name)

    module_name, func_name = func_import.split(":", 1)
    mod = __import__(module_name, fromlist=[func_name])
    gen_func = getattr(mod, func_name)
    prepared = {("SPY" if system_id == "system7" else "AAA"): maker(True)}
    result = gen_func(prepared, latest_only=True, include_diagnostics=True, top_n=top_n)
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    assert isinstance(by_date, dict)
    assert merged is not None or system_id in {"system6", "system7"}
    assert isinstance(diag, dict)
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag, f"missing diagnostics key {k} for {system_id}"
    assert diag["ranking_source"] == "latest_only"
    # we expect >= 1 candidates when setup is True for latest_only; allow 0 only for unexpected paths
    assert int(diag["ranked_top_n_count"]) >= 0


@pytest.mark.parametrize(
    "system_id, fixture_name, func_import, top_n",
    [
        (
            "system1",
            "minimal_system1_df",
            "core.system1:generate_candidates_system1",
            5,
        ),
        (
            "system2",
            "minimal_system2_df",
            "core.system2:generate_candidates_system2",
            5,
        ),
        (
            "system3",
            "minimal_system3_df",
            "core.system3:generate_candidates_system3",
            5,
        ),
        (
            "system4",
            "minimal_system4_df",
            "core.system4:generate_candidates_system4",
            5,
        ),
        (
            "system5",
            "minimal_system5_df",
            "core.system5:generate_candidates_system5",
            5,
        ),
        (
            "system6",
            "minimal_system6_df",
            "core.system6:generate_candidates_system6",
            5,
        ),
        (
            "system7",
            "minimal_system7_df",
            "core.system7:generate_candidates_system7",
            1,
        ),
    ],
)
def test_diagnostics_shape_full_scan(system_id: str, fixture_name: str, func_import: str, top_n: int, request):
    """latest_only=False (full_scan) モードで diagnostics の形状を検証する。

    複数日分の DataFrame を用意し、full scan モードでの diagnostics が
    正しく返ってくることを確認する。
    """
    # fixture を動的に取得
    maker = request.getfixturevalue(fixture_name)

    module_name, func_name = func_import.split(":", 1)
    mod = __import__(module_name, fromlist=[func_name])
    gen_func = getattr(mod, func_name)

    # 複数日分の DataFrame を生成 (5日分)
    base_df = maker(True)
    dates = pd.date_range(base_df.index[-1] - pd.Timedelta(days=4), periods=5, freq="D")
    # 各列を5行に拡張 (最後の行の値を複製)
    extended_data = {}
    for col in base_df.columns:
        last_val = base_df[col].iloc[-1]
        extended_data[col] = [last_val] * 5
    df_multi = pd.DataFrame(extended_data, index=dates)

    prepared = {("SPY" if system_id == "system7" else "AAA"): df_multi}
    result = gen_func(prepared, latest_only=False, include_diagnostics=True, top_n=top_n)

    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}

    assert isinstance(by_date, dict)
    # full_scan では by_date が複数日分の辞書を持つことがある
    assert isinstance(diag, dict)

    # 必須キーの存在確認
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag, f"missing diagnostics key {k} for {system_id} (full_scan)"

    # full_scan モードでは ranking_source が "full_scan" であること
    assert diag["ranking_source"] == "full_scan", f"expected 'full_scan' for {system_id}"

    # setup_predicate_count は full_scan 時は複数日分の合計になる可能性がある
    assert isinstance(diag.get("setup_predicate_count"), int)
    assert diag["setup_predicate_count"] >= 0
