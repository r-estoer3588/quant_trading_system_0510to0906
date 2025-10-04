import json
from pathlib import Path

import pandas as pd

from common.perf_snapshot import enable_global_perf
from strategies.system6_strategy import System6Strategy


def test_system6_perf_snapshot_candidate_count(tmp_path, monkeypatch):
    # ダミー settings でログ先を tmp に
    class DummySettings:
        LOGS_DIR = str(tmp_path / "logs")

    def fake_get_settings(create_dirs=False):  # noqa: D401
        if create_dirs:
            (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        return DummySettings()

    monkeypatch.setattr("common.perf_snapshot.get_settings", fake_get_settings)

    # 最小データ: candidates 2 件を擬似生成させるため generate_candidates_system6 を monkeypatch
    # StrategyBase は設定読み込みを行うため、ここでは標準 __init__ を使用（引数不要）
    s = System6Strategy()

    dummy_candidates = {
        "AAA": {"entry_date": "2024-01-01"},
        "BBB": {"entry_date": "2024-01-01"},
    }
    dummy_df = pd.DataFrame({"Close": [1, 2, 3]})

    def fake_generate_candidates_system6(data_dict, top_n, batch_size, **kwargs):
        return dummy_candidates, dummy_df

    # strategy ファイル内で import 済みのシンボルを差し替え
    monkeypatch.setattr(
        "strategies.system6_strategy.generate_candidates_system6",
        fake_generate_candidates_system6,
    )

    data_dict = {"AAA": dummy_df, "BBB": dummy_df}

    # グローバル perf を有効化（strategy 内の get_global_perf が取得する）
    ps = enable_global_perf(True)
    with ps.run(latest_only=False):
        s.generate_candidates(data_dict)
    assert ps.output_path is not None
    data = json.loads(Path(ps.output_path).read_text(encoding="utf-8"))
    assert "system6" in data["per_system"]
    assert data["per_system"]["system6"]["candidate_count"] == 2
