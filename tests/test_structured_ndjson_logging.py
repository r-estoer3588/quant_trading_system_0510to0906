import json
from pathlib import Path

import pytest

from common.structured_log_ndjson import close_global_writer


@pytest.mark.parametrize("flag", ["1", "true", "yes"])  # 動作フラグ多様性
def test_ndjson_writer_basic(flag, tmp_path, monkeypatch):
    # 出力先を一時ディレクトリに向ける
    monkeypatch.setenv("STRUCTURED_LOG_NDJSON", flag)
    monkeypatch.setenv("STRUCTURED_LOG_NDJSON_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTURED_UI_LOGS", "1")  # 構造化UIログ経由で生成

    # 遅延 import される対象関数を呼び出すため run_all_systems_today から emit を再利用
    from scripts.run_all_systems_today import _emit_ui_log  # type: ignore

    messages = ["alpha", "beta", "gamma"]
    for m in messages:
        _emit_ui_log(m)

    close_global_writer()

    # 生成ファイルを探索
    files = list(Path(tmp_path).glob("*.ndjson"))
    assert files, "NDJSON file not created"
    fp = files[0]
    lines = fp.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(messages)

    parsed = []
    for line in lines:
        parsed.append(json.loads(line))

    # v, ts, iso, lvl, msg が存在し msg の順序保たれている
    assert [p["msg"] for p in parsed] == messages
    for p in parsed:
        assert p.get("v") == 1
        assert isinstance(p["ts"], int)
        assert "T" in p["iso"] and p["iso"].endswith("Z")
        assert p.get("lvl") == "INFO"
        # 新規フィールド: elapsed_ms が 0 以上
        assert isinstance(p.get("elapsed_ms"), int)
        assert p["elapsed_ms"] >= 0
        # system / phase / phase_status はメッセージによっては無いことを許容
        if "system" in p:
            assert p["system"].startswith("system")


def test_ndjson_writer_buffer_lines(tmp_path, monkeypatch):
    """行数バッファでまとめて flush されるか (STRUCTURED_LOG_BUFFER_LINES) を確認。"""
    monkeypatch.setenv("STRUCTURED_LOG_NDJSON", "1")
    monkeypatch.setenv("STRUCTURED_LOG_NDJSON_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTURED_UI_LOGS", "1")
    monkeypatch.setenv("STRUCTURED_LOG_BUFFER_LINES", "2")  # 2 行貯めて flush

    from scripts.run_all_systems_today import _emit_ui_log  # type: ignore

    msgs = ["m1", "m2", "m3"]  # 3 行 => 3 行一括 flush 期待
    for m in msgs:
        _emit_ui_log(m)
    close_global_writer()

    files = list(Path(tmp_path).glob("*.ndjson"))
    assert len(files) == 1, "バッファのみでローテーションは発生しないはず"
    content = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == len(msgs)
    parsed = [json.loads(line_str)["msg"] for line_str in content]
    assert parsed == msgs


def test_ndjson_writer_rotation_by_lines(tmp_path, monkeypatch):
    """行数ベースのローテーション (STRUCTURED_LOG_MAX_LINES) を確認。"""
    monkeypatch.setenv("STRUCTURED_LOG_NDJSON", "1")
    monkeypatch.setenv("STRUCTURED_LOG_NDJSON_DIR", str(tmp_path))
    monkeypatch.setenv("STRUCTURED_UI_LOGS", "1")
    monkeypatch.setenv("STRUCTURED_LOG_MAX_LINES", "2")  # 2 行到達毎に rotate

    from scripts.run_all_systems_today import _emit_ui_log  # type: ignore

    msgs = [f"line{i}" for i in range(5)]  # 5 行 => 3 ファイル (2,2,1)
    for m in msgs:
        _emit_ui_log(m)
    close_global_writer()

    files = sorted(Path(tmp_path).glob("*.ndjson"))
    assert len(files) >= 2, "行数ローテーションで複数ファイルが必要"
    # 総行数が 5 行であること
    total_lines = 0
    collected_msgs: list[str] = []
    for f in files:
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        total_lines += len(lines)
        for line_str in lines:
            collected_msgs.append(json.loads(line_str)["msg"])
    assert total_lines == len(msgs)
    assert collected_msgs == msgs
    # part サフィックス付きファイルが少なくとも 1 つ存在
    assert any("_part" in f.name for f in files), "ローテーション後の part ファイルが見つからない"
