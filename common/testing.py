"""
テストの決定性を高めるためのユーティリティ。

提供機能
- set_test_determinism: 乱数シードと環境の初期化

使い方
>>> from common.testing import set_test_determinism
>>> set_test_determinism()
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timezone


def set_test_determinism(seed: int = 42, tz: str = "UTC", frozen_date: str | None = None) -> None:
    """テストの決定性を確保するための初期化を実施。

    Args:
        seed: 乱数シード値。
        tz:   タイムゾーン（例: "UTC"）。
        frozen_date: `YYYY-MM-DD` 形式で日時を固定する文字列。`freezegun` を使う場合に有効。
    """

    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpyが無い環境でもテスト可能にする
        np = None  # type: ignore

    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass

    os.environ.setdefault("TZ", tz)

    # 任意で日時固定（呼び出し側で freezegun を使用）
    if frozen_date:
        try:
            from freezegun import freeze_time  # type: ignore

            dt = datetime.strptime(frozen_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            _ = freeze_time(dt)
            _.start()
        except Exception:
            # freezegun が無い/失敗してもテストは継続
            pass

