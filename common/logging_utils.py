import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from config.settings import Settings
import time
from typing import Callable


def setup_logging(settings: Settings) -> logging.Logger:
    """ロギング設定を標準 logging で初期化して root ロガーを返す。
    - 日次ローテーション: rotation == "daily"
    - それ以外: サイズローテーション（MB 指定の例: "10 MB" は 10*1024*1024）
    """
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    log_dir = Path(settings.LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / settings.logging.filename

    logger = logging.getLogger()
    logger.setLevel(level)

    # 既存ハンドラをクリア
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rotation = settings.logging.rotation.lower()
    if rotation == "daily":
        handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_path), when="midnight", backupCount=7, encoding="utf-8"
        )
    else:
        # 例: "10 MB" -> 10485760
        size_bytes: Optional[int] = None
        try:
            num = float(rotation.split()[0])
            unit = rotation.split()[1].lower() if len(rotation.split()) > 1 else "b"
            mult = 1
            if unit.startswith("k"):
                mult = 1024
            elif unit.startswith("m"):
                mult = 1024 * 1024
            elif unit.startswith("g"):
                mult = 1024 * 1024 * 1024
            size_bytes = int(num * mult)
        except Exception:
            size_bytes = 10 * 1024 * 1024
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path), maxBytes=size_bytes, backupCount=5, encoding="utf-8"
        )

    handler.setFormatter(fmt)
    logger.addHandler(handler)

    # コンソールにも出す
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.debug("Logging initialized")
    return logger


def log_with_progress(
    i: int,
    total: int,
    start_time: float,
    *,
    prefix: str = "処理",
    batch: int = 50,
    log_func: Callable[[str], None] | None = None,
    progress_func: Callable[[float], None] | None = None,
    extra_msg: str | None = None,
    unit: str = "件",
):
    """ストリームリット/CLIの両方で使える共通進捗ログ。
    - log_func: 文字列を受け取る関数（例: `st.text`, `logger.info`）
    - progress_func: 0..1 の進捗率を受け取る関数（例: `st.progress`）
    """
    if i % batch != 0 and i != total:
        return
    elapsed = time.time() - start_time
    remain = (elapsed / max(i, 1)) * (total - i) if total > 0 else 0
    msg = (
        f"{prefix}: {i}/{total} {unit} 完了 | "
        f"経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒 / "
        f"残り: 約{int(remain // 60)}分{int(remain % 60)}秒"
    )
    if extra_msg:
        msg += f"\n{extra_msg}"
    if log_func:
        try:
            log_func(msg)
        except Exception:
            pass
    if progress_func:
        try:
            progress_func(i / total if total else 0.0)
        except Exception:
            pass
