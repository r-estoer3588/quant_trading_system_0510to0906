"""キャッシュ関連のカスタム例外定義"""

from __future__ import annotations


class CacheError(Exception):
    """キャッシュ操作に関連する基底例外クラス"""

    pass


class CacheIOError(CacheError):
    """ファイルI/O操作で発生する例外"""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.file_path = file_path
        self.original_error = original_error


class CacheDataError(CacheError):
    """データ処理・変換で発生する例外"""

    def __init__(self, message: str, ticker: str | None = None, data_info: dict | None = None):
        super().__init__(message)
        self.ticker = ticker
        self.data_info = data_info or {}


class CacheValidationError(CacheError):
    """データの妥当性チェックで発生する例外"""

    def __init__(self, message: str, validation_type: str, failed_checks: list[str] | None = None):
        super().__init__(message)
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []


class CacheHealthError(CacheError):
    """健全性チェックで発見される問題"""

    def __init__(self, message: str, ticker: str, profile: str, issues: dict | None = None):
        super().__init__(message)
        self.ticker = ticker
        self.profile = profile
        self.issues = issues or {}


class CacheConfigError(CacheError):
    """設定・構成関連の例外"""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message)
        self.config_key = config_key


class CachePruneError(CacheError):
    """Rolling キャッシュの剪定処理で発生する例外"""

    def __init__(
        self,
        message: str,
        anchor_ticker: str | None = None,
        affected_files: list[str] | None = None,
    ):
        super().__init__(message)
        self.anchor_ticker = anchor_ticker
        self.affected_files = affected_files or []
