"""Legacy cache utilities.

save_prepared_data_cache は完全撤去されました。

他モジュールが誤って import しても AttributeError となるだけにし、
ここでは互換のためシンボルを再定義しません。
"""

# Intentionally minimal: previous implementation removed.

__all__ = []
