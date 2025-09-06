from common.config_loader import load_config_file


def test_load_config_file_yaml_default():
    cfg = load_config_file()
    assert isinstance(cfg, dict)
    # 既存の config.yaml がある前提で主要キーの一部を確認
    assert "risk" in cfg
    assert "data" in cfg

