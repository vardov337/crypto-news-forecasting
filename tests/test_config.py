import pytest

from cryptonews import config
from cryptonews.config import REQUIRED_SECTIONS, load_config


def test_config_loads_with_all_sections():
    cfg = load_config()
    for section in REQUIRED_SECTIONS:
        assert section in cfg


def test_feature_sets_use_known_blocks():
    cfg = load_config()
    blocks = {"price"} | set(cfg["features"]["news"]["languages"])
    for name, parts in cfg["features"]["feature_sets"].items():
        assert set(parts) <= blocks, name


def test_protocol_consistency():
    cfg = load_config()
    strategy = cfg["evaluation"]["strategy"]
    assert strategy["primary_cost_bp"] in strategy["costs_bp"]
    assert cfg["evaluation"]["selection"]["cost_bp"] == strategy["primary_cost_bp"]
    val = cfg["validation"]
    assert val["test_months"] % val["fold_months"] == 0
    assert cfg["seeds"]["default"] and cfg["seeds"]["lstm"]
    assert cfg["data"]["prices"]["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_data_dir_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTONEWS_DATA_DIR", str(tmp_path))
    assert load_config()["paths"]["data_dir"] == tmp_path


def test_colab_without_drive_paths_refuses_to_run(monkeypatch):
    """После перезапуска Colab пути к Диску теряются — скрипт должен остановиться,
    а не писать данные во временную папку."""
    monkeypatch.setattr(config, "running_in_colab", lambda: True)
    monkeypatch.setenv("CRYPTONEWS_DATA_DIR", "")
    monkeypatch.setenv("CRYPTONEWS_RESULTS_DIR", "")
    with pytest.raises(SystemExit, match="ячейки 1–3"):
        load_config()


def test_colab_with_drive_paths_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "running_in_colab", lambda: True)
    monkeypatch.setenv("CRYPTONEWS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRYPTONEWS_RESULTS_DIR", str(tmp_path / "results"))
    cfg = load_config()
    assert cfg["paths"]["data_dir"] == tmp_path / "data"
