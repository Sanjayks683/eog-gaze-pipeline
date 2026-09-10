"""
tests/test_config.py
====================
Tests for the global config singleton and YAML overrides.
"""

from __future__ import annotations

import pytest
import yaml

from src.config import CFG, Config, _CONFIG_SECTIONS, load_config_from_yaml


@pytest.fixture()
def restore_cfg():
    """Snapshot the global CFG and restore it after each test."""
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}

    yield CFG

    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _write_yaml(tmp_path, data) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_yaml_overrides_update_global_cfg(tmp_path, restore_cfg):
    """load_config_from_yaml must mutate the GLOBAL CFG (what the pipeline reads)."""
    original_lr = CFG.model.lr
    yaml_path = _write_yaml(tmp_path, {
        "model": {"lr": 0.123, "batch_size": 32},
        "preprocessing": {"highpass_cutoff_hz": 0.3},
        "augmentation": {"noise_std": 0.05},
        "data": {"fs_fallback_hz": 500.0},
    })

    returned = load_config_from_yaml(yaml_path)

    assert returned is CFG
    assert CFG.model.lr == 0.123
    assert CFG.model.lr != original_lr or original_lr == 0.123
    assert CFG.model.batch_size == 32
    assert CFG.preprocessing.highpass_cutoff_hz == 0.3
    assert CFG.augmentation.noise_std == 0.05
    assert CFG.data.fs_fallback_hz == 500.0
    assert CFG.model.max_epochs == Config().model.max_epochs


def test_yaml_unknown_section_rejected(tmp_path, restore_cfg):
    yaml_path = _write_yaml(tmp_path, {"not_a_section": {"x": 1}})
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config_from_yaml(yaml_path)


def test_yaml_unknown_key_rejected(tmp_path, restore_cfg):
    yaml_path = _write_yaml(tmp_path, {"model": {"not_a_real_key": 1}})
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config_from_yaml(yaml_path)


def test_fs_comes_from_documented_table(tmp_path, restore_cfg):
    """All four EyeCon datasets are 256 Hz per their Data Description PDFs;
    the PDF can't be text-parsed, so the table must win over any fallback."""
    from src.data.loaders import _resolve_fs
    for name in ("dataset1", "dataset2", "dataset3", "dataset4"):
        assert _resolve_fs(name, str(tmp_path)) == 256.0
    CFG.data.fs_hz_by_dataset = {}
    assert _resolve_fs("dataset2", str(tmp_path)) is None


def test_yaml_empty_file_keeps_defaults(tmp_path, restore_cfg):
    yaml_path = _write_yaml(tmp_path, {})
    before = CFG.model.lr
    load_config_from_yaml(yaml_path)
    assert CFG.model.lr == before
