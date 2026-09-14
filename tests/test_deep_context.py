"""
tests/test_deep_context.py
==========================
The deep model's optional context features (model.context_features).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import CFG, _CONFIG_SECTIONS
from src.models.deep_multitask import EOGConvLSTMNet, build_loss
from src.training.train import _standardise_context, config_fingerprint, eval_epoch, make_dataloader, train_epoch


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def test_conv_bilstm_joins_context_to_the_regression_head(restore_cfg):
    CFG.model.in_channels, CFG.model.context_features, CFG.model.context_dim = 2, True, 5
    torch.manual_seed(0)
    model = EOGConvLSTMNet().eval()
    x, ctx = torch.randn(4, 2, 77), torch.randn(4, 5)

    _, angles = model(x, ctx)
    _, shifted = model(x, ctx + 1.0)
    assert angles.shape == (4, 2)
    assert not torch.allclose(angles, shifted)
    with pytest.raises(ValueError, match="context"):
        model(x)


def test_context_is_standardised_on_training_windows_and_carried_through_training(restore_cfg):
    rng = np.random.RandomState(0)
    train, test = rng.normal(5.0, 2.0, (200, 3)), rng.normal(5.0, 2.0, (50, 3))
    train[0, 1] = np.nan
    tr, te = _standardise_context(train, test)
    assert np.allclose(tr.mean(axis=0), 0.0, atol=0.05) and np.isfinite(tr).all() and np.all(np.abs(te) <= 10)

    CFG.model.in_channels, CFG.model.context_features, CFG.model.context_dim = 2, True, 3
    CFG.model.mixup_alpha = 0.2
    X = rng.randn(64, 2, 77).astype(np.float32)
    y_cls = rng.randint(0, CFG.model.num_classes, 64)
    y_ang = rng.randn(64, 2).astype(np.float32)
    loader = make_dataloader(X, y_cls, y_ang, 16, context=tr[:64])
    assert len(next(iter(loader))) == 4

    device = torch.device("cpu")
    model, criterion = EOGConvLSTMNet(), build_loss(device=device)
    metrics = train_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=1e-3), criterion, device)
    assert np.isfinite(metrics["loss"])
    assert eval_epoch(model, loader, criterion, device)["angle_pred"].shape == (64, 2)


def test_context_switch_leaves_existing_checkpoint_fingerprints_alone(restore_cfg):
    before = config_fingerprint((10, 2, 77))
    CFG.preprocessing.context_lags_sec = [1.0, 2.0]  # context settings do not reach a plain deep model
    assert config_fingerprint((10, 2, 77)) == before
    CFG.model.context_features, CFG.model.context_dim = True, 8
    assert config_fingerprint((10, 2, 77)) != before
