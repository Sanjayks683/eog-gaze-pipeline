"""
src/models/deep_multitask.py
=============================
Deep multi-task Conv1D model.
  - Shared 1D-CNN backbone → 128-dim embedding
  - Classification head  → 4 classes (rest/saccade_onset/saccade_return/blink)
  - Regression head      → 2 outputs (H angle, V angle in degrees)

Input shape: (batch, 2, window_len)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from src.config import CFG


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool_size: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class EOGMultiTaskNet(nn.Module):
    """
    Conv1D multi-task network for EOG-based gaze estimation.

    Architecture (starting point — scale up/down based on dataset size):
      Conv(2→32, k=7) → BN → ReLU → MaxPool(2)
      Conv(32→64, k=5) → BN → ReLU → MaxPool(2)
      Conv(64→128, k=3) → BN → ReLU → AdaptiveAvgPool(1)
      Shared 128-dim embedding
      ├── Classification: Linear(128→64) → ReLU → Dropout → Linear(64→4)
      └── Regression:     Linear(128→64) → ReLU → Dropout → Linear(64→2)
    """

    def __init__(
        self,
        in_channels: int = None,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        num_classes: int = None,
        regression_outputs: int = None,
        dropout: float = None,
    ):
        super().__init__()
        in_channels = in_channels or CFG.model.in_channels
        conv_channels = conv_channels or CFG.model.conv_channels
        kernel_sizes = kernel_sizes or CFG.model.kernel_sizes
        num_classes = num_classes or CFG.model.num_classes
        regression_outputs = regression_outputs or CFG.model.regression_outputs
        dropout = dropout if dropout is not None else CFG.model.dropout

        assert len(conv_channels) == len(kernel_sizes), \
            "conv_channels and kernel_sizes must have the same length"

        backbone_layers = []
        in_ch = in_channels
        for i, (out_ch, k) in enumerate(zip(conv_channels, kernel_sizes)):
            if i < len(conv_channels) - 1:
                backbone_layers.append(ConvBlock(in_ch, out_ch, k, pool_size=2))
            else:
                backbone_layers.append(nn.Conv1d(in_ch, out_ch, k, padding=k // 2))
                backbone_layers.append(nn.BatchNorm1d(out_ch))
                backbone_layers.append(nn.ReLU())
            in_ch = out_ch

        self.backbone = nn.Sequential(*backbone_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        embed_dim = conv_channels[-1]

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, regression_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, 2, window_len) tensor

        Returns
        -------
        logits : (batch, num_classes) — for CrossEntropyLoss
        angles : (batch, 2)          — H and V gaze angles
        """
        features = self.backbone(x)
        embedding = self.global_pool(features).squeeze(-1)
        logits = self.cls_head(embedding)
        angles = self.reg_head(embedding)
        return logits, angles

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class predictions (batch,)."""
        with torch.no_grad():
            logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Return angle predictions (batch, 2)."""
        with torch.no_grad():
            _, angles = self.forward(x)
        return angles


class EOGLSTMNet(nn.Module):
    """
    LSTM multi-task network for EOG-based gaze estimation.
    Uses a bidirectional LSTM to capture temporal dependencies (drift) over the window.
    
    Improvements over vanilla LSTM:
      - Layer Normalization on LSTM output for training stability
      - Residual connection from raw input statistics to regression head
      - All hyperparameters read from CFG (no hardcoding)
    """
    def __init__(
        self,
        in_channels: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        num_classes: int = None,
        regression_outputs: int = None,
        dropout: float = None,
    ):
        super().__init__()
        in_channels = in_channels or CFG.model.in_channels
        hidden_size = hidden_size or CFG.model.lstm_hidden_size
        num_layers = num_layers or CFG.model.lstm_num_layers
        num_classes = num_classes or CFG.model.num_classes
        regression_outputs = regression_outputs or CFG.model.regression_outputs
        dropout = dropout if dropout is not None else CFG.model.dropout

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        embed_dim = hidden_size * 2
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim + in_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, regression_outputs),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_mean = x.mean(dim=2)
        
        x_seq = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x_seq)
        
        embedding = lstm_out[:, -1, :]
        embedding = self.layer_norm(embedding)
        
        logits = self.cls_head(embedding)
        
        reg_input = torch.cat([embedding, raw_mean], dim=1)
        angles = self.reg_head(reg_input)
        return logits, angles

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_angles(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, angles = self.forward(x)
        return angles

class TemporalAttention(nn.Module):
    """
    Learned temporal attention mechanism over sequence features.
    Computes soft alignment weights over all window time steps, allowing the network
    to focus dynamically on saccade onsets and blink transients regardless of alignment.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_features, max(16, in_features // 2)),
            nn.Tanh(),
            nn.Linear(max(16, in_features // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn_net(x)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context


class EOGConvLSTMNet(nn.Module):
    """
    Hybrid Conv1D + BiLSTM multi-task network with Temporal Attention for EOG gaze estimation.
      - Multi-scale 1D-CNN backbone → extracts short-time morphological features
      - Bidirectional LSTM → models temporal dependencies and baseline drift dynamics
      - Temporal Attention → dynamically pools feature sequence across window time steps
      - Classification head → 4 classes (rest/saccade_onset/saccade_return/blink)
      - Regression head     → 2 continuous gaze angles (H, V in degrees)
    """

    def __init__(
        self,
        in_channels: int = None,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        lstm_hidden: int = None,
        num_classes: int = None,
        regression_outputs: int = None,
        dropout: float = None,
    ):
        super().__init__()
        in_channels = in_channels or CFG.model.in_channels
        n_backbone = CFG.model.conv_bilstm_backbone_layers
        conv_channels = conv_channels or CFG.model.conv_channels[:n_backbone]
        kernel_sizes = kernel_sizes or CFG.model.kernel_sizes[:n_backbone]
        lstm_hidden = lstm_hidden or CFG.model.lstm_hidden_size
        num_classes = num_classes or CFG.model.num_classes
        regression_outputs = regression_outputs or CFG.model.regression_outputs
        dropout = dropout if dropout is not None else CFG.model.dropout

        backbone_layers = []
        in_ch = in_channels
        for out_ch, k in zip(conv_channels, kernel_sizes):
            backbone_layers.append(ConvBlock(in_ch, out_ch, k, pool_size=2))
            in_ch = out_ch

        self.conv_backbone = nn.Sequential(*backbone_layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        embed_dim = lstm_hidden * 2

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.attention = TemporalAttention(embed_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim + in_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, regression_outputs),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_mean = x.mean(dim=2)
        
        conv_feat = self.conv_backbone(x)
        conv_feat = conv_feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(conv_feat)
        embedding = self.attention(lstm_out)
        embedding = self.layer_norm(embedding)

        logits = self.cls_head(embedding)
        
        reg_input = torch.cat([embedding, raw_mean], dim=1)
        angles = self.reg_head(reg_input)
        return logits, angles

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_angles(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, angles = self.forward(x)
        return angles


class MultiTaskLoss(nn.Module):
    """
    Combined loss function supporting:
      1. Static weighting: CE + lambda * MSE
      2. Uncertainty weighting (Kendall et al., 2018):
         Loss = exp(-log_var_cls)*CE + 0.5*log_var_cls + exp(-log_var_reg)*MSE + 0.5*log_var_reg
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        lambda_reg: float = None,
        loss_type: str = None,
    ):
        super().__init__()
        if loss_type is None:
            loss_type = getattr(CFG.model, "loss_type", "uncertainty")
        if lambda_reg is None:
            lambda_reg = CFG.model.lambda_regression_loss

        self.loss_type = loss_type
        self.lambda_reg = lambda_reg
        label_smoothing = getattr(CFG.model, "label_smoothing", 0.0)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.mse_loss = nn.MSELoss()

        if self.loss_type == "uncertainty":
            self.log_var_cls = nn.Parameter(torch.zeros(1, requires_grad=True))
            self.log_var_reg = nn.Parameter(torch.zeros(1, requires_grad=True))
        elif self.loss_type == "range_preserving":
            huber_delta = getattr(CFG.model, "huber_delta", 2.0)
            self.huber_loss = nn.HuberLoss(delta=huber_delta)
            self.lambda_corr = getattr(CFG.model, "lambda_corr", 0.5)
            self.lambda_var = getattr(CFG.model, "lambda_var", 0.2)

    def forward(
        self,
        logits: torch.Tensor,
        y_class: torch.Tensor,
        angles: torch.Tensor,
        y_angle: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, ce_loss, mse_loss).
        Caller can log ce and mse separately for monitoring.
        """
        ce = self.ce_loss(logits, y_class)
        mse = self.mse_loss(angles, y_angle)

        if self.loss_type == "uncertainty":
            precision_cls = torch.exp(-self.log_var_cls)
            precision_reg = torch.exp(-self.log_var_reg)
            total = precision_cls * ce + 0.5 * self.log_var_cls + precision_reg * mse + 0.5 * self.log_var_reg
            total = total.squeeze()
        elif self.loss_type == "range_preserving":
            reg_loss = self.huber_loss(angles, y_angle)
            eps = 1e-6
            pred_h, pred_v = angles[:, 0], angles[:, 1]
            true_h, true_v = y_angle[:, 0], y_angle[:, 1]

            mean_pred_h, mean_true_h = torch.mean(pred_h), torch.mean(true_h)
            mean_pred_v, mean_true_v = torch.mean(pred_v), torch.mean(true_v)

            cov_h = torch.mean((pred_h - mean_pred_h) * (true_h - mean_true_h))
            var_pred_h = torch.var(pred_h, unbiased=False)
            var_true_h = torch.var(true_h, unbiased=False)
            corr_h = cov_h / (torch.sqrt(var_pred_h * var_true_h) + eps)

            cov_v = torch.mean((pred_v - mean_pred_v) * (true_v - mean_true_v))
            var_pred_v = torch.var(pred_v, unbiased=False)
            var_true_v = torch.var(true_v, unbiased=False)
            corr_v = cov_v / (torch.sqrt(var_pred_v * var_true_v) + eps)

            loss_corr = (1.0 - corr_h) + (1.0 - corr_v)

            std_ratio_h = torch.sqrt(var_pred_h + eps) / (torch.sqrt(var_true_h + eps) + eps)
            std_ratio_v = torch.sqrt(var_pred_v + eps) / (torch.sqrt(var_true_v + eps) + eps)
            loss_var = torch.abs(std_ratio_h - 1.0) + torch.abs(std_ratio_v - 1.0)

            total = ce + self.lambda_reg * reg_loss + self.lambda_corr * loss_corr + self.lambda_var * loss_var
            return total, ce, reg_loss
        else:
            total = ce + self.lambda_reg * mse

        return total, ce, mse


def build_model(
    model_type: Optional[str] = None,
    device: torch.device = None,
):
    """Build and return the model on the specified device."""
    if device is None:
        from src.config import get_device
        device = get_device()
    if model_type is None:
        model_type = CFG.model.model_type

    if model_type == "conv1d":
        model = EOGMultiTaskNet()
    elif model_type == "lstm":
        model = EOGLSTMNet()
    elif model_type == "conv_bilstm":
        model = EOGConvLSTMNet()
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            "Supported: 'conv1d', 'lstm', 'conv_bilstm'"
        )

    model = model.to(device)
    print(f"{model.__class__.__name__} (model_type='{model_type}') built on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def build_loss(
    device: torch.device = None,
    loss_type: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> MultiTaskLoss:
    """
    Build loss function with class weights and selected loss weighting mode.

    class_weights: explicit per-class weights (e.g. balanced weights derived
    from a fold's TRAIN labels). If None, falls back to CFG.model.class_weights.
    """
    if class_weights is None and CFG.model.class_weights is not None:
        class_weights = torch.tensor(CFG.model.class_weights, dtype=torch.float32)
        if device is not None:
            class_weights = class_weights.to(device)

    loss_fn = MultiTaskLoss(class_weights=class_weights, loss_type=loss_type)
    if device is not None:
        loss_fn = loss_fn.to(device)
    return loss_fn
