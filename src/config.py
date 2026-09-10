"""
src/config.py
=============
Single source of truth for every tunable parameter.
Nothing downstream should hard-code a magic number that belongs here.

All values are STARTING POINTS verified against the real dataset fs in Phase 0.
Adjust after reading each dataset's Data Description file.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml
import os

@dataclass
class PreprocessingConfig:
    highpass_cutoff_hz: float = 0.2
    lowpass_cutoff_hz: float = 30.0
    filter_order: int = 4
    notch_hz: Optional[float] = 50.0
    notch_quality_factor: float = 30.0
    blink_threshold_std: float = 4.5
    blink_min_duration_ms: float = 40.0
    drift_removal_method: str = "highpass"
    polynomial_detrend_order: int = 2
    use_augmentation: bool = True

@dataclass
class SegmentationConfig:
    window_ms: float = 300.0
    stride_ms: float = 150.0
    classes: List[str] = field(default_factory=lambda: [
        "rest",
        "saccade_onset",
        "saccade_return_or_second",
        "blink"
    ])
    majority_overlap_threshold: float = 0.5

@dataclass
class CVConfig:
    strategy: str = "group_kfold"
    k: int = 5
    random_seed: int = 42
    use_grid_search: bool = False
    grid_search_max_samples: int = 20000
    svm_max_train_samples: int = 50000

@dataclass
class ModelConfig:
    model_type: str = "conv_bilstm"
    loss_type: str = "uncertainty"
    in_channels: int = 2
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    dropout: float = 0.3
    num_classes: int = 4
    regression_outputs: int = 2

    lambda_regression_loss: float = 1.0
    lambda_corr: float = 0.5
    lambda_var: float = 0.2
    huber_delta: float = 2.0
    calibration_prompt_sec: float = 30.0
    class_weights: Optional[List[float]] = None
    auto_class_weights: bool = True

    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 5.0

    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    conv_bilstm_backbone_layers: int = 2

@dataclass
class AugmentationConfig:
    scale_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.02
    channel_dropout_prob: float = 0.1
    shift_max_samples: int = 2

@dataclass
class DataConfig:
    fs_hz_by_dataset: Dict[str, float] = field(default_factory=lambda: {
        "dataset1": 256.0, "dataset2": 256.0, "dataset3": 256.0, "dataset4": 256.0,
    })
    fs_fallback_hz: float = 256.0
    datasets_to_load: List[str] = field(default_factory=lambda: [
        "dataset1", "dataset2", "dataset3", "dataset4",
    ])
    window_channel_mode: str = "bipolar"

class PathConfig:
    """
    Filesystem layout. `folds_file` is a property derived from `data_processed`
    so that overriding data_processed (CLI/YAML/tests) always redirects the
    folds file with it.
    """

    def __init__(self):
        self.project_root: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.data_raw: str = os.path.join(self.project_root, "data", "raw")
        self.data_processed: str = os.path.join(self.project_root, "data", "processed")
        self.reports_figures: str = os.path.join(self.project_root, "reports", "figures")
        self.results: str = os.path.join(self.project_root, "reports")

        self.dataset1_dir: str = os.path.join(self.data_raw, "Dataset_ZeroCentred")
        self.dataset2_dir: str = os.path.join(self.data_raw, "Dataset_Stationary")
        self.dataset3_dir: str = os.path.join(self.data_raw, "Dataset_NonStationary")
        self.dataset4_dir: str = os.path.join(self.data_raw, "Dataset_Isotropic")

    @property
    def folds_file(self) -> str:
        return os.path.join(self.data_processed, "cv_folds.pkl")

@dataclass
class DeviceConfig:
    auto_detect_cuda: bool = True

@dataclass
class Config:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)


CFG = Config()

_CONFIG_SECTIONS = (
    "preprocessing", "segmentation", "cv", "model",
    "augmentation", "data", "paths", "device",
)


def get_device():
    """Return the torch.device to use for training."""
    import torch
    if CFG.device.auto_detect_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def window_samples(fs: float) -> int:
    """Convert window_ms to integer sample count given sampling rate fs."""
    return int(round(CFG.segmentation.window_ms * fs / 1000.0))


def stride_samples(fs: float) -> int:
    """Convert stride_ms to integer sample count given sampling rate fs."""
    return int(round(CFG.segmentation.stride_ms * fs / 1000.0))


def blink_min_samples(fs: float) -> int:
    """Convert blink_min_duration_ms to integer sample count given fs."""
    return int(round(CFG.preprocessing.blink_min_duration_ms * fs / 1000.0))


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    Override the GLOBAL CFG defaults from a YAML file (in place) and return it.
    Only keys present in the YAML override; all others remain at their current
    value. Every downstream module reads the global CFG, so mutating it here is
    what actually applies the overrides.
    """
    with open(yaml_path, "r") as f:
        overrides = yaml.safe_load(f) or {}

    if not isinstance(overrides, dict):
        raise ValueError(f"Config file {yaml_path!r} must contain a YAML mapping.")

    for section, values in overrides.items():
        if section not in _CONFIG_SECTIONS:
            raise ValueError(
                f"Unknown config section {section!r} in {yaml_path}. "
                f"Valid sections: {', '.join(_CONFIG_SECTIONS)}."
            )
        if not isinstance(values, dict):
            raise ValueError(f"Section {section!r} in {yaml_path} must be a mapping.")
        sub_cfg = getattr(CFG, section)
        for k, v in values.items():
            if not hasattr(sub_cfg, k):
                raise ValueError(
                    f"Unknown config key {section}.{k} in {yaml_path}. "
                    "Check src/config.py for valid field names."
                )
            setattr(sub_cfg, k, v)
    return CFG
