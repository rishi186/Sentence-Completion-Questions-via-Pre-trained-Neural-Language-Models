"""Configuration settings for the ESL Sentence Completion project."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the language model."""
    model_name: str = "facebook/bart-large"
    model_type: str = "bart"  # Options: bart, bert, roberta
    max_length: int = 512
    truncation_side: str = "left"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 8


@dataclass
class DataConfig:
    """Configuration for data processing."""
    blank_token: str = "_____"
    data_dir: str = "data"
    sample_file: str = "data/sample_questions.json"


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    output_dir: str = "results"
    save_predictions: bool = True


@dataclass
class TrainConfig:
    """Configuration for fine-tuning."""
    output_dir: str = "checkpoints"
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 500
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1


@dataclass
class Config:
    """Master configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT_CONFIG = Config()
