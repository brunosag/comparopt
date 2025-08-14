from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentDetailsConfig:
    name: str
    seed: int


@dataclass
class LayerConfig:
    nodes: int
    activation: str


@dataclass
class ModelConfig:
    layers: list[LayerConfig]


@dataclass
class DataSplitConfig:
    train_percent: int
    test_percent: int


@dataclass
class DataConfig:
    name: str
    path: str
    split: DataSplitConfig


@dataclass
class OptimizerConfig:
    name: str
    params: dict[str, Any]


@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig
    epochs: int
    batch_size: int


@dataclass
class ExperimentConfig:
    experiment: ExperimentDetailsConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


def load_config(path: Path) -> ExperimentConfig:
    """Loads and parses a YAML config file into structured dataclasses."""
    with open(path, 'r') as file:
        raw_config = yaml.safe_load(file)

    return ExperimentConfig(
        experiment=ExperimentDetailsConfig(**raw_config['experiment']),
        model=ModelConfig(
            layers=[LayerConfig(**layer) for layer in raw_config['model']['layers']]
        ),
        data=DataConfig(
            name=raw_config['data']['name'],
            path=raw_config['data']['path'],
            split=DataSplitConfig(**raw_config['data']['split']),
        ),
        training=TrainingConfig(
            optimizer=OptimizerConfig(**raw_config['training']['optimizer']),
            epochs=raw_config['training']['epochs'],
            batch_size=raw_config['training']['batch_size'],
        ),
    )
