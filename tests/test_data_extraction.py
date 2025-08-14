from pathlib import Path
from textwrap import dedent

from comparopt.data_extraction import load_config


def test_load_config_successfully(tmp_path: Path):
    """
    Tests that the `load_config` function correctly parses a valid YAML file.
    """
    config_content = dedent("""
        experiment:
          name: Test_Experiment
          seed: 42
        model:
          layers:
            - { nodes: 10, activation: relu }
        data:
          name: Test Data
          path: ./test_data
          split: { train_percent: 80, test_percent: 20 }
        training:
          optimizer: { name: SGD, params: { learning_rate: 0.01 } }
          epochs: 10
          batch_size: 16
    """)
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(config_content)

    config = load_config(config_file)

    assert config.experiment.name == 'Test Experiment'
    assert config.training.optimizer.name == 'SGD'
    assert config.model.layers[0].nodes == 10
    assert config.data.split.test_percent == 20
