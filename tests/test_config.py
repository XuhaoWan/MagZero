from pathlib import Path

from magzero.data_utils import load_yaml, yaml2dict


def test_default_config_loads():
    config_path = Path("configs/default.yaml")
    config = load_yaml(config_path)
    assert config.batch_size > 0
    assert config.hidden_dim > 0


def test_yaml2dict_returns_flat_mapping():
    config_path = Path("configs/default.yaml")
    config_dict = yaml2dict(config_path)
    assert isinstance(config_dict, dict)
    assert "atom_feat_dim" in config_dict
