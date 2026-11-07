import copy
from pathlib import Path
import pytest
import torch

from src.utils.config import load_config
from src.data.transforms import (
    MultimodalSpatialAugmentation,
    MultimodalNormalize,
)
from src.data.datamodule import create_dataloaders


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


def test_spatial_augmentation_invariance():
    x = torch.randn(14, 32, 32, dtype=torch.float32)
    aug = MultimodalSpatialAugmentation()
    out = aug(x)
    assert out.shape == (14, 32, 32)
    assert out.dtype == torch.float32


def test_normalization():
    x = torch.randn(14, 32, 32, dtype=torch.float32)
    mean = torch.zeros(14, dtype=torch.float32)
    std = torch.ones(14, dtype=torch.float32)
    norm = MultimodalNormalize(mean=mean, std=std)
    out = norm(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_dataloaders_execution(config):
    data_dir = Path(config["data"]["root_dir"])
    if not data_dir.exists():
        pytest.skip(f"Directorul {data_dir} nu este accesibil in mediul curent.")

    test_cfg = copy.deepcopy(config)
    test_cfg["training"]["batch_size"] = 16
    test_cfg["training"]["num_workers"] = 0

    train_loader, val_loader, test_loader, meta = create_dataloaders(test_cfg)

    assert meta["counts"]["train"] > 0
    assert meta["counts"]["val"] > 0
    assert meta["counts"]["test"] > 0
    assert meta["num_classes"] == 12

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (16, 14, 32, 32)
    assert batch_y.shape == (16,)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.long