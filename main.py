import sys
from pathlib import Path
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datamodule import create_dataloaders


def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg.get("project", {}).get("seed", 42))

    data_dir = Path(cfg["data"]["root_dir"])
    if not data_dir.exists():
        print(f"Calea configurata nu exista: {data_dir}")
        sys.exit(1)

    train_loader, val_loader, test_loader, meta = create_dataloaders(cfg)

    print("=" * 65)
    print(f"Dataset multimodal: DACIA5 (Sentinel-1 SAR + Sentinel-2 MSI)")
    print(f"Numar clase: {meta['num_classes']}")
    print(f"Split Train (2020-2023, 85%): {meta['counts']['train']} patch-uri")
    print(f"Split Val   (2020-2023, 15%): {meta['counts']['val']} patch-uri")
    print(f"Split Test  (2024, 100%):     {meta['counts']['test']} patch-uri")
    print("=" * 65)

    train_x, train_y = next(iter(train_loader))
    val_x, val_y = next(iter(val_loader))
    test_x, test_y = next(iter(test_loader))

    assert train_x.shape[1:] == (14, 32, 32)
    assert val_x.shape[1:] == (14, 32, 32)
    assert test_x.shape[1:] == (14, 32, 32)
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.long

    print(f"Train Batch Tensor: X={tuple(train_x.shape)}, Y={tuple(train_y.shape)}")
    print(f"Val   Batch Tensor: X={tuple(val_x.shape)}, Y={tuple(val_y.shape)}")
    print(f"Test  Batch Tensor: X={tuple(test_x.shape)}, Y={tuple(test_y.shape)}")
    print("-" * 65)
    print(f"Medie canale (0-1 SAR, 2-4 MSI): {meta['mean'][:5].numpy()}")
    print(f"Deviatie std (0-1 SAR, 2-4 MSI): {meta['std'][:5].numpy()}")
    print("=" * 65)
    print("Verificare commit #1 - Dataset si DataLoader validat cu succes.")


if __name__ == "__main__":
    main()