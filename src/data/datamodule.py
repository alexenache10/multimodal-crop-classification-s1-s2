from pathlib import Path
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.dataset import (
    DACIA5MultimodalDataset,
    parse_multimodal_pairs,
    CODE_TO_IDX,
    LABEL_MAPPING,
)
from src.data.transforms import (
    MultimodalSpatialAugmentation,
    MultimodalNormalize,
    ComposeMultimodalTransforms,
    compute_channel_statistics,
)


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    proj_cfg = config.get("project", {})

    root = Path(data_cfg["root_dir"])
    if (root / "training_2020-2023").is_dir():
        base_dir = root
    elif (root / "32x32_patches" / "32x32_SAR+MSI_patches" / "problem1").is_dir():
        base_dir = root / "32x32_patches" / "32x32_SAR+MSI_patches" / "problem1"
    else:
        base_dir = root

    train_dir = base_dir / "training_2020-2023"
    test_dir = base_dir / "test_2024"

    file_format = data_cfg.get("file_format", "tif")
    val_split = float(data_cfg.get("val_split", 0.15))
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 4))
    seed = int(proj_cfg.get("seed", 42))

    train_val_samples = parse_multimodal_pairs(train_dir, "train.csv", file_format)
    test_samples = parse_multimodal_pairs(test_dir, "test.csv", file_format)

    if len(train_val_samples) == 0:
        raise RuntimeError(f"Nu s-au putut incarca esantioane din {train_dir}")

    train_val_labels = [s[2] for s in train_val_samples]
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_split,
        stratify=train_val_labels,
        random_state=seed,
    )

    raw_train_ds = DACIA5MultimodalDataset(train_samples, transform=None)
    stats_cache = Path("data/processed/channel_stats.pt")
    mean, std = compute_channel_statistics(
        raw_train_ds,
        cache_path=stats_cache,
        num_channels=data_cfg.get("in_channels_s1", 2) + data_cfg.get("in_channels_s2", 12),
    )

    train_transform = ComposeMultimodalTransforms(
        [
            MultimodalSpatialAugmentation(),
            MultimodalNormalize(mean=mean, std=std),
        ]
    )
    eval_transform = ComposeMultimodalTransforms(
        [
            MultimodalNormalize(mean=mean, std=std),
        ]
    )

    train_dataset = DACIA5MultimodalDataset(train_samples, transform=train_transform)
    val_dataset = DACIA5MultimodalDataset(val_samples, transform=eval_transform)
    test_dataset = DACIA5MultimodalDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    metadata = {
        "mean": mean,
        "std": std,
        "num_classes": len(CODE_TO_IDX),
        "code_to_idx": CODE_TO_IDX,
        "label_mapping": LABEL_MAPPING,
        "counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    }

    return train_loader, val_loader, test_loader, metadata