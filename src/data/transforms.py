import random
from pathlib import Path
from typing import Tuple, Optional
import torch
from tqdm import tqdm


class MultimodalSpatialAugmentation:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[-1])
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[-2])
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            tensor = torch.rot90(tensor, k=k, dims=[-2, -1])
        return tensor


class MultimodalNormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + self.eps)


class ComposeMultimodalTransforms:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            tensor = t(tensor)
        return tensor


def compute_channel_statistics(
    dataset,
    cache_path: Optional[Path] = None,
    num_channels: int = 14
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None and cache_path.is_file():
        cached = torch.load(cache_path, weights_only=True)
        return cached["mean"], cached["std"]

    channel_sum = torch.zeros(num_channels, dtype=torch.float64)
    channel_sq_sum = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0

    for i in tqdm(range(len(dataset)), desc="Calcul statistici canale"):
        tensor, _ = dataset.get_raw(i)
        tensor_64 = tensor.to(torch.float64)
        channel_sum += tensor_64.sum(dim=[-2, -1])
        channel_sq_sum += (tensor_64 ** 2).sum(dim=[-2, -1])
        total_pixels += tensor.shape[-2] * tensor.shape[-1]

    mean = (channel_sum / total_pixels).to(torch.float32)
    variance = (channel_sq_sum / total_pixels) - (channel_sum / total_pixels) ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-8)).to(torch.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, cache_path)

    return mean, std