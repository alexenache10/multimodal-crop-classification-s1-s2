import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
import pandas as pd
import rasterio
import scipy.io
import torch
from torch.utils.data import Dataset


LABEL_MAPPING: Dict[int, str] = {
    101: "Grau comun de toamna",
    108: "Porumb",
    131: "Porumb siloz",
    151: "Mazare",
    202: "Rapita de toamna",
    253: "Cartofi tarzii",
    254: "Alte culturi de cartofi",
    1010: "Grau comun de primavara",
    2037: "Soia",
    3017: "Sfecla de zahar",
    9747: "Lucerna soiul 1",
    9748: "Lucerna soiul 2",
}

CODE_TO_IDX: Dict[int, int] = {
    code: idx for idx, code in enumerate(sorted(LABEL_MAPPING.keys()))
}

IDX_TO_CODE: Dict[int, int] = {
    idx: code for code, idx in CODE_TO_IDX.items()
}


class DACIA5MultimodalDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path, int]],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _read_file(self, path: Path) -> np.ndarray:
        if path.suffix == ".mat":
            mat = scipy.io.loadmat(str(path))
            for k, val in mat.items():
                if not k.startswith("__"):
                    arr = np.asarray(val, dtype=np.float32)
                    if arr.ndim == 3 and arr.shape[0] not in (2, 12, 14) and arr.shape[2] in (2, 12, 14):
                        arr = np.transpose(arr, (2, 0, 1))
                    return arr
        with rasterio.open(path) as src:
            data = src.read()
        return data.astype(np.float32)

    def get_raw(self, idx: int) -> Tuple[torch.Tensor, int]:
        s1_path, s2_path, label = self.samples[idx]
        s1_data = self._read_file(s1_path)
        s2_data = self._read_file(s2_path)
        fused = np.concatenate([s1_data, s2_data], axis=0)
        return torch.from_numpy(fused), label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor, label = self.get_raw(idx)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)


def extract_crop_code(name: str) -> Optional[int]:
    match = re.search(r"_\d{8}_(\d+)_", name)
    if match:
        code = int(match.group(1))
        if code in CODE_TO_IDX:
            return code
    tokens = Path(name).stem.split("_")
    if len(tokens) >= 4 and tokens[-2].isdigit():
        code = int(tokens[-2])
        if code in CODE_TO_IDX:
            return code
    return None


def get_modality_directory(stage_dir: Path, modality: str, file_format: str = "tif") -> Path:
    candidates = [
        stage_dir / f"{modality}_patches_{file_format}",
        stage_dir / f"{modality}_patches_tiff",
        stage_dir / f"{modality}_patches_tif",
        stage_dir / f"{modality}_patches_mat",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def resolve_patch_file(folder: Path, raw_filename: str, file_format: str = "tif") -> Optional[Path]:
    clean_stem = Path(raw_filename).stem
    candidates = [
        folder / f"{clean_stem}.{file_format}",
        folder / f"{clean_stem}.tif",
        folder / f"{clean_stem}.tiff",
        folder / f"{clean_stem}.mat",
        folder / raw_filename,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def parse_multimodal_pairs(
    stage_dir: Path,
    csv_filename: str = "train.csv",
    file_format: str = "tif"
) -> List[Tuple[Path, Path, int]]:
    csv_path = stage_dir / csv_filename
    s1_dir = get_modality_directory(stage_dir, "sentinel1", file_format)
    s2_dir = get_modality_directory(stage_dir, "sentinel2", file_format)

    df = pd.read_csv(csv_path)
    if any("patch" in str(col).lower() for col in df.columns):
        df = pd.read_csv(csv_path, header=None)

    s1_col = df.columns[0]
    s2_col = df.columns[1]
    for col in df.columns:
        c_low = str(col).lower()
        first_val = str(df[col].iloc[0]).lower() if len(df) > 0 else ""
        if "s1" in c_low or "sentinel1" in c_low or "s1" in first_val:
            s1_col = col
        elif "s2" in c_low or "sentinel2" in c_low or "s2" in first_val:
            s2_col = col

    samples: List[Tuple[Path, Path, int]] = []
    for _, row in df.iterrows():
        s1_raw = str(row[s1_col]).strip()
        s2_raw = str(row[s2_col]).strip()

        s1_path = resolve_patch_file(s1_dir, s1_raw, file_format)
        s2_path = resolve_patch_file(s2_dir, s2_raw, file_format)

        if s1_path is None or s2_path is None:
            continue

        crop_code = extract_crop_code(s1_path.name)
        if crop_code is None or crop_code not in CODE_TO_IDX:
            continue

        samples.append((s1_path, s2_path, CODE_TO_IDX[crop_code]))

    return samples