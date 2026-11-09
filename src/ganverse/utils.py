"""Utilidades compartidas: semillas, dispositivo de cómputo y guardado de imágenes."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """Fija la semilla en random, numpy y torch (CPU + backends) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Elige MPS (Apple GPU) > CUDA > CPU, salvo que se fuerce un dispositivo."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convierte un batch de tensores NCHW en [-1, 1] a un arreglo NHWC uint8 en [0, 255]."""
    images = images.detach().cpu().clamp(-1, 1)
    images = (images + 1.0) / 2.0 * 255.0
    arr = images.permute(0, 2, 3, 1).round().to(torch.uint8).numpy()
    return arr


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 4, padding: int = 2,
                     bg_value: int = 24) -> None:
    """Guarda un batch de imágenes NCHW en [-1, 1] como una rejilla PNG."""
    arr = tensor_to_uint8(images)
    n, h, w, c = arr.shape
    ncol = nrow
    nrow_grid = int(np.ceil(n / ncol))
    grid = np.full((nrow_grid * h + (nrow_grid + 1) * padding,
                     ncol * w + (ncol + 1) * padding, c), bg_value, dtype=np.uint8)
    for idx in range(n):
        r, col = divmod(idx, ncol)
        y0 = padding + r * (h + padding)
        x0 = padding + col * (w + padding)
        grid[y0:y0 + h, x0:x0 + w] = arr[idx]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


def save_single_image(image: torch.Tensor, path: str | Path) -> None:
    """Guarda una única imagen CHW en [-1, 1] como PNG."""
    arr = tensor_to_uint8(image.unsqueeze(0))[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
