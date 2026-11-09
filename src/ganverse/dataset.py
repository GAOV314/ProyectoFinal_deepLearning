"""Generador procedural del universo **Glimmerfen** y Dataset de PyTorch.

Universo visual (ver PROJECT.md sección 1): pequeños "espíritus guardianes" de
bosque, con cuerpo tipo blob, ojos grandes y apéndices elementales (cuernos,
alas, orejas, cola, antena). Cada espíritu pertenece a uno de seis elementos
(fuego, agua, tierra, aire, luz, sombra), lo que fija su paleta de color y
mantiene un estilo visual consistente en todo el dataset.

El dataset se genera por código (no se descarga de internet ni usa rostros o
personajes de terceros), lo que resuelve de raíz el requisito de "origen,
licencia o permiso de uso": el equipo es autor y titular de cada imagen.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMG_SIZE = 64
SUPERSAMPLE = 4  # se dibuja a 256px y se reduce a 64px (antialiasing tipo "pixel art suave")

# Paleta por elemento: (color de piel/cuerpo, color secundario/panza, color de acento/ojos-brillo)
ELEMENTS: dict[str, dict] = {
    "fuego":  {"body": (224, 84, 56),  "belly": (255, 176, 86),  "accent": (255, 221, 120)},
    "agua":   {"body": (58, 120, 199), "belly": (138, 208, 235), "accent": (217, 245, 255)},
    "tierra": {"body": (123, 92, 58),  "belly": (186, 150, 98),  "accent": (168, 201, 104)},
    "aire":   {"body": (176, 196, 222),"belly": (228, 236, 246), "accent": (255, 255, 255)},
    "luz":    {"body": (247, 214, 110),"belly": (255, 244, 196), "accent": (255, 255, 255)},
    "sombra": {"body": (78, 63, 112),  "belly": (134, 110, 168), "accent": (201, 170, 255)},
}
ELEMENT_NAMES = list(ELEMENTS.keys())
BACKGROUND_RGB = (30, 33, 41)  # fondo plano consistente en todo el dataset


def _lighten(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in rgb)


def _darken(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(c * (1 - amount)) for c in rgb)


def _box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Caja de coordenadas para PIL, garantizando x0<=x1 y y0<=y1 (PIL exige ese
    orden; con `sign in (-1, 1)` los apéndices espejados pueden invertirlo)."""
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


@dataclass
class SpriteParams:
    """Parámetros muestreados para un espíritu individual (controla la variedad)."""
    element: str
    body_w: float
    body_h: float
    eye_count: int
    eye_size: float
    eye_spread: float
    appendage: str  # "cuernos" | "alas" | "orejas" | "cola" | "antena" | "ninguno"
    pattern: str    # "liso" | "manchas" | "franjas"
    seed: int = 0


def sample_params(rng: random.Random) -> SpriteParams:
    return SpriteParams(
        element=rng.choice(ELEMENT_NAMES),
        body_w=rng.uniform(0.52, 0.72),
        body_h=rng.uniform(0.50, 0.68),
        eye_count=rng.choice([1, 2, 2, 2]),  # casi siempre 2 ojos, a veces 1 (cíclope)
        eye_size=rng.uniform(0.10, 0.16),
        eye_spread=rng.uniform(0.16, 0.26),
        appendage=rng.choice(["cuernos", "alas", "orejas", "cola", "antena", "ninguno"]),
        pattern=rng.choice(["liso", "liso", "manchas", "franjas"]),
    )


def render_sprite(params: SpriteParams, size: int = IMG_SIZE, supersample: int = SUPERSAMPLE) -> Image.Image:
    """Dibuja un espíritu Glimmerfen en una imagen RGB `size`x`size`."""
    s = size * supersample
    img = Image.new("RGB", (s, s), BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)
    cx, cy = s / 2, s * 0.56
    palette = ELEMENTS[params.element]
    body_rgb, belly_rgb, accent_rgb = palette["body"], palette["belly"], palette["accent"]

    bw, bh = params.body_w * s, params.body_h * s

    # --- apéndices detrás del cuerpo ---
    if params.appendage == "alas":
        wing_rgb = _lighten(body_rgb, 0.15)
        for sign in (-1, 1):
            draw.polygon([
                (cx + sign * bw * 0.05, cy - bh * 0.1),
                (cx + sign * bw * 0.85, cy - bh * 0.55),
                (cx + sign * bw * 0.75, cy + bh * 0.05),
                (cx + sign * bw * 0.35, cy + bh * 0.2),
            ], fill=wing_rgb)
    elif params.appendage == "cola":
        tail_rgb = body_rgb
        pts = [(cx + bw * 0.35, cy + bh * 0.25)]
        for i in range(1, 6):
            pts.append((cx + bw * (0.35 + 0.12 * i), cy + bh * (0.25 - 0.08 * i) + 10 * math.sin(i)))
        draw.line(pts, fill=tail_rgb, width=int(s * 0.05), joint="curve")
        draw.ellipse([pts[-1][0] - s * 0.04, pts[-1][1] - s * 0.04,
                      pts[-1][0] + s * 0.04, pts[-1][1] + s * 0.04], fill=tail_rgb)

    # --- cuerpo (blob = elipse con "panza") ---
    body_box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    draw.ellipse(body_box, fill=body_rgb)
    belly_box = [cx - bw * 0.32, cy - bh * 0.05, cx + bw * 0.32, cy + bh * 0.46]
    draw.ellipse(belly_box, fill=belly_rgb)

    # --- patrón ---
    if params.pattern == "manchas":
        rng = random.Random(params.seed + 777)
        for _ in range(rng.randint(3, 6)):
            px = cx + rng.uniform(-0.38, 0.38) * bw
            py = cy + rng.uniform(-0.3, 0.35) * bh
            r = rng.uniform(0.04, 0.08) * bw
            draw.ellipse([px - r, py - r, px + r, py + r], fill=_darken(body_rgb, 0.25))
    elif params.pattern == "franjas":
        dark = _darken(body_rgb, 0.3)
        for i, frac in enumerate((-0.25, 0.0, 0.25)):
            y = cy + frac * bh
            draw.line([(cx - bw * 0.45, y), (cx + bw * 0.45, y)], fill=dark, width=int(s * 0.025))

    # --- apéndices delante del cuerpo (parte superior) ---
    if params.appendage == "cuernos":
        for sign in (-1, 1):
            draw.polygon([
                (cx + sign * bw * 0.18, cy - bh * 0.42),
                (cx + sign * bw * 0.30, cy - bh * 0.85),
                (cx + sign * bw * 0.08, cy - bh * 0.48),
            ], fill=accent_rgb)
    elif params.appendage == "orejas":
        for sign in (-1, 1):
            draw.ellipse(_box(cx + sign * bw * 0.32, cy - bh * 0.65,
                               cx + sign * bw * 0.52, cy - bh * 0.25), fill=body_rgb)
            draw.ellipse(_box(cx + sign * bw * 0.36, cy - bh * 0.58,
                               cx + sign * bw * 0.46, cy - bh * 0.34), fill=belly_rgb)
    elif params.appendage == "antena":
        draw.line([(cx, cy - bh * 0.48), (cx, cy - bh * 0.85)], fill=body_rgb, width=int(s * 0.02))
        r = s * 0.035
        draw.ellipse([cx - r, cy - bh * 0.85 - r, cx + r, cy - bh * 0.85 + r], fill=accent_rgb)

    # --- ojos ---
    eye_r = params.eye_size * bw
    if params.eye_count == 1:
        offsets = [0.0]
    else:
        offsets = [-params.eye_spread, params.eye_spread]
    for off in offsets:
        ex, ey = cx + off * bw, cy - bh * 0.12
        draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r], fill=(250, 250, 250))
        pr = eye_r * 0.5
        draw.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=(20, 20, 24))
        hr = eye_r * 0.18
        draw.ellipse([ex - pr * 0.3 - hr, ey - pr * 0.3 - hr, ex - pr * 0.3 + hr, ey - pr * 0.3 + hr],
                     fill=(255, 255, 255))

    # --- sombreado plano (luz superior-izquierda) ---
    highlight = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    hd.ellipse([cx - bw * 0.42, cy - bh * 0.48, cx + bw * 0.05, cy - bh * 0.02], fill=(255, 255, 255, 60))
    img = Image.alpha_composite(img.convert("RGBA"), highlight).convert("RGB")
    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse([cx - bw * 0.1, cy + bh * 0.15, cx + bw * 0.48, cy + bh * 0.52], fill=(0, 0, 0, 45))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")

    img = img.resize((size, size), Image.LANCZOS)
    return img


def generate_dataset_array(n: int, seed: int = 42) -> tuple[np.ndarray, list[SpriteParams]]:
    """Genera `n` espíritus y los devuelve como arreglo uint8 (N, H, W, 3)."""
    rng = random.Random(seed)
    arr = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    all_params = []
    for i in range(n):
        params = sample_params(rng)
        params.seed = seed * 1_000_003 + i
        img = render_sprite(params)
        arr[i] = np.array(img)
        all_params.append(params)
    return arr, all_params


def save_dataset(n: int, out_dir: str | Path, seed: int = 42, n_png_samples: int = 24) -> Path:
    """Genera el dataset completo, lo guarda como .npy (rápido de cargar) y exporta
    una muestra en PNG para documentación/inspección visual.

    También persiste las etiquetas de elemento (``labels_elemento.npy``, un
    entero 0-5 por imagen según `ELEMENT_NAMES`) — necesarias para la
    Conditional GAN de la Fase 2 del Proyecto Final. La generación es
    determinista por semilla, así que estas etiquetas corresponden
    exactamente a las mismas imágenes que ya se usaron en el Proyecto 2."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr, all_params = generate_dataset_array(n, seed=seed)
    np.save(out_dir / "glimmerfen_64x64.npy", arr)
    labels = np.array([ELEMENT_NAMES.index(p.element) for p in all_params], dtype=np.int64)
    np.save(out_dir / "labels_elemento.npy", labels)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for i in range(min(n_png_samples, n)):
        Image.fromarray(arr[i]).save(samples_dir / f"sample_{i:03d}.png")

    meta = {
        "n_images": n,
        "resolution": IMG_SIZE,
        "channels": 3,
        "seed": seed,
        "origin": "Generado proceduralmente con PIL por el equipo (src/ganverse/dataset.py). "
                  "No contiene rostros reales ni personajes de terceros.",
        "license": "CC0 / propiedad del equipo (Cindy Gualim y Gadiel Ocaña)",
        "universe": "Glimmerfen: espiritus guardianes elementales de bosque",
        "elements": ELEMENT_NAMES,
    }
    from .utils import save_json
    save_json(meta, out_dir / "metadata.json")
    return out_dir / "glimmerfen_64x64.npy"


class GlimmerfenDataset(Dataset):
    """Dataset de PyTorch sobre el arreglo .npy generado por `save_dataset`.

    Normaliza a [-1, 1] (rango de salida de Tanh del generador) y aplica
    aumento de datos por flip horizontal (los espíritus son simétricos por diseño,
    así que el flip no introduce artefactos y duplica la variedad efectiva).
    """

    def __init__(self, npy_path: str | Path, augment: bool = True):
        self.data = np.load(npy_path)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.data[idx]
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :]
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
        tensor = tensor.permute(2, 0, 1) / 127.5 - 1.0
        return tensor


class ConditionalGlimmerfenDataset(Dataset):
    """Igual que ``GlimmerfenDataset``, pero también devuelve la etiqueta de
    elemento (entero 0-5) de cada imagen, para entrenar la Conditional GAN."""

    def __init__(self, npy_path: str | Path, labels_path: str | Path, augment: bool = True):
        self.data = np.load(npy_path)
        self.labels = np.load(labels_path)
        assert len(self.data) == len(self.labels)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.data[idx]
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :]
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
        tensor = tensor.permute(2, 0, 1) / 127.5 - 1.0
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return tensor, label
