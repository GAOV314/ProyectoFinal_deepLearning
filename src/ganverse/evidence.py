"""Las tres evidencias obligatorias de la sección 3 del enunciado:

1. ``fixed_noise_evolution``  -> rejilla de ruido fijo a lo largo del entrenamiento.
2. ``plot_loss_curves``       -> curvas de pérdida de G y D, con lectura automática
                                  de señales de alarma (D -> 0, colapso, etc.).
3. ``nearest_neighbor_test``  -> vecino más cercano en el set de entrenamiento para
                                  cada personaje final, por distancia de píxeles y
                                  por distancia en el espacio de features del
                                  discriminador (CNN propia, entrenada en el dominio).
"""
from __future__ import annotations

from pathlib import Path

# No se fuerza el backend de matplotlib aquí: este módulo lo usan tanto
# scripts headless (que ya configuran "Agg" ellos mismos antes de importarlo)
# como el notebook interactivo, que necesita su backend inline para mostrar
# las figuras — forzar Agg a nivel de librería rompería ese inline silenciosamente.
import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import load_json, save_json, tensor_to_uint8


def plot_loss_curves(loss_history_path: str | Path, out_path: str | Path, title: str) -> dict:
    history = load_json(loss_history_path)
    its = [h["iter"] for h in history]
    d = np.array([h["d_loss"] for h in history])
    g = np.array([h["g_loss"] for h in history])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(its, d, label="Pérdida D (discriminador/crítico)", color="#d1495b", linewidth=1.2)
    ax.plot(its, g, label="Pérdida G (generador)", color="#2e86ab", linewidth=1.2)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Pérdida")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    tail = max(1, len(d) // 5)  # último 20% del entrenamiento
    d_tail, g_tail = d[-tail:], g[-tail:]
    summary = {
        "d_loss_final_mean": float(d_tail.mean()),
        "d_loss_final_std": float(d_tail.std()),
        "g_loss_final_mean": float(g_tail.mean()),
        "g_loss_final_std": float(g_tail.std()),
        "n_iterations": len(d),
    }
    return summary


def fixed_noise_evolution(run_dir: str | Path, out_path: str | Path, n_frames: int = 5) -> list[Path]:
    """Combina snapshots guardados durante el entrenamiento (mismo ruido fijo en
    todos) en una sola figura, para mostrar cómo evolucionó el generador."""
    run_dir = Path(run_dir)
    snaps = sorted((run_dir / "snapshots").glob("epoch_*.png"))
    if not snaps:
        raise FileNotFoundError(f"No hay snapshots en {run_dir / 'snapshots'}")
    idxs = np.linspace(0, len(snaps) - 1, min(n_frames, len(snaps))).round().astype(int)
    chosen = [snaps[i] for i in idxs]

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.2 * len(chosen), 3.4))
    if len(chosen) == 1:
        axes = [axes]
    for ax, path in zip(axes, chosen):
        img = plt.imread(path)
        ax.imshow(img)
        epoch = int(path.stem.split("_")[1])
        ax.set_title(f"época {epoch}")
        ax.axis("off")
    fig.suptitle("Rejilla de ruido fijo a lo largo del entrenamiento (mismos z en cada panel)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return chosen


def _pixel_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


@torch.no_grad()
def _feature_distance(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    a = feat_a.flatten()
    b = feat_b.flatten()
    return float(torch.nn.functional.mse_loss(a, b).item())


@torch.no_grad()
def nearest_neighbor_test(gallery_images: torch.Tensor, train_images_uint8: np.ndarray,
                           discriminator=None, device: torch.device | None = None) -> dict:
    """Para cada imagen final, busca su vecino más cercano en el set de
    entrenamiento por distancia de píxeles y (si se provee un discriminador) por
    distancia en su espacio de features convolucional. Devuelve, por personaje,
    el índice del vecino y ambas distancias, para poder juzgar honestamente si
    el personaje es "nuevo" o casi idéntico a algo ya visto.
    """
    gallery_uint8 = tensor_to_uint8(gallery_images)
    results = []
    feats_train = None
    if discriminator is not None:
        device = device or next(discriminator.parameters()).device
        train_t = torch.from_numpy(train_images_uint8).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        feats_chunks = []
        for i in range(0, len(train_t), 256):
            chunk = train_t[i:i + 256].to(device)
            _, f = discriminator(chunk, return_features=True)
            feats_chunks.append(f.cpu())
        feats_train = torch.cat(feats_chunks, dim=0)

    for g_idx in range(len(gallery_uint8)):
        g_img = gallery_uint8[g_idx]
        pixel_dists = np.array([_pixel_distance(g_img, train_images_uint8[j])
                                 for j in range(len(train_images_uint8))])
        nn_pixel_idx = int(pixel_dists.argmin())

        entry = {
            "character_index": g_idx,
            "nearest_train_index_pixel": nn_pixel_idx,
            "pixel_mse": float(pixel_dists[nn_pixel_idx]),
        }

        if discriminator is not None:
            g_tensor = gallery_images[g_idx:g_idx + 1].to(device)
            _, g_feat = discriminator(g_tensor, return_features=True)
            g_feat = g_feat.cpu()
            feat_dists = torch.stack([
                torch.nn.functional.mse_loss(g_feat.flatten(), feats_train[j].flatten())
                for j in range(len(feats_train))
            ])
            nn_feat_idx = int(feat_dists.argmin().item())
            entry["nearest_train_index_feature"] = nn_feat_idx
            entry["feature_mse"] = float(feat_dists[nn_feat_idx].item())

        results.append(entry)

    # umbral heurístico de "casi idéntico": MSE de píxel muy bajo relativo al rango [0,255]
    threshold = 60.0
    flagged = [r["character_index"] for r in results if r["pixel_mse"] < threshold]
    summary = {"threshold_pixel_mse": threshold, "flagged_as_too_similar": flagged, "results": results}
    return summary
