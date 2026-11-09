"""Construcción de la galería final: genera candidatos, declara la tasa de
selección, elige 10 personajes diversos y les asigna nombre + clase/rol
(bonificación de la sección 7) de forma determinista a partir de la imagen.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .dataset import ELEMENTS
from .utils import save_json, save_single_image, tensor_to_uint8

# Nombre + clase/rol por elemento, coherente con el universo Glimmerfen
# (ver PROJECT.md sección "Universo"). 10 opciones por elemento = 60 nombres;
# de sobra para que una galería de 10 (incluso si cayera toda en un solo
# elemento) nunca agote el pool y tenga que usar el nombre de respaldo.
NAME_POOL: dict[str, list[tuple[str, str]]] = {
    "fuego": [
        ("Pyra", "Centinela de Fuego"), ("Ascua", "Guardián Forjallamas"),
        ("Rubor", "Explorador de Cenizas"), ("Ember", "Vigía del Volcán"),
        ("Chispa", "Aprendiz de la Forja"), ("Magma", "Guardián de la Caldera"),
        ("Ignis", "Centinela del Cráter"), ("Brasa", "Vigía de la Hoguera"),
        ("Solfira", "Explorador Incandescente"), ("Carbon", "Guardián de Rescoldos"),
    ],
    "agua": [
        ("Nerea", "Guardiana de Marea"), ("Gota", "Centinela del Riachuelo"),
        ("Perla", "Vigía de Marejada"), ("Caudal", "Explorador de Lagunas"),
        ("Marina", "Guardiana del Arrecife"), ("Brumax", "Centinela de Niebla"),
        ("Coral", "Vigía de Profundidad"), ("Rocío", "Explorador de Manantial"),
        ("Oleaje", "Guardián de la Corriente"), ("Lirio", "Centinela del Estanque"),
    ],
    "tierra": [
        ("Terro", "Guardián de Raíces"), ("Musgo", "Centinela del Sotobosque"),
        ("Grava", "Vigía de Cavernas"), ("Humus", "Explorador de Madriguera"),
        ("Roble", "Guardián del Bosque Viejo"), ("Arcilla", "Centinela de Barranco"),
        ("Pedregal", "Vigía de Montaña"), ("Semilla", "Explorador de Vivero"),
        ("Granito", "Guardián de Cantera"), ("Hiedra", "Centinela de Ruinas"),
    ],
    "aire": [
        ("Brisa", "Centinela del Viento"), ("Pluma", "Guardián de las Alturas"),
        ("Céfiro", "Vigía de Tormenta"), ("Nimbo", "Explorador de Nubes"),
        ("Ráfaga", "Guardián del Risco"), ("Aura", "Centinela de Cumbre"),
        ("Torbellino", "Vigía del Vendaval"), ("Pluma Gris", "Explorador de Corriente"),
        ("Eolo", "Guardián del Horizonte"), ("Zéfira", "Centinela de Planeo"),
    ],
    "luz": [
        ("Lumen", "Guardián del Amanecer"), ("Destello", "Centinela Radiante"),
        ("Aurea", "Vigía del Claro"), ("Halo", "Explorador de Luciérnagas"),
        ("Radian", "Guardián del Mediodía"), ("Fulgor", "Centinela del Faro"),
        ("Dorado", "Vigía del Trigal"), ("Claridad", "Explorador de Vitrales"),
        ("Solaris", "Guardián del Cenit"), ("Brillantina", "Centinela de Estrellas"),
    ],
    "sombra": [
        ("Umbra", "Centinela Nocturno"), ("Eclipse", "Guardián del Ocaso"),
        ("Penumbra", "Vigía de la Bruma"), ("Noctis", "Explorador de Sombras"),
        ("Grisalla", "Guardián de la Cripta"), ("Tiniebla", "Centinela del Pozo"),
        ("Sigilo", "Vigía Silencioso"), ("Medianoche", "Explorador del Abismo"),
        ("Obsidiana", "Guardián de la Caverna Oscura"), ("Fantasma", "Centinela de Vigilia"),
    ],
}


def _dominant_element(img_uint8: np.ndarray) -> str:
    """Clasifica el elemento comparando el color promedio del cuerpo (región
    central) contra las paletas definidas en dataset.ELEMENTS."""
    h, w, _ = img_uint8.shape
    crop = img_uint8[int(h * 0.3):int(h * 0.8), int(w * 0.25):int(w * 0.75)]
    mean_color = crop.reshape(-1, 3).mean(axis=0)
    best_elem, best_dist = None, float("inf")
    for name, palette in ELEMENTS.items():
        dist = np.linalg.norm(mean_color - np.array(palette["body"]))
        if dist < best_dist:
            best_dist, best_elem = dist, name
    return best_elem


@torch.no_grad()
def generate_candidates(generator, n: int, latent_dim: int, seed: int,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Genera `n` candidatos con ruido reproducible (semilla declarada)."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, latent_dim, generator=gen)
    generator.eval()
    images = generator(z.to(device)).cpu()
    return images, z


@torch.no_grad()
def score_candidates(discriminator, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    discriminator.eval()
    scores = discriminator(images.to(device)).cpu()
    return scores


def select_diverse_top_k(images: torch.Tensor, scores: torch.Tensor, k: int = 10,
                          quality_pool_frac: float = 0.6) -> list[int]:
    """Selección por calidad + diversidad real ("farthest-point"/k-center):

    1. Se descarta el 40% de candidatos con peor puntaje de realismo (ruido,
       artefactos claros), quedando un `quality_pool` de buena calidad.
    2. Dentro de ese pool se corre una búsqueda voraz de "punto más lejano":
       se arranca con el candidato de mayor puntaje y, en cada paso, se añade
       el candidato cuya distancia en píxeles al MÁS CERCANO de los ya
       elegidos sea máxima. Esto maximiza la diversidad mutua del conjunto
       final en vez de solo pedir que supere un umbral fijo frente a los
       elegidos anteriores (criterio más débil: dejaba pasar variaciones
       casi idénticas del mismo personaje, prohibidas en la sección 1).
    """
    n = len(scores)
    pool_size = max(k, int(n * quality_pool_frac))
    pool = torch.argsort(scores, descending=True)[:pool_size].tolist()
    arr = tensor_to_uint8(images).astype(np.float32)

    selected = [pool[0]]
    remaining = pool[1:]
    while len(selected) < k and remaining:
        best_idx, best_min_dist = None, -1.0
        for idx in remaining:
            min_dist = min(float(np.mean((arr[idx] - arr[s]) ** 2)) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist, best_idx = min_dist, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected[:k]


def build_gallery(generator, discriminator, out_dir: str | Path, n_candidates: int = 200,
                   k: int = 10, latent_dim: int = 100, seed: int = 2026,
                   device: torch.device | None = None) -> dict:
    device = device or next(generator.parameters()).device
    images, z = generate_candidates(generator, n_candidates, latent_dim, seed, device)
    scores = score_candidates(discriminator, images, device)
    selected = select_diverse_top_k(images, scores, k=k)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("char_*.png"):  # evita mezclar con una selección anterior
        old_file.unlink()
    used_names: set[str] = set()
    rows = []
    for rank, idx in enumerate(selected):
        img = images[idx]
        arr = tensor_to_uint8(img.unsqueeze(0))[0]
        element = _dominant_element(arr)
        for name, role in NAME_POOL[element]:
            if name not in used_names:
                used_names.add(name)
                break
        else:
            name, role = f"{element.capitalize()}-{rank}", "Guardián sin clasificar"

        fname = f"char_{rank + 1:02d}_{name.lower()}.png"
        save_single_image(img, out_dir / fname)
        rows.append({
            "rank": rank + 1,
            "file": fname,
            "name": name,
            "role": role,
            "element": element,
            "candidate_index": idx,
            "master_seed": seed,
            "discriminator_score": float(scores[idx].item()),
        })

    torch.save(torch.stack([z[i] for i in selected]), out_dir / "z_vectors.pt")
    with open(out_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    selection_info = {
        "n_candidates_generated": n_candidates,
        "n_selected": k,
        "selection_rate": f"{k} de {n_candidates}",
        "selection_criterion": (
            f"Se generaron {n_candidates} imágenes con ruido reproducible (semilla "
            f"{seed}). Se descartó el 40% de peor puntaje de realismo del "
            "discriminador entrenado (filtro de calidad) y, sobre el resto, se "
            "corrió una búsqueda voraz de 'punto más lejano' (farthest-point / "
            "k-center) en distancia de píxeles: se parte del candidato de mayor "
            "puntaje y en cada paso se agrega el candidato cuya distancia al más "
            "cercano de los ya elegidos es máxima, maximizando la diversidad "
            "mutua del conjunto final y evitando variaciones casi idénticas del "
            "mismo personaje."
        ),
        "master_seed": seed,
        "rows": rows,
    }
    save_json(selection_info, out_dir / "selection_info.json")
    return selection_info
