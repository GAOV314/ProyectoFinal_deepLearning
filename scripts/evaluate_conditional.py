"""Valida la hipótesis de la Fase 2 (sección 5 del README):

Hipótesis: condicionar la GAN al elemento permite pedir un elemento
específico sin perder diversidad, con una tasa de acierto muy por encima
de la línea base (modelo incondicional, ~1/6 por azar).

Métrica: de N personajes generados pidiendo el elemento X, qué porcentaje
tiene el elemento X como dominante según el clasificador de color ya
existente (`gallery._dominant_element`, un evaluador automático externo al
modelo, no parte de su entrenamiento).

Criterio de éxito definido antes de entrenar: >=80% de acierto por elemento.

Uso:
    PYTHONPATH=src python scripts/evaluate_conditional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from ganverse.dataset import ELEMENT_NAMES
from ganverse.gallery import _dominant_element
from ganverse.models import ConditionalGenerator, Generator
from ganverse.utils import save_json, tensor_to_uint8

REPO_ROOT = Path(__file__).resolve().parents[1]
N_PER_CLASS = 100
SEED = 123


@torch.no_grad()
def evaluate_conditional(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    G = ConditionalGenerator(cfg["latent_dim"], cfg["feature_maps"],
                              n_classes=cfg["n_classes"], embed_dim=cfg["embed_dim"])
    G.load_state_dict(ckpt["G"])
    G.eval()

    rng = torch.Generator().manual_seed(SEED)
    per_class_hits = {}
    confusion = {name: {n2: 0 for n2 in ELEMENT_NAMES} for name in ELEMENT_NAMES}
    for class_idx, element in enumerate(ELEMENT_NAMES):
        z = torch.randn(N_PER_CLASS, cfg["latent_dim"], generator=rng)
        labels = torch.full((N_PER_CLASS,), class_idx, dtype=torch.long)
        imgs = G(z, labels)
        arrs = tensor_to_uint8(imgs)
        predicted = [_dominant_element(arrs[i]) for i in range(N_PER_CLASS)]
        hits = sum(1 for p in predicted if p == element)
        per_class_hits[element] = hits / N_PER_CLASS
        for p in predicted:
            confusion[element][p] += 1

    overall = sum(per_class_hits.values()) / len(per_class_hits)
    return {
        "n_per_class": N_PER_CLASS,
        "seed": SEED,
        "hit_rate_per_element": per_class_hits,
        "overall_hit_rate": overall,
        "confusion_matrix": confusion,
    }


@torch.no_grad()
def evaluate_baseline(checkpoint_path: Path) -> dict:
    """Línea base: el Generador incondicional (`stab_spectral_norm`). No
    puede pedir elemento, así que se mide la tasa de acierto *esperable* si
    se le pide un elemento y se acepta la primera muestra generada: qué
    fracción de muestras al azar cae en cada elemento."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    G = Generator(cfg["latent_dim"], cfg["feature_maps"])
    G.load_state_dict(ckpt["G"])
    G.eval()

    rng = torch.Generator().manual_seed(SEED)
    n = N_PER_CLASS * len(ELEMENT_NAMES)
    z = torch.randn(n, cfg["latent_dim"], generator=rng)
    imgs = G(z)
    arrs = tensor_to_uint8(imgs)
    predicted = [_dominant_element(arrs[i]) for i in range(n)]
    counts = {name: predicted.count(name) / n for name in ELEMENT_NAMES}
    return {"n_samples": n, "seed": SEED, "natural_rate_per_element": counts,
            "overall_hit_rate_if_requested": sum(counts.values()) / len(counts)}


if __name__ == "__main__":
    cond_ckpt = REPO_ROOT / "experiments" / "conditional" / "checkpoint.pt"
    base_ckpt = REPO_ROOT / "checkpoints" / "stab_spectral_norm.pt"

    results = {
        "conditional": evaluate_conditional(cond_ckpt),
        "baseline_unconditional": evaluate_baseline(base_ckpt),
        "success_criterion": "hit_rate >= 0.80 por elemento, definido antes de entrenar",
    }
    out_path = REPO_ROOT / "experiments" / "conditional" / "evaluation.json"
    save_json(results, out_path)

    print("=== Conditional GAN: tasa de acierto por elemento ===")
    for elem, rate in results["conditional"]["hit_rate_per_element"].items():
        print(f"  {elem:8s}: {rate:.1%}")
    print(f"  overall : {results['conditional']['overall_hit_rate']:.1%}")
    print()
    print("=== Línea base (incondicional): tasa natural por elemento ===")
    for elem, rate in results["baseline_unconditional"]["natural_rate_per_element"].items():
        print(f"  {elem:8s}: {rate:.1%}")
    print(f"\nGuardado en: {out_path}")
