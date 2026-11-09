"""Entrena la Conditional GAN (Fase 2 del Proyecto Final) y la guarda en
`experiments/conditional/checkpoint.pt`.

Uso:
    PYTHONPATH=src python scripts/train_conditional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ganverse.train import ConditionalTrainConfig, train_conditional  # noqa: E402

if __name__ == "__main__":
    cfg = ConditionalTrainConfig(name="conditional", epochs=30)
    run_dir = train_conditional(cfg)
    print(f"Listo. Checkpoint en: {run_dir}")
