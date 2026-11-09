"""Servicio de producción: genera un personaje nuevo del universo Glimmerfen,
opcionalmente de un elemento elegido por el usuario (Conditional GAN, Fase 2).

No reentrena nada: carga los pesos ya entrenados desde `experiments/conditional/
checkpoint.pt` (o, si no existe aún, cae de vuelta al Generador incondicional
de `checkpoints/stab_spectral_norm.pt` de la Fase 1) y expone una app web
(Gradio) que cualquier persona ajena al equipo puede usar.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ganverse.dataset import ELEMENT_NAMES  # noqa: E402
from ganverse.models import ConditionalGenerator, Generator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
CONDITIONAL_CKPT = REPO_ROOT / "experiments" / "conditional" / "checkpoint.pt"
BASELINE_CKPT = REPO_ROOT / "checkpoints" / "stab_spectral_norm.pt"
DEVICE = torch.device("cpu")  # servicio público: CPU alcanza, el generador es liviano

USE_CONDITIONAL = CONDITIONAL_CKPT.exists()
OPTIONS = ["Aleatorio"] + ELEMENT_NAMES

if USE_CONDITIONAL:
    _checkpoint = torch.load(CONDITIONAL_CKPT, map_location=DEVICE)
    _cfg = _checkpoint["config"]
    generator = ConditionalGenerator(latent_dim=_cfg["latent_dim"], feature_maps=_cfg["feature_maps"],
                                      n_classes=_cfg["n_classes"], embed_dim=_cfg["embed_dim"])
    generator.load_state_dict(_checkpoint["G"])
else:
    _checkpoint = torch.load(BASELINE_CKPT, map_location=DEVICE)
    _cfg = _checkpoint["config"]
    generator = Generator(latent_dim=_cfg["latent_dim"], feature_maps=_cfg["feature_maps"])
    generator.load_state_dict(_checkpoint["G"])

generator.eval().to(DEVICE)


def generate_character(seed: int, elemento: str) -> "object":
    """Genera un espíritu Glimmerfen nuevo. Si el modelo es condicional y se
    pide un elemento específico, el personaje pertenece a ese elemento."""
    rng = torch.Generator().manual_seed(int(seed))
    z = torch.randn(1, generator.latent_dim, generator=rng, device=DEVICE)
    with torch.no_grad():
        if USE_CONDITIONAL:
            if elemento == "Aleatorio":
                label_idx = torch.randint(0, len(ELEMENT_NAMES), (1,), generator=rng).item()
            else:
                label_idx = ELEMENT_NAMES.index(elemento)
            labels = torch.tensor([label_idx], dtype=torch.long, device=DEVICE)
            image = generator(z, labels)[0]
        else:
            image = generator(z)[0]
    image = (image.clamp(-1, 1) + 1) / 2.0  # [-1, 1] -> [0, 1] para mostrarlo
    return image.permute(1, 2, 0).numpy()


with gr.Blocks(title="Glimmerfen — Generador de personajes") as demo:
    gr.Markdown(
        "# Glimmerfen — Generador de personajes\n"
        + ("Conditional GAN (normalización espectral) — elige el elemento y "
           "obtén un espíritu guardián nuevo de ese elemento."
           if USE_CONDITIONAL else
           "DCGAN (normalización espectral) entrenado sobre un dataset "
           "procedural propio. Cada semilla produce un espíritu nuevo.")
        + "\n\nVer el repositorio del proyecto para la arquitectura, el "
          "entrenamiento y la validación de hipótesis completas."
    )
    with gr.Row():
        seed_input = gr.Number(value=2026, label="Semilla (seed)", precision=0)
        element_input = gr.Dropdown(choices=OPTIONS, value="Aleatorio", label="Elemento",
                                     visible=USE_CONDITIONAL)
        generate_btn = gr.Button("Generar personaje", variant="primary")
    output_image = gr.Image(label="Personaje generado", type="numpy")

    generate_btn.click(fn=generate_character, inputs=[seed_input, element_input], outputs=output_image)
    demo.load(fn=generate_character, inputs=[seed_input, element_input], outputs=output_image)

if __name__ == "__main__":
    # Render (y la mayoría de los hosts gratuitos) asignan el puerto por la
    # variable de entorno PORT y exigen escuchar en 0.0.0.0, no en localhost.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
