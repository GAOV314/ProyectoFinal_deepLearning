"""Funciones de pérdida: BCE no saturante (base) y WGAN-GP (experimento de pérdida).

Referencias: Goodfellow et al. (2014) para la pérdida no saturante;
Arjovsky et al. (2017, WGAN) y Gulrajani et al. (2017, WGAN-GP).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_bce = nn.BCEWithLogitsLoss()


def bce_discriminator_loss(d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor) -> torch.Tensor:
    real_labels = torch.ones_like(d_real_logits)
    fake_labels = torch.zeros_like(d_fake_logits)
    loss_real = _bce(d_real_logits, real_labels)
    loss_fake = _bce(d_fake_logits, fake_labels)
    return loss_real + loss_fake


def bce_generator_loss(d_fake_logits: torch.Tensor) -> torch.Tensor:
    """Pérdida NO saturante: maximiza log(D(G(z))) en vez de minimizar log(1 - D(G(z)))."""
    real_labels = torch.ones_like(d_fake_logits)
    return _bce(d_fake_logits, real_labels)


def gradient_penalty(discriminator, real: torch.Tensor, fake: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    """Penalización de gradiente de WGAN-GP: fuerza ||∇D(x̂)||₂ ≈ 1 en puntos
    interpolados entre reales y falsas, sustituyendo el recorte de pesos de WGAN."""
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    penalty = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return penalty


def wgan_gp_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor, gp: torch.Tensor,
                                lambda_gp: float = 10.0) -> torch.Tensor:
    return d_fake.mean() - d_real.mean() + lambda_gp * gp


def wgan_gp_generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()
