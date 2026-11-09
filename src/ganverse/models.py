"""Arquitectura DCGAN (Radford et al., 2015) para imágenes de 64x64x3.

El Discriminador admite tres modos de normalización para poder ejecutar los
experimentos del proyecto sin duplicar código:

- ``"batch"``    : BatchNorm clásico de DCGAN (configuración base).
- ``"spectral"`` : Normalización espectral en cada Conv2d, sin BatchNorm
                   (experimento de estabilización, sección 2 punto 5).
- ``"none"``     : sin normalización, usado por el crítico WGAN-GP, donde
                   BatchNorm está desaconsejado porque invalida la penalización
                   de gradiente por muestra (Gulrajani et al., 2017).
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import spectral_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import spectral_norm


def weights_init(m: nn.Module) -> None:
    """Inicialización N(0, 0.02) recomendada en el paper original de DCGAN."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """Latente (N, latent_dim, 1, 1) -> imagen (N, 3, 64, 64) en rango [-1, 1]."""

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, img_channels: int = 3):
        super().__init__()
        fm = feature_maps
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fm * 8, 4, 1, 0, bias=False),   # 1x1 -> 4x4
            nn.BatchNorm2d(fm * 8), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 8, fm * 4, 4, 2, 1, bias=False),       # 4x4 -> 8x8
            nn.BatchNorm2d(fm * 4), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 4, fm * 2, 4, 2, 1, bias=False),       # 8x8 -> 16x16
            nn.BatchNorm2d(fm * 2), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 2, fm, 4, 2, 1, bias=False),           # 16x16 -> 32x32
            nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.ConvTranspose2d(fm, img_channels, 4, 2, 1, bias=False),     # 32x32 -> 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(*z.shape, 1, 1)
        return self.net(z)


def _conv_block(in_c: int, out_c: int, normalization: str, is_first: bool) -> list[nn.Module]:
    conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=(normalization != "batch"))
    layers: list[nn.Module] = []
    if normalization == "spectral":
        layers.append(spectral_norm(conv))
    else:
        layers.append(conv)
    if normalization == "batch" and not is_first:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    """Imagen (N, 3, 64, 64) -> logit crudo (N,) de "realismo" (sin sigmoide).

    Se deja la sigmoide fuera del módulo para poder usar ``BCEWithLogitsLoss``
    (BCE no saturante) o el puntaje crudo de WGAN-GP con el mismo forward.
    """

    def __init__(self, feature_maps: int = 64, img_channels: int = 3, normalization: str = "batch"):
        super().__init__()
        assert normalization in {"batch", "spectral", "none"}
        fm = feature_maps
        layers: list[nn.Module] = []
        layers += _conv_block(img_channels, fm, normalization, is_first=True)       # 64 -> 32
        layers += _conv_block(fm, fm * 2, normalization, is_first=False)            # 32 -> 16
        layers += _conv_block(fm * 2, fm * 4, normalization, is_first=False)        # 16 -> 8
        layers += _conv_block(fm * 4, fm * 8, normalization, is_first=False)        # 8 -> 4
        self.features = nn.Sequential(*layers)
        final_conv = nn.Conv2d(fm * 8, 1, 4, 1, 0, bias=True)
        self.final = spectral_norm(final_conv) if normalization == "spectral" else final_conv
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.features(x)
        logit = self.final(feats).view(-1)
        if return_features:
            return logit, feats
        return logit


class ConditionalGenerator(nn.Module):
    """Igual que ``Generator``, pero condicionado a una etiqueta de clase
    (elemento: fuego/agua/tierra/aire/luz/sombra).

    Mejora del Proyecto Final sobre el Generator incondicional del Proyecto 2
    (``PROJECT.md`` sección 13): se embebe la etiqueta y se concatena al
    ruido `z` antes de la primera capa, para poder *pedir* un elemento en
    vez de solo samplear al azar.
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, img_channels: int = 3,
                 n_classes: int = 6, embed_dim: int = 50):
        super().__init__()
        fm = feature_maps
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embed_dim, fm * 8, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(fm * 8), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 8, fm * 4, 4, 2, 1, bias=False),       # 4x4 -> 8x8
            nn.BatchNorm2d(fm * 4), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 4, fm * 2, 4, 2, 1, bias=False),       # 8x8 -> 16x16
            nn.BatchNorm2d(fm * 2), nn.ReLU(True),
            nn.ConvTranspose2d(fm * 2, fm, 4, 2, 1, bias=False),           # 16x16 -> 32x32
            nn.BatchNorm2d(fm), nn.ReLU(True),
            nn.ConvTranspose2d(fm, img_channels, 4, 2, 1, bias=False),     # 32x32 -> 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(*z.shape, 1, 1)
        embed = self.label_embed(labels).view(labels.size(0), -1, 1, 1)
        x = torch.cat([z, embed], dim=1)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """Igual que ``Discriminator`` (normalización espectral, la variante más
    estable del Proyecto 2), pero recibe la etiqueta de elemento proyectada
    como un canal espacial extra concatenado a la imagen (enfoque estándar
    de cGAN condicional por canal, Mirza & Osindero 2014)."""

    def __init__(self, feature_maps: int = 64, img_channels: int = 3, n_classes: int = 6,
                 normalization: str = "spectral", img_size: int = 64):
        super().__init__()
        assert normalization in {"batch", "spectral", "none"}
        self.img_size = img_size
        self.label_embed = nn.Embedding(n_classes, img_size * img_size)
        fm = feature_maps
        layers: list[nn.Module] = []
        layers += _conv_block(img_channels + 1, fm, normalization, is_first=True)     # 64 -> 32
        layers += _conv_block(fm, fm * 2, normalization, is_first=False)              # 32 -> 16
        layers += _conv_block(fm * 2, fm * 4, normalization, is_first=False)          # 16 -> 8
        layers += _conv_block(fm * 4, fm * 8, normalization, is_first=False)          # 8 -> 4
        self.features = nn.Sequential(*layers)
        final_conv = nn.Conv2d(fm * 8, 1, 4, 1, 0, bias=True)
        self.final = spectral_norm(final_conv) if normalization == "spectral" else final_conv
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_embed(labels).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([x, label_map], dim=1)
        feats = self.features(x)
        return self.final(feats).view(-1)
