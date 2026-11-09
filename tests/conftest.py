"""Fixtures compartidas: configuraciones diminutas para que las pruebas corran
en segundos (no se entrena de verdad, solo se validan formas y matemáticas)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TINY_LATENT = 8
TINY_FM = 8
TINY_BATCH = 4
IMG_SIZE = 64


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_dataset_array():
    from ganverse.dataset import generate_dataset_array
    arr, params = generate_dataset_array(n=6, seed=1)
    return arr, params


@pytest.fixture
def tiny_generator():
    from ganverse.models import Generator
    return Generator(latent_dim=TINY_LATENT, feature_maps=TINY_FM)


@pytest.fixture(params=["batch", "spectral", "none"])
def tiny_discriminator(request):
    from ganverse.models import Discriminator
    return Discriminator(feature_maps=TINY_FM, normalization=request.param)
