import torch

from ganverse.models import Discriminator, Generator

TINY_LATENT = 8
TINY_FM = 8
TINY_BATCH = 4


def test_generator_output_shape_and_range(tiny_generator):
    z = torch.randn(TINY_BATCH, TINY_LATENT)
    out = tiny_generator(z)
    assert out.shape == (TINY_BATCH, 3, 64, 64)
    assert out.min().item() >= -1.0001
    assert out.max().item() <= 1.0001


def test_generator_accepts_4d_latent(tiny_generator):
    z = torch.randn(TINY_BATCH, TINY_LATENT, 1, 1)
    out = tiny_generator(z)
    assert out.shape == (TINY_BATCH, 3, 64, 64)


def test_discriminator_output_shape_all_normalizations(tiny_discriminator):
    x = torch.randn(TINY_BATCH, 3, 64, 64)
    logits = tiny_discriminator(x)
    assert logits.shape == (TINY_BATCH,)
    assert torch.isfinite(logits).all()


def test_discriminator_return_features(tiny_discriminator):
    x = torch.randn(TINY_BATCH, 3, 64, 64)
    logits, feats = tiny_discriminator(x, return_features=True)
    assert logits.shape == (TINY_BATCH,)
    assert feats.dim() == 4
    assert feats.shape[0] == TINY_BATCH


def test_discriminator_spectral_norm_has_no_batchnorm():
    d = Discriminator(feature_maps=TINY_FM, normalization="spectral")
    names = [type(m).__name__ for m in d.modules()]
    assert "BatchNorm2d" not in names


def test_discriminator_batch_mode_uses_batchnorm():
    d = Discriminator(feature_maps=TINY_FM, normalization="batch")
    names = [type(m).__name__ for m in d.modules()]
    assert "BatchNorm2d" in names


def test_generator_gradients_flow():
    g = Generator(latent_dim=TINY_LATENT, feature_maps=TINY_FM)
    z = torch.randn(2, TINY_LATENT, requires_grad=True)
    out = g(z)
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
