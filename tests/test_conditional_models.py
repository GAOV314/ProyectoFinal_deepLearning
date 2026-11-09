import torch

from ganverse.models import ConditionalDiscriminator, ConditionalGenerator

TINY_LATENT = 8
TINY_FM = 8
TINY_EMBED = 4
TINY_BATCH = 4
N_CLASSES = 6


def _labels():
    return torch.randint(0, N_CLASSES, (TINY_BATCH,))


def test_conditional_generator_output_shape_and_range():
    g = ConditionalGenerator(TINY_LATENT, TINY_FM, n_classes=N_CLASSES, embed_dim=TINY_EMBED)
    z = torch.randn(TINY_BATCH, TINY_LATENT)
    out = g(z, _labels())
    assert out.shape == (TINY_BATCH, 3, 64, 64)
    assert out.min().item() >= -1.0001
    assert out.max().item() <= 1.0001


def test_conditional_generator_different_labels_different_output():
    g = ConditionalGenerator(TINY_LATENT, TINY_FM, n_classes=N_CLASSES, embed_dim=TINY_EMBED)
    z = torch.randn(1, TINY_LATENT).repeat(2, 1)
    labels = torch.tensor([0, 1])
    out = g(z, labels)
    assert not torch.allclose(out[0], out[1])


def test_conditional_discriminator_output_shape():
    d = ConditionalDiscriminator(TINY_FM, n_classes=N_CLASSES, normalization="spectral")
    x = torch.randn(TINY_BATCH, 3, 64, 64)
    logits = d(x, _labels())
    assert logits.shape == (TINY_BATCH,)
    assert torch.isfinite(logits).all()


def test_conditional_generator_gradients_flow():
    g = ConditionalGenerator(TINY_LATENT, TINY_FM, n_classes=N_CLASSES, embed_dim=TINY_EMBED)
    z = torch.randn(TINY_BATCH, TINY_LATENT, requires_grad=True)
    out = g(z, _labels())
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
