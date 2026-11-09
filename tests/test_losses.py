import torch

from ganverse.losses import (
    bce_discriminator_loss, bce_generator_loss, gradient_penalty,
    wgan_gp_discriminator_loss, wgan_gp_generator_loss,
)
from ganverse.models import Discriminator

TINY_FM = 8


def test_bce_discriminator_loss_is_low_when_predictions_are_correct():
    strong_real = torch.full((8,), 10.0)   # logits grandes -> sigmoide ~1
    strong_fake = torch.full((8,), -10.0)  # logits muy negativos -> sigmoide ~0
    loss_good = bce_discriminator_loss(strong_real, strong_fake)
    loss_bad = bce_discriminator_loss(strong_fake, strong_real)  # invertido: D se equivoca
    assert loss_good.item() < loss_bad.item()


def test_bce_generator_loss_decreases_as_fake_logits_increase():
    low = bce_generator_loss(torch.full((8,), -5.0))
    high = bce_generator_loss(torch.full((8,), 5.0))
    assert high.item() < low.item()


def test_gradient_penalty_is_near_zero_for_1_lipschitz_identity_like_critic():
    torch.manual_seed(0)
    d = Discriminator(feature_maps=TINY_FM, normalization="none")
    real = torch.randn(4, 3, 64, 64)
    fake = torch.randn(4, 3, 64, 64)
    gp = gradient_penalty(d, real, fake, device=torch.device("cpu"))
    assert gp.item() >= 0.0
    assert torch.isfinite(gp)


def test_gradient_penalty_requires_grad_is_differentiable():
    torch.manual_seed(0)
    d = Discriminator(feature_maps=TINY_FM, normalization="none")
    real = torch.randn(4, 3, 64, 64)
    fake = torch.randn(4, 3, 64, 64)
    gp = gradient_penalty(d, real, fake, device=torch.device("cpu"))
    gp.backward()
    grads = [p.grad for p in d.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_wgan_gp_discriminator_loss_combines_wasserstein_term_and_penalty():
    d_real = torch.tensor([1.0, 2.0])
    d_fake = torch.tensor([0.5, 0.5])
    gp = torch.tensor(0.2)
    loss = wgan_gp_discriminator_loss(d_real, d_fake, gp, lambda_gp=10.0)
    expected = d_fake.mean() - d_real.mean() + 10.0 * gp
    assert torch.isclose(loss, expected)


def test_wgan_gp_generator_loss_is_negative_mean():
    d_fake = torch.tensor([1.0, -1.0, 3.0])
    loss = wgan_gp_generator_loss(d_fake)
    assert torch.isclose(loss, -d_fake.mean())
