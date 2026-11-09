import json

import torch

from ganverse.dataset import save_dataset
from ganverse.train import TrainConfig, train


def _tiny_cfg(name: str, loss_type: str, normalization: str) -> TrainConfig:
    return TrainConfig(
        name=name, loss_type=loss_type, normalization=normalization,
        epochs=1, batch_size=4, latent_dim=8, feature_maps=8,
        n_critic=2, fixed_noise_n=4, snapshot_every=1, seed=0,
    )


def _run_tiny(tmp_path, cfg):
    data_dir = tmp_path / "data"
    save_dataset(n=12, out_dir=data_dir, seed=1, n_png_samples=1)
    run_dir = train(
        cfg,
        data_npy=data_dir / "glimmerfen_64x64.npy",
        device=torch.device("cpu"),
        verbose=False,
        experiments_dir=tmp_path / "experiments",
    )
    return run_dir


def test_bce_training_step_runs_without_nan(tmp_path):
    cfg = _tiny_cfg("baseline_test", "bce", "batch")
    run_dir = _run_tiny(tmp_path, cfg)

    assert (run_dir / "checkpoint.pt").exists()
    history = json.loads((run_dir / "loss_history.json").read_text())
    assert len(history) > 0
    for h in history:
        assert all(torch.isfinite(torch.tensor(float(h[k]))) for k in ("d_loss", "g_loss"))


def test_wgan_gp_training_step_runs_without_nan(tmp_path):
    cfg = _tiny_cfg("loss_wgangp_test", "wgan_gp", "none")
    run_dir = _run_tiny(tmp_path, cfg)

    history = json.loads((run_dir / "loss_history.json").read_text())
    assert len(history) > 0
    for h in history:
        assert all(torch.isfinite(torch.tensor(float(h[k]))) for k in ("d_loss", "g_loss"))


def test_spectral_norm_training_step_runs_without_nan(tmp_path):
    cfg = _tiny_cfg("stab_spectral_norm_test", "bce", "spectral")
    run_dir = _run_tiny(tmp_path, cfg)

    history = json.loads((run_dir / "loss_history.json").read_text())
    assert len(history) > 0


def test_checkpoint_contains_generator_and_discriminator_weights(tmp_path):
    cfg = _tiny_cfg("baseline_ckpt_test", "bce", "batch")
    run_dir = _run_tiny(tmp_path, cfg)
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    assert "G" in ckpt and "D" in ckpt and "config" in ckpt


def test_fixed_noise_is_reused_across_runs_with_same_path(tmp_path):
    from ganverse.train import get_or_create_fixed_noise
    path = tmp_path / "noise.pt"
    a = get_or_create_fixed_noise(latent_dim=8, n=4, seed=7, path=path)
    b = get_or_create_fixed_noise(latent_dim=8, n=4, seed=999, path=path)
    assert torch.equal(a, b)  # la segunda llamada debe reutilizar el archivo, no la nueva semilla
