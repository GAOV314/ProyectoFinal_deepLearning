import numpy as np
import torch

from ganverse.evidence import nearest_neighbor_test, plot_loss_curves
from ganverse.models import Discriminator
from ganverse.utils import save_json

TINY_FM = 8


def test_plot_loss_curves_writes_png_and_summary(tmp_path):
    history = [{"epoch": e, "iter": i, "d_loss": 0.5 + 0.01 * i, "g_loss": 1.0 - 0.01 * i}
               for e in range(3) for i in range(10)]
    history_path = tmp_path / "loss_history.json"
    save_json(history, history_path)

    out_path = tmp_path / "curve.png"
    summary = plot_loss_curves(history_path, out_path, title="prueba")

    assert out_path.exists()
    assert summary["n_iterations"] == len(history)
    assert "d_loss_final_mean" in summary and "g_loss_final_mean" in summary


def test_nearest_neighbor_test_finds_exact_match_by_pixels():
    torch.manual_seed(0)
    train_arr = (torch.rand(10, 64, 64, 3) * 255).to(torch.uint8).numpy()
    # la "galería" reusa exactamente la imagen de entrenamiento índice 3
    gallery_img = torch.from_numpy(train_arr[3]).float().permute(2, 0, 1) / 127.5 - 1.0
    gallery = gallery_img.unsqueeze(0)

    result = nearest_neighbor_test(gallery, train_arr, discriminator=None)
    assert result["results"][0]["nearest_train_index_pixel"] == 3
    assert result["results"][0]["pixel_mse"] < 1.0
    assert 0 in result["flagged_as_too_similar"]


def test_nearest_neighbor_test_with_discriminator_features():
    torch.manual_seed(0)
    d = Discriminator(feature_maps=TINY_FM, normalization="batch")
    train_arr = (np.random.rand(5, 64, 64, 3) * 255).astype(np.uint8)
    gallery = torch.randn(2, 3, 64, 64)

    result = nearest_neighbor_test(gallery, train_arr, discriminator=d, device=torch.device("cpu"))
    assert len(result["results"]) == 2
    for entry in result["results"]:
        assert "nearest_train_index_feature" in entry
        assert 0 <= entry["nearest_train_index_feature"] < 5


def test_nearest_neighbor_flags_distant_images_as_new():
    train_arr = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    gallery = torch.ones(1, 3, 64, 64)  # imagen blanca, muy distinta del entrenamiento negro
    result = nearest_neighbor_test(gallery, train_arr, discriminator=None)
    assert result["results"][0]["pixel_mse"] > result["threshold_pixel_mse"]
    assert len(result["flagged_as_too_similar"]) == 0
