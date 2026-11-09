import numpy as np
import torch

from ganverse.dataset import (
    ELEMENT_NAMES, IMG_SIZE, GlimmerfenDataset, generate_dataset_array, sample_params, render_sprite,
)
import random


def test_generate_dataset_array_shape_and_dtype(tiny_dataset_array):
    arr, params = tiny_dataset_array
    assert arr.shape == (6, IMG_SIZE, IMG_SIZE, 3)
    assert arr.dtype == np.uint8
    assert len(params) == 6


def test_generate_dataset_is_deterministic_given_seed():
    arr_a, _ = generate_dataset_array(n=4, seed=123)
    arr_b, _ = generate_dataset_array(n=4, seed=123)
    assert np.array_equal(arr_a, arr_b)


def test_generate_dataset_different_seeds_differ():
    arr_a, _ = generate_dataset_array(n=4, seed=1)
    arr_b, _ = generate_dataset_array(n=4, seed=2)
    assert not np.array_equal(arr_a, arr_b)


def test_sample_params_uses_known_elements():
    rng = random.Random(0)
    for _ in range(20):
        params = sample_params(rng)
        assert params.element in ELEMENT_NAMES
        assert params.appendage in {"cuernos", "alas", "orejas", "cola", "antena", "ninguno"}


def test_render_sprite_produces_correct_size_image():
    rng = random.Random(5)
    params = sample_params(rng)
    params.seed = 5
    img = render_sprite(params)
    assert img.size == (IMG_SIZE, IMG_SIZE)
    assert img.mode == "RGB"


def test_glimmerfen_dataset_normalizes_to_minus1_1(tmp_path):
    from ganverse.dataset import save_dataset
    save_dataset(n=5, out_dir=tmp_path, seed=10, n_png_samples=2)
    ds = GlimmerfenDataset(tmp_path / "glimmerfen_64x64.npy", augment=False)
    assert len(ds) == 5
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, IMG_SIZE, IMG_SIZE)
    assert sample.min() >= -1.0 - 1e-5
    assert sample.max() <= 1.0 + 1e-5


def test_save_dataset_writes_metadata_with_license_and_origin(tmp_path):
    from ganverse.dataset import save_dataset
    from ganverse.utils import load_json
    save_dataset(n=3, out_dir=tmp_path, seed=1, n_png_samples=1)
    meta = load_json(tmp_path / "metadata.json")
    assert meta["n_images"] == 3
    assert "origin" in meta and "license" in meta
    assert (tmp_path / "samples" / "sample_000.png").exists()
