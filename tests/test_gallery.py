import numpy as np
import torch

from ganverse.gallery import (
    NAME_POOL, _dominant_element, build_gallery, generate_candidates,
    score_candidates, select_diverse_top_k,
)
from ganverse.dataset import ELEMENTS
from ganverse.models import Discriminator, Generator

TINY_LATENT = 8
TINY_FM = 8


def test_generate_candidates_is_reproducible_given_seed():
    g = Generator(latent_dim=TINY_LATENT, feature_maps=TINY_FM)
    images_a, z_a = generate_candidates(g, n=6, latent_dim=TINY_LATENT, seed=42, device=torch.device("cpu"))
    images_b, z_b = generate_candidates(g, n=6, latent_dim=TINY_LATENT, seed=42, device=torch.device("cpu"))
    assert torch.equal(z_a, z_b)
    assert torch.allclose(images_a, images_b)


def test_select_diverse_top_k_respects_k_and_uniqueness():
    torch.manual_seed(0)
    images = torch.randn(20, 3, 64, 64)
    scores = torch.randn(20)
    selected = select_diverse_top_k(images, scores, k=5, quality_pool_frac=1.0)
    assert len(selected) == 5
    assert len(set(selected)) == 5


def test_select_diverse_top_k_starts_from_highest_score_in_pool():
    # valores dentro de [-1, 1] (rango real de salida de Tanh); fuera de ese
    # rango `tensor_to_uint8` los satura y dos imágenes distintas podrían
    # cuantizarse al mismo uint8.
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    images = torch.stack([torch.full((3, 64, 64), v) for v in values])
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.0])
    selected = select_diverse_top_k(images, scores, k=2, quality_pool_frac=1.0)
    assert selected[0] == 1  # mayor puntaje arranca la búsqueda farthest-point
    # el segundo debe ser el más lejano a idx1 (valor -0.5): idx4 (valor 1.0)
    assert selected[1] == 4


def test_select_diverse_top_k_maximizes_mutual_diversity():
    # Dos "modos" casi idénticos (0 y 1) muy bien puntuados, y dos modos
    # realmente distintos (2 y 3) algo peor puntuados. Un criterio ingenuo de
    # "solo top-k por puntaje" elegiría {0,1} (casi duplicados); farthest-point
    # debe preferir diversidad real una vez que ambos pasan el filtro de calidad.
    values = [-1.0, -0.95, 0.3, 0.9]
    images = torch.stack([torch.full((3, 64, 64), v) for v in values])
    scores = torch.tensor([1.0, 0.95, 0.5, 0.4])
    selected = select_diverse_top_k(images, scores, k=2, quality_pool_frac=1.0)
    assert selected[0] == 0  # arranca en el de mayor puntaje
    assert selected[1] == 3  # el más distinto a 0, no el casi-duplicado idx1


def test_dominant_element_matches_known_palette():
    for element, palette in ELEMENTS.items():
        img = np.tile(np.array(palette["body"], dtype=np.uint8), (64, 64, 1))
        assert _dominant_element(img) == element


def test_build_gallery_produces_k_images_with_names_and_metadata(tmp_path):
    torch.manual_seed(0)
    g = Generator(latent_dim=TINY_LATENT, feature_maps=TINY_FM)
    d = Discriminator(feature_maps=TINY_FM, normalization="batch")
    info = build_gallery(g, d, out_dir=tmp_path, n_candidates=30, k=5,
                          latent_dim=TINY_LATENT, seed=1, device=torch.device("cpu"))

    assert info["n_selected"] == 5
    assert info["selection_rate"] == "5 de 30"
    pngs = list(tmp_path.glob("char_*.png"))
    assert len(pngs) == 5
    assert (tmp_path / "metadata.csv").exists()
    assert (tmp_path / "z_vectors.pt").exists()
    assert (tmp_path / "selection_info.json").exists()

    names_used = {row["name"] for row in info["rows"]}
    all_names = {n for pairs in NAME_POOL.values() for n, _ in pairs}
    assert names_used.issubset(all_names)


def test_build_gallery_clears_stale_files_from_a_previous_selection(tmp_path):
    # Simula una corrida previa (p. ej. con otro seed/criterio) cuyos PNGs no
    # deben sobrevivir a una nueva generación de la galería en el mismo out_dir.
    (tmp_path / "char_01_viejo.png").write_bytes(b"not a real png")
    (tmp_path / "char_07_otro_viejo.png").write_bytes(b"not a real png")

    torch.manual_seed(0)
    g = Generator(latent_dim=TINY_LATENT, feature_maps=TINY_FM)
    d = Discriminator(feature_maps=TINY_FM, normalization="batch")
    build_gallery(g, d, out_dir=tmp_path, n_candidates=20, k=5,
                  latent_dim=TINY_LATENT, seed=1, device=torch.device("cpu"))

    pngs = {p.name for p in tmp_path.glob("char_*.png")}
    assert "char_01_viejo.png" not in pngs
    assert "char_07_otro_viejo.png" not in pngs
    assert len(pngs) == 5
