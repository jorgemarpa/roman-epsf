"""Tests for roman_epsf.star_selection (method-agnostic shared utilities)."""

import numpy as np
import pandas as pd
import pytest

from roman_epsf.star_selection import extract_star_cutouts, select_stars


def _make_catalog(n: int = 30, image_shape=(200, 200), rng=None) -> pd.DataFrame:
    """Synthetic catalog with pixel coordinates spread across the detector."""
    if rng is None:
        rng = np.random.default_rng(42)
    ny, nx = image_shape
    return pd.DataFrame(
        {
            "x_centroid": rng.uniform(20, nx - 20, n),
            "y_centroid": rng.uniform(20, ny - 20, n),
            "segment_flux": rng.uniform(100, 10_000, n),
        }
    )


def _make_image_with_stars(catalog: pd.DataFrame, image_shape=(200, 200), rng=None) -> np.ndarray:
    """Tiny Gaussian-star image matching the catalog positions."""
    if rng is None:
        rng = np.random.default_rng(42)
    ny, nx = image_shape
    y_grid, x_grid = np.mgrid[:ny, :nx].astype(np.float64)
    image = rng.normal(1.0, 0.05, (ny, nx))  # background ~1 DN
    for _, row in catalog.iterrows():
        sigma = 1.5
        amp = row["segment_flux"] / (2 * np.pi * sigma**2)
        image += amp * np.exp(
            -(((x_grid - row["x_centroid"]) ** 2 + (y_grid - row["y_centroid"]) ** 2) / (2 * sigma**2))
        )
    return image


class TestSelectStars:
    def test_returns_dataframe_with_pix_columns(self):
        cat = _make_catalog(n=20)
        selected = select_stars(cat, image_shape=(200, 200), cutout_size=15)
        assert isinstance(selected, pd.DataFrame)
        assert "x_pix" in selected.columns
        assert "y_pix" in selected.columns

    def test_isolation_criterion_enforced(self):
        # Two stars very close together — only one should survive
        cat = pd.DataFrame(
            {
                "x_centroid": [100.0, 101.0, 50.0],
                "y_centroid": [100.0, 100.5, 50.0],
                "segment_flux": [5000.0, 5000.0, 5000.0],
            }
        )
        selected = select_stars(cat, image_shape=(200, 200), cutout_size=15, flux_percentile_range=(0, 100))
        xs = selected["x_pix"].to_numpy()
        ys = selected["y_pix"].to_numpy()
        if len(xs) > 1:
            min_sep = 15 * 1.2
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    dist = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    assert dist > min_sep, f"Stars {i} and {j} are too close: {dist:.2f} px"

    def test_edge_stars_excluded(self):
        edge_cat = pd.DataFrame(
            {
                "x_centroid": [1.0, 100.0],
                "y_centroid": [1.0, 100.0],
                "segment_flux": [5000.0, 5000.0],
            }
        )
        selected = select_stars(edge_cat, image_shape=(200, 200), cutout_size=25, flux_percentile_range=(0, 100))
        # The star at (1, 1) should be excluded by the edge buffer
        assert all(selected["x_pix"] > 17)
        assert all(selected["y_pix"] > 17)

    def test_raises_without_coord_columns(self):
        bad_cat = pd.DataFrame({"flux": [1000.0]})
        with pytest.raises(KeyError):
            select_stars(bad_cat, image_shape=(200, 200), cutout_size=15)

    def test_radec_columns_require_gwcs(self):
        cat = pd.DataFrame({"ra": [10.0], "dec": [20.0], "segment_flux": [500.0]})
        with pytest.raises(ValueError, match="gwcs"):
            select_stars(cat, image_shape=(200, 200), cutout_size=15)


class TestExtractStarCutouts:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.shape = (200, 200)
        self.cutout_size = 15
        self.cat = _make_catalog(n=30, image_shape=self.shape, rng=self.rng)
        self.image = _make_image_with_stars(self.cat, image_shape=self.shape, rng=self.rng)
        self.selected = select_stars(
            self.cat,
            image_shape=self.shape,
            cutout_size=self.cutout_size,
            flux_percentile_range=(0, 100),
        )

    def _extract(self, background, bkg2d_kwargs=None):
        return extract_star_cutouts(
            self.image,
            dq=None,
            selected_stars=self.selected,
            cutout_size=self.cutout_size,
            snr_threshold=2.0,
            min_snr_pixels=3,
            background=background,
            bkg2d_kwargs=bkg2d_kwargs,
        )

    def test_global_background(self):
        nddata, stars, df, std = self._extract("global")
        assert len(stars) > 0
        assert np.isfinite(std)
        assert std > 0

    def test_2d_background(self):
        nddata, stars, df, std = self._extract("2d", bkg2d_kwargs={"box_size": 50, "filter_size": 3})
        assert len(stars) > 0
        assert np.isfinite(std)

    def test_no_background(self):
        nddata, stars, df, std = self._extract(None)
        assert len(stars) > 0
        assert not np.isfinite(std)

    def test_invalid_background_raises(self):
        with pytest.raises(ValueError, match="background"):
            self._extract("invalid_mode")

    def test_output_shapes_consistent(self):
        _, stars, df, _ = self._extract("global")
        assert len(df) == len(stars)

    def test_dq_mask_applied(self):
        dq = np.zeros(self.shape, dtype=np.uint32)
        # Mark a rectangle as bad
        dq[90:110, 90:110] = 1
        nddata, stars, df, std = extract_star_cutouts(
            self.image,
            dq=dq,
            selected_stars=self.selected,
            cutout_size=self.cutout_size,
            background="global",
        )
        # Just verify it runs without error and returns something
        assert nddata is not None
