"""Tests for roman_epsf.photutils.builder and roman_epsf.photutils.io."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roman_epsf import (
    GridEPSF,
    build_global_epsf,
    build_grid_epsf,
    build_multiepoch_epsf,
    load_epsf,
    save_epsf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _gaussian_star(image, x0, y0, amplitude, sigma=1.5):
    ny, nx = image.shape
    y_g, x_g = np.mgrid[:ny, :nx].astype(np.float64)
    image += amplitude * np.exp(
        -(((x_g - x0) ** 2 + (y_g - y0) ** 2) / (2 * sigma**2))
    )


def make_synthetic_scene(n_stars=25, image_size=150, cutout_size=15, seed=7):
    """Return (image, catalog, cutout_size) for a reproducible test scene."""
    rng = np.random.default_rng(seed)
    margin = cutout_size + 5
    xs = rng.uniform(margin, image_size - margin, n_stars)
    ys = rng.uniform(margin, image_size - margin, n_stars)
    amps = rng.uniform(200, 2000, n_stars)

    image = rng.normal(1.0, 0.1, (image_size, image_size))
    for x0, y0, amp in zip(xs, ys, amps):
        _gaussian_star(image, x0, y0, amp, sigma=1.5)

    catalog = pd.DataFrame(
        {
            "x_centroid": xs,
            "y_centroid": ys,
            "segment_flux": amps * 10,
        }
    )
    return image, catalog


class TestBuildGlobalEpsf:
    def setup_method(self):
        self.image, self.catalog = make_synthetic_scene()
        self.cutout_size = 15
        self.oversampling = 2

    def _build(self, **kwargs):
        return build_global_epsf(
            self.image,
            self.catalog,
            cutout_size=self.cutout_size,
            oversampling=self.oversampling,
            maxiters=3,
            flux_percentile_range=(0, 100),
            min_sep_factor=1.0,
            snr_threshold=2.0,
            min_snr_pixels=3,
            **kwargs,
        )

    def test_returns_epsf_and_df(self):
        epsf, fitted, df = self._build()
        assert epsf is not None
        assert len(df) > 0

    def test_output_shape(self):
        epsf, _, _ = self._build()
        # photutils EPSFBuilder output size = cutout_size * oversampling - (oversampling - 1)
        expected = self.cutout_size * self.oversampling - (self.oversampling - 1)
        assert epsf.data.shape == (expected, expected)

    def test_global_background_mode(self):
        epsf, _, _ = self._build(background="global")
        assert np.isfinite(epsf.data).all()

    def test_2d_background_mode(self):
        epsf, _, _ = self._build(
            background="2d",
            bkg2d_kwargs={"box_size": 50, "filter_size": 3},
        )
        assert np.isfinite(epsf.data).all()

    def test_no_background_mode(self):
        epsf, _, _ = self._build(background=None)
        assert np.isfinite(epsf.data).all()


class TestBuildGridEpsf:
    def setup_method(self):
        self.image, self.catalog = make_synthetic_scene(n_stars=40, image_size=200)
        self.cutout_size = 15
        self.oversampling = 2

    def test_returns_grid_epsf(self):
        grid = build_grid_epsf(
            self.image,
            self.catalog,
            n_grid_x=2,
            n_grid_y=2,
            cutout_size=self.cutout_size,
            oversampling=self.oversampling,
            maxiters=2,
            flux_percentile_range=(0, 100),
            min_sep_factor=1.0,
            snr_threshold=2.0,
            min_snr_pixels=3,
            min_stars_per_cell=1,
        )
        assert isinstance(grid, GridEPSF)

    def test_grid_data_shape(self):
        grid = build_grid_epsf(
            self.image,
            self.catalog,
            n_grid_x=2,
            n_grid_y=2,
            cutout_size=self.cutout_size,
            oversampling=self.oversampling,
            maxiters=2,
            flux_percentile_range=(0, 100),
            min_sep_factor=1.0,
            snr_threshold=2.0,
            min_snr_pixels=3,
            min_stars_per_cell=1,
        )
        expected_psf = self.cutout_size * self.oversampling - (self.oversampling - 1)
        assert grid.data.shape == (2, 2, expected_psf, expected_psf)

    def test_star_counts_shape(self):
        grid = build_grid_epsf(
            self.image,
            self.catalog,
            n_grid_x=2,
            n_grid_y=2,
            cutout_size=self.cutout_size,
            oversampling=self.oversampling,
            maxiters=2,
            flux_percentile_range=(0, 100),
            min_sep_factor=1.0,
            snr_threshold=2.0,
            min_snr_pixels=3,
            min_stars_per_cell=1,
        )
        assert grid.star_counts.shape == (2, 2)


class TestBuildMultiepochEpsf:
    def _make_cutout_list(self, n_epochs=8, cutout_size=15, seed=3):
        rng = np.random.default_rng(seed)
        ny = nx = cutout_size
        y_g, x_g = np.mgrid[:ny, :nx].astype(np.float64)
        sigma = 1.5
        cx, cy = cutout_size / 2.0, cutout_size / 2.0

        cutout_list = []
        for _ in range(n_epochs):
            dx, dy = rng.uniform(-0.4, 0.4, 2)
            amp = 500.0
            data = rng.normal(1.0, 0.05, (ny, nx))
            data += amp * np.exp(
                -(((x_g - (cx + dx)) ** 2 + (y_g - (cy + dy)) ** 2) / (2 * sigma**2))
            )
            cutout_list.append(
                {
                    "data": data,
                    "dq": None,
                    "cutout_center": (cx + dx, cy + dy),
                    "origin": (100.0, 100.0),
                }
            )
        return cutout_list

    def test_converges(self):
        cutouts = self._make_cutout_list()
        epsf, fitted = build_multiepoch_epsf(
            cutouts,
            oversampling=2,
            maxiters=3,
            snr_threshold=2.0,
            min_snr_pixels=3,
        )
        assert epsf is not None
        assert np.isfinite(epsf.data).all()

    def test_raises_on_empty_input(self):
        with pytest.raises((ValueError, Exception)):
            build_multiepoch_epsf([], oversampling=2, maxiters=3)


class TestFitsIO:
    def setup_method(self):
        self.image, self.catalog = make_synthetic_scene()
        self.cutout_size = 15
        self.oversampling = 2
        self.epsf, _, _ = build_global_epsf(
            self.image,
            self.catalog,
            cutout_size=self.cutout_size,
            oversampling=self.oversampling,
            maxiters=2,
            flux_percentile_range=(0, 100),
            min_sep_factor=1.0,
            snr_threshold=2.0,
            min_snr_pixels=3,
        )

    def test_single_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)

        try:
            save_epsf(self.epsf, path)
            loaded, hdr = load_epsf(path)
            assert np.allclose(self.epsf.data, loaded.data)

            # Verify evaluate() matches
            yy, xx = np.mgrid[-5:6, -5:6].astype(float)
            orig_eval = self.epsf.evaluate(xx, yy, flux=1.0, x_0=0.25, y_0=-0.15)
            load_eval = loaded.evaluate(xx, yy, flux=1.0, x_0=0.25, y_0=-0.15)
            assert np.allclose(orig_eval, load_eval)
        finally:
            path.unlink(missing_ok=True)

    def test_extra_header_preserved(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            save_epsf(self.epsf, path, extra_header={"FILTER": "F146", "NSTARS": 10})
            _, hdr = load_epsf(path)
            assert hdr.get("FILTER") == "F146"
            assert hdr.get("NSTARS") == 10
        finally:
            path.unlink(missing_ok=True)

    def test_grid_roundtrip(self):
        grid = build_grid_epsf(
            self.image,
            self.catalog,
            n_grid_x=2,
            n_grid_y=2,
            cutout_size=self.cutout_size,
            oversampling=self.oversampling,
            maxiters=2,
            flux_percentile_range=(0, 100),
            min_sep_factor=1.0,
            snr_threshold=2.0,
            min_snr_pixels=3,
            min_stars_per_cell=1,
        )

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            save_epsf(grid, path)
            loaded_grid, hdr = load_epsf(path)
            assert isinstance(loaded_grid, GridEPSF)
            assert np.allclose(grid.data, loaded_grid.data)
            assert np.allclose(grid.x_centers, loaded_grid.x_centers)
            assert np.allclose(grid.y_centers, loaded_grid.y_centers)
            assert np.array_equal(grid.star_counts, loaded_grid.star_counts)
        finally:
            path.unlink(missing_ok=True)
