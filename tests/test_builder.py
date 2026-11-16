"""Tests for roman_epsf.photutils.builder and roman_epsf.photutils.io."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photutils.psf import ImagePSF

from roman_epsf import (
    GridEPSF,
    build_global_epsf,
    build_grid_epsf,
    build_multiepoch_epsf,
    create_stpsf_epsf,
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
        expected = self.cutout_size * self.oversampling
        if expected % 2 == 0:
            expected += 1
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

    def test_initial_epsf_false(self):
        epsf, _, _ = self._build(initial_epsf=False)
        assert epsf is not None
        assert np.isfinite(epsf.data).all()

    def test_initial_epsf_custom_imagepsf(self):
        expected_size = self.cutout_size * self.oversampling
        if expected_size % 2 == 0:
            expected_size += 1
        # Create a synthetic normalized Gaussian ImagePSF
        y, x = np.mgrid[:expected_size, :expected_size].astype(np.float64)
        c = (expected_size - 1) / 2.0
        synth_data = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * (2.0 * self.oversampling) ** 2))
        synth_data /= synth_data.sum()

        custom_epsf = ImagePSF(synth_data, oversampling=self.oversampling)
        epsf, fitted, df = self._build(initial_epsf=custom_epsf)
        assert epsf is not None
        assert epsf.data.shape == (expected_size, expected_size)
        assert np.isfinite(epsf.data).all()

    def test_initial_epsf_stpsf_mocked(self, monkeypatch):
        from unittest.mock import MagicMock
        from astropy.io import fits

        expected_size = self.cutout_size * self.oversampling
        if expected_size % 2 == 0:
            expected_size += 1
        # Create mock HDUList that mimics stpsf.calc_psf
        stpsf_fov_pixels = self.cutout_size * self.oversampling
        mock_data = np.zeros((stpsf_fov_pixels, stpsf_fov_pixels), dtype=np.float64)
        c = (stpsf_fov_pixels - 1) / 2.0
        y, x = np.mgrid[:stpsf_fov_pixels, :stpsf_fov_pixels]
        mock_data = np.exp(-((x - c) ** 2 + (y - c) ** 2) / 10.0)
        mock_data /= mock_data.sum()

        primary_hdu = fits.PrimaryHDU(data=mock_data)
        hdul = fits.HDUList([primary_hdu])

        mock_wfi = MagicMock()
        mock_wfi.calc_psf.return_value = hdul
        mock_roman = MagicMock()
        mock_roman.WFI.return_value = mock_wfi

        import sys
        mock_stpsf = MagicMock()
        mock_stpsf.roman = mock_roman
        monkeypatch.setitem(sys.modules, "stpsf", mock_stpsf)
        monkeypatch.setitem(sys.modules, "stpsf.roman", mock_roman)

        epsf, fitted, df = self._build(
            initial_epsf=True,
            detector="WFI02",
            filter_name="F158",
        )
        assert epsf is not None
        assert epsf.data.shape == (expected_size, expected_size)
        assert mock_wfi.detector == "WFI02"
        assert mock_wfi.filter == "F158"

    def test_initial_epsf_invalid_type(self):
        with pytest.raises(TypeError, match="initial_epsf must be False, True, or an ImagePSF/EPSFModel instance"):
            self._build(initial_epsf="invalid_model_type")



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
        expected_psf = self.cutout_size * self.oversampling
        if expected_psf % 2 == 0:
            expected_psf += 1
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


class TestCreateStpsfEpsf:
    def test_missing_stpsf_raises_import_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "stpsf", None)
        with pytest.raises(ImportError, match="The 'stpsf' package is required"):
            create_stpsf_epsf()

    def test_create_stpsf_epsf_success(self, monkeypatch):
        from unittest.mock import MagicMock
        from astropy.io import fits
        import sys

        cutout_size = 15
        oversampling = 3
        expected_size = cutout_size * oversampling
        if expected_size % 2 == 0:
            expected_size += 1

        raw_size = cutout_size * oversampling
        mock_data = np.ones((raw_size, raw_size), dtype=np.float64)
        hdul = fits.HDUList([fits.PrimaryHDU(data=mock_data)])

        mock_wfi = MagicMock()
        mock_wfi.calc_psf.return_value = hdul
        mock_roman = MagicMock()
        mock_roman.WFI.return_value = mock_wfi
        mock_stpsf = MagicMock()
        mock_stpsf.roman = mock_roman

        monkeypatch.setitem(sys.modules, "stpsf", mock_stpsf)
        monkeypatch.setitem(sys.modules, "stpsf.roman", mock_roman)

        model = create_stpsf_epsf(
            detector=4,
            filter_name="F146",
            oversampling=oversampling,
            cutout_size=cutout_size,
        )
        assert isinstance(model, ImagePSF)
        assert model.data.shape == (expected_size, expected_size)
        assert np.isclose(model.data.sum(), 1.0)
        assert mock_wfi.detector == "WFI04"
        assert mock_wfi.filter == "F146"


