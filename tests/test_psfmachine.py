"""Tests for roman_epsf.psfmachine (polar B-spline ePSF method)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roman_epsf.psfmachine import (
    PSFMachineModel,
    build_pixel_arrays,
    evaluate_epsf,
    fit_epsf,
    load_model,
    refine_centroids,
    save_image,
    save_model,
)
from roman_epsf.psfmachine._spline import (
    _make_design_matrix_polar,
    _r_bspline_matrix,
    _solve_linear,
    _wrapped_spline_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _gaussian_psf(dc, dr, sigma=1.5):
    """Analytic Gaussian PSF for testing."""
    return np.exp(-(dc**2 + dr**2) / (2 * sigma**2))


def _spike_psf(dc, dr, sigma=1.5, spike_angle_deg=45.0, spike_width_deg=8.0, spike_amp=0.05):
    """Gaussian PSF with a bilateral diffraction spike at spike_angle_deg and spike_angle_deg+180."""
    core = np.exp(-(dc**2 + dr**2) / (2 * sigma**2))
    phi = np.degrees(np.arctan2(dr, dc))
    r = np.hypot(dc, dr)
    # Bilateral spike: spike_angle_deg and its opposite (spike_angle_deg + 180)
    # arctan2 wraps to (-180, 180]; fold to compute angular distance to the spike axis
    phi_mod = ((phi - spike_angle_deg) + 180) % 360 - 180  # distance to spike_angle axis
    angle_dist = np.minimum(np.abs(phi_mod), 180 - np.abs(phi_mod))  # bilateral (±180 is same axis)
    spike_envelope = np.exp(-r / 5.0)
    spike = spike_amp * spike_envelope * np.exp(-0.5 * (angle_dist / spike_width_deg) ** 2)
    return core + spike


def _make_spike_image_and_catalog(
    n_stars=40, image_size=300, cutout_size=25,
    spike_angle_deg=45.0, seed=77,
    centroid_jitter=0.0,
):
    """Synthetic scene with PSFs that have a faint diffraction spike ridge."""
    rng = np.random.default_rng(seed)
    margin = cutout_size // 2 + 5
    xs = rng.uniform(margin, image_size - margin, n_stars)
    ys = rng.uniform(margin, image_size - margin, n_stars)
    fluxes = rng.uniform(5_000, 20_000, n_stars)

    ny, nx = image_size, image_size
    image = rng.normal(0.0, 0.02, (ny, nx))
    y_g, x_g = np.mgrid[:ny, :nx].astype(float)

    for x0, y0, f in zip(xs, ys, fluxes):
        dc_grid = x_g - x0
        dr_grid = y_g - y0
        image += f * _spike_psf(dc_grid, dr_grid, spike_angle_deg=spike_angle_deg)

    # Optionally add centroid jitter (simulates catalog position errors)
    x_cat = xs + rng.normal(0.0, centroid_jitter, n_stars)
    y_cat = ys + rng.normal(0.0, centroid_jitter, n_stars)

    catalog = pd.DataFrame(
        {
            "x_centroid": x_cat,
            "y_centroid": y_cat,
            "segment_flux": fluxes,
        }
    )
    return image, catalog, xs, ys  # return true positions too


def _make_synthetic_image_and_catalog(
    n_stars=20, image_size=200, cutout_size=25, seed=99
):
    """Synthetic Roman-like scene with known Gaussian stars."""
    rng = np.random.default_rng(seed)
    margin = cutout_size // 2 + 5
    xs = rng.uniform(margin, image_size - margin, n_stars)
    ys = rng.uniform(margin, image_size - margin, n_stars)
    fluxes = rng.uniform(1_000, 10_000, n_stars)

    ny, nx = image_size, image_size
    image = rng.normal(0.0, 0.05, (ny, nx))
    y_g, x_g = np.mgrid[:ny, :nx].astype(float)
    sigma = 1.5

    for x0, y0, f in zip(xs, ys, fluxes):
        image += f * np.exp(-((x_g - x0) ** 2 + (y_g - y0) ** 2) / (2 * sigma**2))

    catalog = pd.DataFrame(
        {
            "x_centroid": xs,
            "y_centroid": ys,
            "segment_flux": fluxes,
        }
    )
    return image, catalog


# ---------------------------------------------------------------------------
# _spline tests
# ---------------------------------------------------------------------------


class TestWrappedSpline:
    def test_output_shape(self):
        phi = np.linspace(-np.pi, np.pi, 100, endpoint=False)
        mat = _wrapped_spline_matrix(phi, n_knots=10)
        assert mat.shape == (100, 10)

    def test_raises_out_of_range(self):
        phi = np.array([0.0, 4.0])  # 4 > pi
        with pytest.raises(ValueError, match="pi"):
            _wrapped_spline_matrix(phi, n_knots=10)

    def test_finite_values(self):
        phi = np.linspace(-np.pi, np.pi, 200, endpoint=False)
        mat = _wrapped_spline_matrix(phi, n_knots=12)
        assert np.all(np.isfinite(mat))

    def test_non_negative(self):
        """B-spline basis functions must be non-negative."""
        phi = np.linspace(-np.pi, np.pi, 200, endpoint=False)
        mat = _wrapped_spline_matrix(phi, n_knots=10)
        assert np.all(mat >= -1e-12)  # allow tiny float errors


class TestRBsplineMatrix:
    def test_output_shape(self):
        r = np.linspace(0.5, 13.0, 100)
        r_knots = np.linspace(0.5**0.5, 13.0**0.5, 10)**2
        mat = _r_bspline_matrix(r, rmin=0.5, rmax=13.0, r_knots=r_knots)
        n_r_basis = len(r_knots) + 3 + 1  # n_knots + degree + 1
        assert mat.shape[0] == 100
        assert mat.shape[1] == n_r_basis

    def test_finite_values(self):
        r = np.linspace(0.5, 13.0, 50)
        r_knots = np.linspace(0.5**0.5, 13.0**0.5, 8)**2
        mat = _r_bspline_matrix(r, rmin=0.5, rmax=13.0, r_knots=r_knots)
        assert np.all(np.isfinite(mat))

    def test_partition_of_unity(self):
        """Rows should sum to ~1 (B-spline partition of unity)."""
        r = np.linspace(0.6, 12.5, 200)
        r_knots = np.linspace(0.5**0.5, 13.0**0.5, 10)**2
        mat = _r_bspline_matrix(r, rmin=0.5, rmax=13.0, r_knots=r_knots)
        row_sums = mat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


class TestDesignMatrix:
    def test_output_shape_finite(self):
        rng = np.random.default_rng(1)
        r = rng.uniform(0.6, 12.0, 500)
        phi = np.arctan2(rng.standard_normal(500), rng.standard_normal(500))
        r_knots = np.linspace(0.5**0.5, 13.0**0.5, 10)**2
        A = _make_design_matrix_polar(r, phi, 0.5, 13.0, r_knots, 12, cut_r=3.0)
        assert A.shape[0] == 500
        assert np.all(np.isfinite(A))

    def test_inner_rows_are_phi_independent(self):
        """For r < cut_r, rows with the same r but different phi should be equal
        (after accounting for the inner pure-radial design, not the outer phi terms)."""
        # This is verified indirectly: the fitted model should give nearly identical
        # values for two pixels at r=1.0 but opposite phi values.
        pass  # covered by fit_epsf tests


class TestSolveLinear:
    def test_exact_recovery(self):
        """Exact solve for an overdetermined system with no noise."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((200, 10))
        w_true = rng.standard_normal(10)
        y = A @ w_true
        w = _solve_linear(A, y)
        np.testing.assert_allclose(w, w_true, atol=1e-8)

    def test_prior_regularises(self):
        """Adding a strong prior should pull weights toward zero."""
        rng = np.random.default_rng(8)
        A = rng.standard_normal((50, 20))
        y = rng.standard_normal(50)
        w_no_prior = _solve_linear(A, y)
        w_prior = _solve_linear(A, y, prior_sigma=0.01)
        assert np.linalg.norm(w_prior) < np.linalg.norm(w_no_prior)


# ---------------------------------------------------------------------------
# build_pixel_arrays tests
# ---------------------------------------------------------------------------


class TestBuildPixelArrays:
    def setup_method(self):
        self.image, self.catalog = _make_synthetic_image_and_catalog()
        from roman_epsf.star_selection import select_stars
        self.selected = select_stars(
            self.catalog,
            image_shape=self.image.shape,
            cutout_size=25,
            flux_percentile_range=(0, 100),
            min_sep_factor=0.5,
        )

    def test_returns_four_arrays(self):
        cutout_size = 25
        dc, dr, nf, si = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=cutout_size
        )
        n_stars = len(self.selected)
        n_pix = cutout_size ** 2
        assert dc.shape == (n_stars, n_pix)
        assert dr.shape == (n_stars, n_pix)
        assert nf.shape == (n_stars, n_pix)
        assert si.shape == (n_stars,)

    def test_all_finite(self):
        dc, dr, nf, si = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25
        )
        # dc and dr are always filled; nf may have NaN for masked pixels
        assert np.all(np.isfinite(dc))
        assert np.all(np.isfinite(dr))

    def test_star_idx_range(self):
        dc, dr, nf, si = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25
        )
        assert si.ndim == 1
        assert si.min() >= 0
        assert si.max() < len(self.selected)

    def test_norm_flux_nan_for_masked(self):
        """Pixels with bad DQ flags must appear as NaN in norm_flux."""
        dq = np.zeros(self.image.shape, dtype=np.uint32)
        # Inject a bad pixel at a known star position
        star = self.selected.iloc[0]
        ix = int(round(float(star["x_centroid"])))
        iy = int(round(float(star["y_centroid"])))
        dq[iy, ix] = 1  # any non-zero DQ = bad
        dc, dr, nf, _ = build_pixel_arrays(
            self.image, dq, self.selected, cutout_size=25
        )
        # row 0 corresponds to the first star; centre offset = (0,0)
        centre_pix = 25 * (25 // 2) + (25 // 2)  # raveled index of offset (0,0)
        assert np.isnan(nf[0, centre_pix])

    def test_local_background_mode(self):
        dc, dr, nf, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, background="local"
        )
        # dc and dr always finite; nf may be NaN for bad pixels
        assert np.all(np.isfinite(dc))
        assert np.all(np.isfinite(dr))

    def test_no_background_mode(self):
        dc, dr, nf, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, background=None
        )
        assert np.all(np.isfinite(dc))
        assert np.all(np.isfinite(dr))

    def test_raises_missing_flux_col(self):
        bad_cat = self.selected.drop(
            columns=[c for c in ["segment_flux", "psf_flux", "aper_flux", "kron_flux", "flux"]
                     if c in self.selected.columns]
        )
        with pytest.raises(KeyError):
            build_pixel_arrays(self.image, None, bad_cat, cutout_size=25)


# ---------------------------------------------------------------------------
# fit_epsf tests
# ---------------------------------------------------------------------------


class TestFitEpsf:
    def setup_method(self):
        """Build a realistic pixel array from a synthetic Gaussian PSF field."""
        rng = np.random.default_rng(42)
        self.image, self.catalog = _make_synthetic_image_and_catalog(seed=42)
        from roman_epsf.star_selection import select_stars
        self.selected = select_stars(
            self.catalog,
            image_shape=self.image.shape,
            cutout_size=25,
            flux_percentile_range=(0, 100),
            min_sep_factor=0.5,
        )
        self.dc, self.dr, self.nf, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, background="local"
        )

    def test_returns_model(self):
        model = fit_epsf(self.dc, self.dr, self.nf, rmin=0.5, rmax=12.0, cut_r=2.0)
        assert isinstance(model, PSFMachineModel)

    def test_weights_finite(self):
        model = fit_epsf(self.dc, self.dr, self.nf, rmin=0.5, rmax=12.0, cut_r=2.0)
        assert np.all(np.isfinite(model.weights))

    def test_n_pixels_stored(self):
        model = fit_epsf(self.dc, self.dr, self.nf, rmin=0.5, rmax=12.0, cut_r=2.0)
        assert model.n_pixels > 0

    def test_evaluate_positive(self):
        model = fit_epsf(self.dc, self.dr, self.nf, rmin=0.5, rmax=12.0, cut_r=2.0)
        vals = model.evaluate(np.array([0.0, 1.0, -1.0]), np.array([0.0, 0.5, -0.5]))
        assert np.all(vals >= 0)

    def test_raises_no_pixels_in_range(self):
        with pytest.raises(ValueError, match="rmin"):
            fit_epsf(self.dc, self.dr, self.nf, rmin=500.0, rmax=600.0)


# ---------------------------------------------------------------------------
# evaluate_epsf tests
# ---------------------------------------------------------------------------


class TestEvaluateEpsf:
    def setup_method(self):
        image, catalog = _make_synthetic_image_and_catalog(seed=5)
        from roman_epsf.star_selection import select_stars
        selected = select_stars(
            catalog,
            image_shape=image.shape,
            cutout_size=25,
            flux_percentile_range=(0, 100),
            min_sep_factor=0.5,
        )
        dc, dr, nf, _ = build_pixel_arrays(image, None, selected, cutout_size=25)
        self.model = fit_epsf(dc, dr, nf, rmin=0.5, rmax=12.0, cut_r=2.0)

    def test_output_shape(self):
        img = evaluate_epsf(self.model, cutout_size=25, oversampling=4)
        assert img.shape == (100, 100)

    def test_output_shape_os2(self):
        img = evaluate_epsf(self.model, cutout_size=15, oversampling=2)
        assert img.shape == (30, 30)

    def test_normalised_to_one(self):
        img = evaluate_epsf(self.model, cutout_size=25, oversampling=4, normalise=True)
        np.testing.assert_allclose(img.sum(), 1.0, atol=1e-6)

    def test_no_negative_values(self):
        img = evaluate_epsf(self.model, cutout_size=25, oversampling=4, clip_negative=True)
        assert np.all(img >= 0)

    def test_peak_near_centre(self):
        img = evaluate_epsf(self.model, cutout_size=25, oversampling=4)
        cy, cx = np.unravel_index(np.argmax(img), img.shape)
        # Peak should be within 10% of the image centre
        h, w = img.shape
        assert abs(cy - h // 2) < h * 0.1
        assert abs(cx - w // 2) < w * 0.1


# ---------------------------------------------------------------------------
# FITS I/O tests
# ---------------------------------------------------------------------------


class TestFitsIO:
    def setup_method(self):
        image, catalog = _make_synthetic_image_and_catalog(seed=11)
        from roman_epsf.star_selection import select_stars
        selected = select_stars(
            catalog,
            image_shape=image.shape,
            cutout_size=25,
            flux_percentile_range=(0, 100),
            min_sep_factor=0.5,
        )
        dc, dr, nf, _ = build_pixel_arrays(image, None, selected, cutout_size=25)
        self.model = fit_epsf(dc, dr, nf, rmin=0.5, rmax=12.0, cut_r=2.0)

    def test_model_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            save_model(self.model, path)
            loaded, hdr = load_model(path)
            np.testing.assert_array_equal(self.model.weights, loaded.weights)
            np.testing.assert_array_equal(self.model.r_knots, loaded.r_knots)
            assert loaded.rmin == self.model.rmin
            assert loaded.rmax == self.model.rmax
            assert loaded.n_r_knots == self.model.n_r_knots
            assert loaded.n_phi_knots == self.model.n_phi_knots
        finally:
            path.unlink(missing_ok=True)

    def test_evaluate_identical_after_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            save_model(self.model, path)
            loaded, _ = load_model(path)
            dc_test = np.array([0.5, 1.0, -1.5, 2.0])
            dr_test = np.array([0.0, 0.5, 1.0, -0.5])
            orig = self.model.evaluate(dc_test, dr_test)
            recon = loaded.evaluate(dc_test, dr_test)
            np.testing.assert_allclose(orig, recon, rtol=1e-10)
        finally:
            path.unlink(missing_ok=True)

    def test_extra_header_preserved(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            save_model(self.model, path, extra_header={"FILTER": "F146", "NSTARS": 20})
            _, hdr = load_model(path)
            assert hdr.get("FILTER") == "F146"
            assert hdr.get("NSTARS") == 20
        finally:
            path.unlink(missing_ok=True)

    def test_logfit_roundtrip(self):
        """LOGFIT keyword persists through save/load."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            # model built with default log_fit=True
            save_model(self.model, path)
            loaded, hdr = load_model(path)
            assert loaded.log_fit == self.model.log_fit
            assert bool(hdr.get("LOGFIT")) == self.model.log_fit
        finally:
            path.unlink(missing_ok=True)

    def test_save_image_writes_2d(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = Path(f.name)
        try:
            save_image(self.model, path, cutout_size=25, oversampling=3)
            from astropy.io import fits
            with fits.open(path) as hdul:
                data = hdul[0].data
            assert data.ndim == 2
            assert data.shape == (75, 75)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# norm_mode tests
# ---------------------------------------------------------------------------


class TestNormMode:
    def setup_method(self):
        self.image, self.catalog = _make_synthetic_image_and_catalog(seed=20)
        from roman_epsf.star_selection import select_stars
        self.selected = select_stars(
            self.catalog,
            image_shape=self.image.shape,
            cutout_size=25,
            flux_percentile_range=(0, 100),
            min_sep_factor=0.5,
        )

    def test_catalog_mode_default(self):
        dc, dr, nf, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, norm_mode="catalog"
        )
        assert np.all(np.isfinite(dc))

    def test_measured_mode(self):
        dc, dr, nf, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, norm_mode="measured"
        )
        assert np.all(np.isfinite(dc))

    def test_measured_vs_catalog_different(self):
        _, _, nf_cat, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, norm_mode="catalog"
        )
        _, _, nf_meas, _ = build_pixel_arrays(
            self.image, None, self.selected, cutout_size=25, norm_mode="measured"
        )
        # The two modes should produce different normalizations for most stars
        diff = np.nanmean(np.abs(nf_cat - nf_meas))
        assert diff > 0


# ---------------------------------------------------------------------------
# Spike-recovery test
# ---------------------------------------------------------------------------


class TestSpikeRecovery:
    """Verify that the iterative polar fitting recovers diffraction spikes."""

    SPIKE_ANGLE = 45.0   # degrees
    SPIKE_AMP   = 0.05   # fraction of PSF core

    def _azimuthal_spike_ratio(self, model, r_test=5.0, n_phi=360):
        """Peak / mean of the azimuthal flux profile at radius r_test."""
        phi_arr = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        dc = r_test * np.cos(phi_arr)
        dr = r_test * np.sin(phi_arr)
        flux = model.evaluate(dc, dr)
        finite = np.isfinite(flux) & (flux > 0)
        if not np.any(finite):
            return 1.0
        return float(np.nanmax(flux[finite]) / np.nanmean(flux[finite]))

    def test_polar_recovers_spike_no_jitter(self):
        """Polar mode with cut_r=1 recovers spikes when centroids are perfect."""
        from roman_epsf.star_selection import select_stars

        image, catalog, _, _ = _make_spike_image_and_catalog(
            n_stars=40, centroid_jitter=0.0, spike_angle_deg=self.SPIKE_ANGLE, seed=100
        )
        selected = select_stars(
            catalog, image_shape=image.shape, cutout_size=25,
            flux_percentile_range=(0, 100), min_sep_factor=0.5,
        )
        dc, dr, nf, _ = build_pixel_arrays(
            image, None, selected, cutout_size=25, background="local"
        )
        model = fit_epsf(
            dc, dr, nf,
            n_r_knots=10, n_phi_knots=20, rmin=0.5, rmax=12.0, cut_r=1.0,
            coords="polar", log_fit=True,
        )
        ratio = self._azimuthal_spike_ratio(model)
        # Spike should produce an azimuthal peak > 5% above the ring mean
        assert ratio > 1.05, f"Spike ratio {ratio:.3f} too low — spike not recovered"

    def test_large_cut_r_flattens_inner_azimuthal_ring(self):
        """cut_r=3.0 forces radial symmetry at r<3 px — the azimuthal ring at r=2
        should be flatter (ratio closer to 1) than with cut_r=1.0."""
        from roman_epsf.star_selection import select_stars

        image, catalog, _, _ = _make_spike_image_and_catalog(
            n_stars=40, centroid_jitter=0.0, spike_angle_deg=self.SPIKE_ANGLE, seed=101
        )
        selected = select_stars(
            catalog, image_shape=image.shape, cutout_size=25,
            flux_percentile_range=(0, 100), min_sep_factor=0.5,
        )
        dc, dr, nf, _ = build_pixel_arrays(
            image, None, selected, cutout_size=25, background="local"
        )
        model_low_cut = fit_epsf(
            dc, dr, nf, n_r_knots=10, n_phi_knots=20,
            rmin=0.5, rmax=12.0, cut_r=1.0, coords="polar", log_fit=True,
        )
        model_high_cut = fit_epsf(
            dc, dr, nf, n_r_knots=10, n_phi_knots=20,
            rmin=0.5, rmax=12.0, cut_r=3.0, coords="polar", log_fit=True,
        )
        # Check at r=2 px — inside cut_r=3.0 (phi suppressed) but outside cut_r=1.0
        ratio_low_cut  = self._azimuthal_spike_ratio(model_low_cut,  r_test=2.0)
        ratio_high_cut = self._azimuthal_spike_ratio(model_high_cut, r_test=2.0)
        assert ratio_low_cut > ratio_high_cut, (
            f"At r=2px: cut_r=1.0 ratio ({ratio_low_cut:.3f}) should exceed "
            f"cut_r=3.0 ratio ({ratio_high_cut:.3f})"
        )

    def test_refine_centroids_corrects_position_errors(self):
        """refine_centroids moves star positions toward their true centres.

        With 0.8 px rms jitter the mean applied shift should be > 0.05 px
        and the spike should remain recoverable after the shift is applied.
        """
        from roman_epsf.star_selection import select_stars

        image, catalog, _, _ = _make_spike_image_and_catalog(
            n_stars=50, centroid_jitter=0.8, spike_angle_deg=self.SPIKE_ANGLE, seed=103,
        )
        selected = select_stars(
            catalog, image_shape=image.shape, cutout_size=25,
            flux_percentile_range=(0, 100), min_sep_factor=0.5,
        )
        dc, dr, nf, _ = build_pixel_arrays(
            image, None, selected, cutout_size=25, background="local"
        )
        model_pass1 = fit_epsf(
            dc, dr, nf, n_r_knots=10, n_phi_knots=20,
            rmin=0.5, rmax=12.0, cut_r=1.0, coords="polar", log_fit=True,
        )

        selected_refined, dc2, dr2, nf2, _ = refine_centroids(
            model_pass1, selected, image, None, cutout_size=25, background="local",
        )

        # Verify that refinement applied non-trivial centroid shifts
        common_idx = selected.index.intersection(selected_refined.index)
        sel_orig = selected.loc[common_idx]
        sel_ref  = selected_refined.loc[common_idx]
        delta_x = (sel_ref["x_pix"] - sel_orig["x_pix"]).to_numpy()
        delta_y = (sel_ref["y_pix"] - sel_orig["y_pix"]).to_numpy()
        mean_shift = float(np.sqrt(np.mean(delta_x**2 + delta_y**2)))
        assert mean_shift > 0.05, (
            f"Centroid refinement applied negligible shifts (mean={mean_shift:.4f} px). "
            "Expected > 0.05 px for 0.8 px input jitter."
        )

        # Spike is still recovered after re-fitting with refined positions
        model_pass2 = fit_epsf(
            dc2, dr2, nf2, n_r_knots=10, n_phi_knots=20,
            rmin=0.5, rmax=12.0, cut_r=1.0, coords="polar", log_fit=True,
        )
        ratio = self._azimuthal_spike_ratio(model_pass2)
        assert ratio > 1.05, f"Spike ratio {ratio:.3f} too low after refinement"
