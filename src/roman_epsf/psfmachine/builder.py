"""
roman_epsf.psfmachine.builder
==============================
PSFMachine-style polar B-spline ePSF construction using pure NumPy / SciPy.

Methodology (Hedges et al. 2021)
---------------------------------
1. Extract cutout stamps around a selection of stars.
2. Normalise each stamp's pixel values by the catalog flux estimate for the
   corresponding star.
3. Record the per-pixel offset from the star position in detector pixels
   (``delta_col``, ``delta_row``).
4. Convert to polar coordinates ``(r, phi)`` and fit a B-spline model on the
   polar grid to the assembled normalised-flux vs. position data.
5. Evaluate the fitted model on any oversampled pixel grid.

The star-selection step deliberately uses looser criteria than the photutils
path (fainter stars are accepted because flux normalisation removes the
brightness dependence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats

from roman_epsf.psfmachine._spline import (
    _cartesian_knots_1d,
    _make_design_matrix_cartesian,
    _make_design_matrix_polar,
    _solve_linear,
)

log = logging.getLogger(__name__)

__all__ = ["PSFMachineModel", "build_pixel_arrays", "fit_epsf", "evaluate_epsf"]


@dataclass
class PSFMachineModel:
    """Container for a fitted B-spline ePSF model.

    Supports two coordinate systems controlled by the *coords* attribute:

    ``"polar"``
        Tensor product of radial clamped B-splines × periodic azimuthal
        B-splines, with a radial-only core inside *cut_r*.  Uses ``r_knots``
        for the radial basis.

    ``"cartesian"``
        Full 2D tensor product of clamped B-splines along the column and row
        axes, with sqrt-spaced knots inside *cut_r* for higher resolution at
        the PSF core.  Uses ``r_knots`` for the column axis and ``row_knots``
        for the row axis.

    The :meth:`evaluate` method automatically dispatches to the correct design
    matrix based on *coords*.

    Attributes
    ----------
    weights : np.ndarray
        Fitted B-spline coefficients.
    r_knots : np.ndarray
        Radial knot positions for polar mode, or column-axis knot positions
        for Cartesian mode.
    n_r_knots : int
        Number of radial / column knots.
    n_phi_knots : int
        Number of azimuthal knots (polar mode).  ``0`` for Cartesian mode.
    rmin : float
        Inner radial limit in detector pixels (polar mode).
    rmax : float
        Outer radial / domain half-width limit in detector pixels.
    cut_r : float
        Polar: radius below which only radial terms are used.
        Cartesian: transition radius from sqrt-spaced to linear knots.
    degree : int
        B-spline degree (default 3 = cubic).
    n_stars : int
        Number of stars used to build the model.
    n_pixels : int
        Total number of valid pixels used to fit the model.
    coords : str
        Coordinate system: ``"polar"`` (default) or ``"cartesian"``.
    row_knots : np.ndarray or None
        Row-axis knot positions for Cartesian mode.  ``None`` for polar mode.
    """

    weights: np.ndarray
    r_knots: np.ndarray
    n_r_knots: int
    n_phi_knots: int
    rmin: float
    rmax: float
    cut_r: float
    degree: int = 3
    n_stars: int = 0
    n_pixels: int = 0
    log_fit: bool = True
    coords: str = "polar"
    row_knots: np.ndarray | None = field(default=None, repr=False)
    _A_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def evaluate(
        self,
        delta_col: np.ndarray,
        delta_row: np.ndarray,
        clip_negative: bool = False,
    ) -> np.ndarray:
        """Evaluate the ePSF model at arbitrary ``(delta_col, delta_row)`` offsets.

        Automatically dispatches to the polar or Cartesian design matrix based
        on :attr:`coords`.  When :attr:`log_fit` is ``True`` (the default),
        raw model outputs are in log₁₀ space and are transformed back to flux
        via ``10 ** val`` before returning.

        Parameters
        ----------
        delta_col, delta_row : np.ndarray
            1D arrays of pixel offsets from the star centre (detector pixels).
        clip_negative : bool
            Set negative model values to zero after evaluation.  Default
            ``False`` (unnecessary when ``log_fit=True`` since ``10^x > 0``).

        Returns
        -------
        np.ndarray
            Evaluated normalised flux at each supplied position.
        """
        dc = np.asarray(delta_col, dtype=float)
        dr = np.asarray(delta_row, dtype=float)

        if self.coords == "cartesian":
            return self._evaluate_cartesian(dc, dr, clip_negative)
        return self._evaluate_polar(dc, dr, clip_negative)

    def _evaluate_polar(
        self,
        dc: np.ndarray,
        dr: np.ndarray,
        clip_negative: bool,
    ) -> np.ndarray:
        r = np.hypot(dc, dr)
        phi = np.arctan2(dr, dc)
        in_range = (r >= self.rmin) & (r <= self.rmax)
        result = np.zeros_like(r)
        if np.any(in_range):
            A = _make_design_matrix_polar(
                r[in_range], phi[in_range],
                self.rmin, self.rmax,
                self.r_knots, self.n_phi_knots, self.cut_r, self.degree,
            )
            vals = A @ self.weights
            if self.log_fit:
                vals = 10.0 ** vals
            if clip_negative:
                vals = np.clip(vals, 0.0, None)
            result[in_range] = vals
        return result

    def _evaluate_cartesian(
        self,
        dc: np.ndarray,
        dr: np.ndarray,
        clip_negative: bool,
    ) -> np.ndarray:
        if self.row_knots is None:
            raise ValueError(
                "PSFMachineModel.row_knots is None for a Cartesian model. "
                "Re-fit with coords='cartesian' or load from the correct FITS file."
            )
        in_range = (np.abs(dc) <= self.rmax) & (np.abs(dr) <= self.rmax)
        result = np.zeros_like(dc)
        if np.any(in_range):
            A = _make_design_matrix_cartesian(
                dc[in_range], dr[in_range],
                self.rmax, self.r_knots, self.row_knots, self.degree,
            )
            vals = A @ self.weights
            if self.log_fit:
                vals = 10.0 ** vals
            if clip_negative:
                vals = np.clip(vals, 0.0, None)
            result[in_range] = vals
        return result


def build_pixel_arrays(
    image: np.ndarray,
    dq: np.ndarray | None,
    selected_stars: pd.DataFrame,
    cutout_size: int,
    flux_col: str | None = None,
    background: str | None = "local",
    sigma_clip_bg: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble per-pixel offset and normalised-flux arrays from star cutouts.

    For each star in *selected_stars* a ``cutout_size × cutout_size`` stamp is
    extracted, background subtracted (optionally), normalised by the catalog
    flux estimate, and flattened.  All stars are concatenated into 1D arrays
    ready for :func:`fit_epsf`.

    The star-selection can be looser than for the photutils path: fainter and
    less isolated stars are acceptable because flux normalisation removes the
    brightness dependence.

    Parameters
    ----------
    image : np.ndarray
        2D background-subtracted science image (detector pixels).
    dq : np.ndarray or None
        2D data-quality array.  Bits ``DO_NOT_USE`` and ``SATURATED`` are
        masked when *roman-datamodels* is available; otherwise all non-zero
        DQ values are masked.
    selected_stars : pd.DataFrame
        Catalog rows with columns ``x_pix``, ``y_pix``, and a flux column
        (auto-detected from ``flux_col`` or common catalog column names).
    cutout_size : int
        Side length of each star stamp (pixels).
    flux_col : str or None
        Catalog column containing the flux normalisation value.  If ``None``,
        the first matching column among ``['segment_flux', 'psf_flux',
        'aper_flux', 'kron_flux', 'flux']`` is used.
    background : {'local', None}
        Per-cutout background mode:
        * ``'local'``: subtract a sigma-clipped median estimated from the
          outer annulus of each stamp.
        * ``None``: no per-cutout subtraction (assumes *image* is already
          background-free).
    sigma_clip_bg : float
        Sigma threshold for the local background clipping.  Default ``3.0``.

    Returns
    -------
    delta_col : np.ndarray
        Column offsets from star centre, shape ``(n_valid_stars, cutout_size²)``.
        Always fully populated (no NaN).
    delta_row : np.ndarray
        Row offsets from star centre, shape ``(n_valid_stars, cutout_size²)``.
        Always fully populated (no NaN).
    norm_flux : np.ndarray
        Normalised pixel fluxes ``pixel_flux / catalog_flux``, shape
        ``(n_valid_stars, cutout_size²)``.  Masked or bad pixels are ``NaN``.
    star_idx : np.ndarray
        1D integer array of shape ``(n_valid_stars,)`` mapping output row *i*
        to the row position of that star in *selected_stars*.
    """
    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape
    half = cutout_size // 2

    # --- resolve flux normalisation column ----------------------------------
    if flux_col is None:
        flux_col = next(
            (c for c in ["segment_flux", "psf_flux", "aper_flux", "kron_flux", "flux"]
             if c in selected_stars.columns),
            None,
        )
    if flux_col is None or flux_col not in selected_stars.columns:
        raise KeyError(
            f"Cannot find a flux column in selected_stars. "
            f"Available: {list(selected_stars.columns)}.  "
            f"Pass flux_col= explicitly."
        )

    # --- build global bad-pixel mask ----------------------------------------
    bad_global = ~np.isfinite(image)
    if dq is not None:
        dq_arr = np.asarray(dq)
        try:
            from roman_datamodels.dqflags import pixel

            bad_global |= (dq_arr & (pixel.DO_NOT_USE | pixel.SATURATED)) != 0
        except ImportError:
            bad_global |= dq_arr != 0

    all_dc, all_dr, all_nf, all_si = [], [], [], []

    for si, (_, row) in enumerate(selected_stars.iterrows()):
        x0 = float(row["x_pix"])
        y0 = float(row["y_pix"])
        flux_est = float(row[flux_col])

        if flux_est <= 0 or not np.isfinite(flux_est):
            continue

        # --- integer bounds -------------------------------------------------
        xc = int(np.round(x0))
        yc = int(np.round(y0))
        x_lo = xc - half
        x_hi = xc + half + 1
        y_lo = yc - half
        y_hi = yc + half + 1

        if x_lo < 0 or y_lo < 0 or x_hi > nx or y_hi > ny:
            continue

        stamp = image[y_lo:y_hi, x_lo:x_hi].copy()
        bad_stamp = bad_global[y_lo:y_hi, x_lo:x_hi]

        # --- per-cutout local background ------------------------------------
        if background == "local":
            border = np.zeros(stamp.shape, dtype=bool)
            border[0, :] = border[-1, :] = True
            border[:, 0] = border[:, -1] = True
            good_border = border & ~bad_stamp & np.isfinite(stamp)
            if np.any(good_border):
                _, med_bg, _ = sigma_clipped_stats(
                    stamp[good_border], sigma=sigma_clip_bg
                )
                stamp -= med_bg

        # --- pixel coordinate offsets (sub-pixel accurate) ------------------
        col_pix = np.arange(x_lo, x_hi)
        row_pix = np.arange(y_lo, y_hi)
        col_grid, row_grid = np.meshgrid(col_pix, row_pix)

        dc = (col_grid - x0).ravel()          # shape (n_pix,)
        dr = (row_grid - y0).ravel()          # shape (n_pix,)
        flux_flat = stamp.ravel() / flux_est  # normalised flux (n_pix,)
        bad_flat = bad_stamp.ravel()

        # Mark masked or non-finite pixels as NaN in norm_flux;
        # delta_col / delta_row are always the full grid (no NaN).
        nf = np.where(bad_flat | ~np.isfinite(flux_flat), np.nan, flux_flat)

        all_dc.append(dc)
        all_dr.append(dr)
        all_nf.append(nf)
        all_si.append(si)

    if len(all_dc) == 0:
        raise ValueError("No valid stars extracted from selected_stars.")

    delta_col = np.stack(all_dc)   # (n_valid_stars, n_pix)
    delta_row = np.stack(all_dr)   # (n_valid_stars, n_pix)
    norm_flux  = np.stack(all_nf)  # (n_valid_stars, n_pix) — NaN where bad
    star_idx   = np.array(all_si, dtype=int)  # (n_valid_stars,)

    n_valid = int(np.sum(np.isfinite(norm_flux)))
    log.info(
        "build_pixel_arrays: %d stars, %d total pixels (%d valid).",
        len(all_si), delta_col.size, n_valid,
    )
    return delta_col, delta_row, norm_flux, star_idx


def fit_epsf(
    delta_col: np.ndarray,
    delta_row: np.ndarray,
    norm_flux: np.ndarray,
    n_r_knots: int = 12,
    n_phi_knots: int = 15,
    rmin: float = 0.5,
    rmax: float = 13.0,
    cut_r: float = 3.0,
    degree: int = 3,
    prior_sigma: float | None = 1e3,
    norm_flux_err: np.ndarray | None = None,
    coords: str = "polar",
    n_col_knots: int | None = None,
    n_row_knots: int | None = None,
    log_fit: bool = True,
) -> PSFMachineModel:
    """Fit a B-spline ePSF model to normalised pixel data.

    Two coordinate systems are supported via *coords*:

    ``"polar"`` (default)
        Fits a tensor-product model in ``(r, phi)`` polar coordinates.  The
        radial basis uses clamped cubic B-splines on sqrt-spaced knots.  The
        azimuthal basis uses periodic wrapped cubic B-splines.  Inside *cut_r*
        only radial terms are active (no azimuthal modulation).

    ``"cartesian"``
        Fits a full 2D tensor-product model in ``(delta_col, delta_row)``.
        Knots are sqrt-spaced near the PSF core (within *cut_r*) for both
        axes, transitioning to linear spacing outside.  The design matrix is
        the outer product of two 1D B-spline bases.

    Inputs can be either 1D ``(n_pixels,)`` or 2D ``(n_stars, n_pixels_per_star)``
    arrays (the output shape of :func:`build_pixel_arrays`).  2D inputs are
    ravelled internally before fitting.

    Parameters
    ----------
    delta_col, delta_row : np.ndarray
        Pixel offsets from star centres.  Shape ``(n_pixels,)`` or
        ``(n_stars, n_pixels_per_star)``.
    norm_flux : np.ndarray
        Normalised pixel fluxes.  ``NaN`` values (masked pixels) are excluded
        automatically.  Same shape as *delta_col*.
    n_r_knots : int
        Polar: number of radial knots (sqrt-spaced in ``[rmin, rmax]``).
        Cartesian: number of knots per axis when *n_col_knots* / *n_row_knots*
        are not supplied.  Default ``12``.
    n_phi_knots : int
        Number of azimuthal knots (polar mode only).  Ignored for Cartesian.
        Default ``15``.
    rmin : float
        Inner radial limit in detector pixels (polar mode).  Default ``0.5``.
    rmax : float
        Outer radial limit (polar) or domain half-width (Cartesian) in
        detector pixels.  Default ``13.0``.
    cut_r : float
        Polar: radius inside which only radial terms are modelled.
        Cartesian: transition radius from sqrt-spaced to linear knots.
        Default ``3.0``.
    degree : int
        B-spline degree.  Default ``3`` (cubic).
    prior_sigma : float or None
        Standard deviation of the zero-mean Gaussian prior on weights.
        Regularises the normal equations.  Default ``1000`` (weak prior).
    norm_flux_err : np.ndarray or None
        Per-pixel uncertainties on *norm_flux* used as observation weights.
        Must have the same shape as *norm_flux*.
    coords : str
        Coordinate system: ``"polar"`` (default) or ``"cartesian"``.
    n_col_knots : int or None
        Cartesian: number of column-axis knots.  Defaults to *n_r_knots*.
    n_row_knots : int or None
        Cartesian: number of row-axis knots.  Defaults to *n_r_knots*.
    log_fit : bool
        When ``True`` (default), fit ``log₁₀(norm_flux)`` instead of raw flux.
        Improves numerical stability because normalised pixel fluxes are
        ``O(10⁻³)`` or smaller.  Pixels with ``norm_flux ≤ 0`` are excluded
        automatically.  :meth:`~PSFMachineModel.evaluate` transforms back to
        linear flux via ``10 ** (A @ weights)``.

    Returns
    -------
    PSFMachineModel
        Fitted model.  :attr:`~PSFMachineModel.coords` and
        :attr:`~PSFMachineModel.log_fit` record the fitting configuration so
        that :meth:`~PSFMachineModel.evaluate` and :func:`evaluate_epsf`
        automatically apply the correct transforms.
    """
    if coords not in ("polar", "cartesian"):
        raise ValueError(f"coords must be 'polar' or 'cartesian', got {coords!r}.")

    # Accept both 1D (n_pixels,) and 2D (n_stars, n_pixels_per_star) inputs
    dc = np.asarray(delta_col, dtype=float).ravel()
    dr = np.asarray(delta_row, dtype=float).ravel()
    nf = np.asarray(norm_flux, dtype=float).ravel()
    err_flat = (
        np.asarray(norm_flux_err, dtype=float).ravel()
        if norm_flux_err is not None else None
    )

    if coords == "polar":
        return _fit_epsf_polar(
            dc, dr, nf,
            n_r_knots=n_r_knots, n_phi_knots=n_phi_knots,
            rmin=rmin, rmax=rmax, cut_r=cut_r, degree=degree,
            prior_sigma=prior_sigma, norm_flux_err=err_flat,
            log_fit=log_fit,
        )

    # --- Cartesian branch ----------------------------------------------------
    _n_col = n_col_knots if n_col_knots is not None else n_r_knots
    _n_row = n_row_knots if n_row_knots is not None else n_r_knots
    return _fit_epsf_cartesian(
        dc, dr, nf,
        n_col_knots=_n_col, n_row_knots=_n_row,
        rmax=rmax, cut_r=cut_r, degree=degree,
        prior_sigma=prior_sigma, norm_flux_err=err_flat,
        log_fit=log_fit,
    )


def _fit_epsf_polar(
    dc: np.ndarray,
    dr: np.ndarray,
    nf: np.ndarray,
    n_r_knots: int,
    n_phi_knots: int,
    rmin: float,
    rmax: float,
    cut_r: float,
    degree: int,
    prior_sigma: float | None,
    norm_flux_err: np.ndarray | None,
    log_fit: bool,
) -> PSFMachineModel:
    r = np.hypot(dc, dr)
    phi = np.arctan2(dr, dc)

    in_range = (r >= rmin) & (r <= rmax) & np.isfinite(nf)
    if log_fit:
        in_range &= nf > 0
    if not np.any(in_range):
        raise ValueError(
            f"No pixels fall within rmin={rmin} – rmax={rmax} px range.  "
            "Check cutout_size and rmax."
        )

    r_knots = np.linspace(rmin ** 0.5, rmax ** 0.5, n_r_knots) ** 2
    y = np.log10(nf[in_range]) if log_fit else nf[in_range]

    A = _make_design_matrix_polar(
        r[in_range], phi[in_range], rmin, rmax, r_knots, n_phi_knots, cut_r, degree
    )
    err_fit = norm_flux_err[in_range] if norm_flux_err is not None else None
    weights = _solve_linear(A, y, y_err=err_fit, prior_sigma=prior_sigma)

    log.info(
        "fit_epsf (polar, log=%s): %d pixels, %d weights.",
        log_fit, int(np.sum(in_range)), len(weights),
    )
    return PSFMachineModel(
        weights=weights, r_knots=r_knots,
        n_r_knots=n_r_knots, n_phi_knots=n_phi_knots,
        rmin=rmin, rmax=rmax, cut_r=cut_r, degree=degree,
        n_pixels=int(np.sum(in_range)),
        log_fit=log_fit, coords="polar", row_knots=None,
    )


def _fit_epsf_cartesian(
    dc: np.ndarray,
    dr: np.ndarray,
    nf: np.ndarray,
    n_col_knots: int,
    n_row_knots: int,
    rmax: float,
    cut_r: float,
    degree: int,
    prior_sigma: float | None,
    norm_flux_err: np.ndarray | None,
    log_fit: bool,
) -> PSFMachineModel:
    in_range = (np.abs(dc) <= rmax) & (np.abs(dr) <= rmax) & np.isfinite(nf)
    if log_fit:
        in_range &= nf > 0
    if not np.any(in_range):
        raise ValueError(
            f"No pixels fall within the Cartesian domain [-{rmax}, {rmax}] px.  "
            "Check cutout_size and rmax."
        )

    col_knots = _cartesian_knots_1d(rmax, n_col_knots, cut_r)
    row_knots = _cartesian_knots_1d(rmax, n_row_knots, cut_r)
    y = np.log10(nf[in_range]) if log_fit else nf[in_range]

    A = _make_design_matrix_cartesian(
        dc[in_range], dr[in_range], rmax, col_knots, row_knots, degree
    )
    err_fit = norm_flux_err[in_range] if norm_flux_err is not None else None
    weights = _solve_linear(A, y, y_err=err_fit, prior_sigma=prior_sigma)

    log.info(
        "fit_epsf (cartesian, log=%s): %d pixels, %d weights.",
        log_fit, int(np.sum(in_range)), len(weights),
    )
    return PSFMachineModel(
        weights=weights, r_knots=col_knots,
        n_r_knots=n_col_knots, n_phi_knots=0,
        rmin=-rmax, rmax=rmax, cut_r=cut_r, degree=degree,
        n_pixels=int(np.sum(in_range)),
        log_fit=log_fit, coords="cartesian", row_knots=row_knots,
    )


def evaluate_epsf(
    model: PSFMachineModel,
    cutout_size: int = 25,
    oversampling: int = 4,
    x_0: float = 0.0,
    y_0: float = 0.0,
    clip_negative: bool = False,
    normalise: bool = True,
) -> np.ndarray:
    """Evaluate the ePSF model on an oversampled pixel grid.

    The grid is centred at ``(x_0, y_0)`` (sub-pixel offset from the nearest
    detector pixel centre) and spans ``cutout_size`` native pixels in each
    dimension sampled at ``oversampling`` times the native pixel spacing.

    Parameters
    ----------
    model : PSFMachineModel
        Fitted B-spline ePSF model.
    cutout_size : int
        Stamp size in native detector pixels.  Default ``25``.
    oversampling : int
        Oversampling factor (number of sub-pixels per native pixel).
        Default ``4``.
    x_0, y_0 : float
        Sub-pixel offset of the star centre from the nearest pixel centre
        (in native pixels).  Default ``(0, 0)``.
    clip_negative : bool
        Set negative model values to zero.  Default ``False`` (unnecessary
        when ``model.log_fit=True`` since ``10^x > 0``).
    normalise : bool
        Normalise the image so it sums to 1 on the native pixel grid (or
        to ``oversampling²`` on an oversampled grid before any renormalisation).
        Default ``True``.

    Returns
    -------
    np.ndarray
        2D oversampled ePSF image of shape
        ``(cutout_size * oversampling, cutout_size * oversampling)``.
    """
    half = cutout_size / 2.0
    n_out = cutout_size * oversampling

    # Sub-pixel positions in native detector pixels centred at (x_0, y_0)
    coords_1d = np.linspace(-half, half, n_out, endpoint=False) + 0.5 / oversampling
    dc_grid, dr_grid = np.meshgrid(coords_1d - x_0, coords_1d - y_0)

    psf_flat = model.evaluate(
        dc_grid.ravel(), dr_grid.ravel(), clip_negative=clip_negative
    )
    psf_img = psf_flat.reshape(n_out, n_out)

    if normalise:
        total = psf_img.sum()
        if total > 0:
            psf_img /= total

    return psf_img
