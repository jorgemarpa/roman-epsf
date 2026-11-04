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
    _make_design_matrix_polar,
    _solve_linear,
)

log = logging.getLogger(__name__)

__all__ = ["PSFMachineModel", "build_pixel_arrays", "fit_epsf", "evaluate_epsf"]


@dataclass
class PSFMachineModel:
    """Container for a fitted polar B-spline ePSF model.

    All parameters stored here are sufficient to (a) evaluate the model at
    any ``(delta_col, delta_row)`` position and (b) save/load the model
    to/from a FITS file.

    Attributes
    ----------
    weights : np.ndarray
        Fitted B-spline coefficients.
    r_knots : np.ndarray
        Radial knot positions (sqrt-spaced in ``[rmin, rmax]``).
    n_r_knots : int
        Number of radial knots.
    n_phi_knots : int
        Number of azimuthal (angular) knots.
    rmin : float
        Inner radial limit in detector pixels.
    rmax : float
        Outer radial limit in detector pixels.
    cut_r : float
        Radius (pixels) below which only radial dependence is modelled.
    degree : int
        B-spline degree (default 3 = cubic).
    n_stars : int
        Number of stars used to build the model.
    n_pixels : int
        Total number of valid pixels used to fit the model.
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
    # lazy cache: design matrix is rebuilt on demand
    _A_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def evaluate(
        self,
        delta_col: np.ndarray,
        delta_row: np.ndarray,
        clip_negative: bool = True,
    ) -> np.ndarray:
        """Evaluate the ePSF model at arbitrary ``(delta_col, delta_row)`` offsets.

        Parameters
        ----------
        delta_col, delta_row : np.ndarray
            1D arrays of pixel offsets from the star centre (in detector pixels).
        clip_negative : bool
            Set negative model values to zero after evaluation.  Default ``True``.

        Returns
        -------
        np.ndarray
            Evaluated normalised flux at each supplied position.
        """
        dc = np.asarray(delta_col, dtype=float)
        dr = np.asarray(delta_row, dtype=float)
        r = np.hypot(dc, dr)
        phi = np.arctan2(dr, dc)

        in_range = (r >= self.rmin) & (r <= self.rmax)
        result = np.zeros_like(r)

        if np.any(in_range):
            A = _make_design_matrix_polar(
                r[in_range],
                phi[in_range],
                self.rmin,
                self.rmax,
                self.r_knots,
                self.n_phi_knots,
                self.cut_r,
                self.degree,
            )
            vals = A @ self.weights
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
        1D array of pixel column offsets ``col_pix - x_star`` [n_valid_pixels].
    delta_row : np.ndarray
        1D array of pixel row offsets ``row_pix - y_star`` [n_valid_pixels].
    norm_flux : np.ndarray
        1D array of normalised pixel fluxes ``pixel_flux / catalog_flux``
        [n_valid_pixels].
    star_idx : np.ndarray
        1D integer array mapping each valid pixel to its parent star index in
        *selected_stars* [n_valid_pixels].
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
            # Use outer annulus (border pixels) for background estimation
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
        col_pix = np.arange(x_lo, x_hi)   # detector column indices
        row_pix = np.arange(y_lo, y_hi)   # detector row indices
        col_grid, row_grid = np.meshgrid(col_pix, row_pix)

        dc = (col_grid - x0).ravel()
        dr = (row_grid - y0).ravel()
        flux_flat = stamp.ravel()
        bad_flat = bad_stamp.ravel()

        # --- quality filter: finite, not masked, positive stamp sum ---------
        valid = ~bad_flat & np.isfinite(flux_flat)
        if not np.any(valid):
            continue

        all_dc.append(dc[valid])
        all_dr.append(dr[valid])
        all_nf.append(flux_flat[valid] / flux_est)
        all_si.append(np.full(np.sum(valid), si, dtype=int))

    if len(all_dc) == 0:
        raise ValueError("No valid pixels extracted from selected_stars.")

    delta_col = np.concatenate(all_dc)
    delta_row = np.concatenate(all_dr)
    norm_flux = np.concatenate(all_nf)
    star_idx = np.concatenate(all_si)

    log.info(
        "build_pixel_arrays: %d valid pixels from %d stars.",
        len(delta_col),
        len(all_dc),
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
) -> PSFMachineModel:
    """Fit a polar B-spline ePSF model to normalised pixel data.

    The model is fitted as a linear system:  ``A(r, phi) @ w = norm_flux``
    where ``A`` is the polar design matrix and ``w`` are the spline weights.

    Parameters
    ----------
    delta_col, delta_row : np.ndarray
        Pixel offsets from star centres (output of :func:`build_pixel_arrays`).
    norm_flux : np.ndarray
        Normalised pixel fluxes (output of :func:`build_pixel_arrays`).
    n_r_knots : int
        Number of radial knots (sqrt-spaced between *rmin* and *rmax*).
        Default ``12``.
    n_phi_knots : int
        Number of azimuthal (angular) knots.  Default ``15``.
    rmin : float
        Inner radial limit in detector pixels.  Default ``0.5``.
    rmax : float
        Outer radial limit in detector pixels.  Default ``13.0``.
    cut_r : float
        Radius (pixels) inside which only radial variation is modelled.
        Default ``3.0``.
    degree : int
        B-spline degree.  Default ``3`` (cubic).
    prior_sigma : float or None
        Standard deviation of the zero-mean Gaussian prior on the weights.
        Acts as regularisation.  Default ``1000`` (weak prior).
    norm_flux_err : np.ndarray or None
        Per-pixel uncertainties on *norm_flux* used as observation weights.
        Shape must match *norm_flux*.

    Returns
    -------
    PSFMachineModel
        Fitted model with weights and all parameters required for evaluation
        and serialisation.
    """
    dc = np.asarray(delta_col, dtype=float)
    dr = np.asarray(delta_row, dtype=float)
    nf = np.asarray(norm_flux, dtype=float)

    # --- convert to polar ---------------------------------------------------
    r = np.hypot(dc, dr)
    phi = np.arctan2(dr, dc)

    # --- restrict to the radial fitting range --------------------------------
    in_range = (r >= rmin) & (r <= rmax) & np.isfinite(nf)
    if not np.any(in_range):
        raise ValueError(
            f"No pixels fall within rmin={rmin} – rmax={rmax} px range.  "
            "Check cutout_size and rmax."
        )

    r_fit = r[in_range]
    phi_fit = phi[in_range]
    nf_fit = nf[in_range]
    err_fit = norm_flux_err[in_range] if norm_flux_err is not None else None

    log.info("fit_epsf: fitting with %d pixels in [%.2f, %.2f] px.", len(r_fit), rmin, rmax)

    # --- radial knots (sqrt-spaced) -----------------------------------------
    r_knots = np.linspace(rmin ** 0.5, rmax ** 0.5, n_r_knots) ** 2

    # --- design matrix -------------------------------------------------------
    A = _make_design_matrix_polar(
        r_fit, phi_fit, rmin, rmax, r_knots, n_phi_knots, cut_r, degree
    )

    # --- solve linear system -------------------------------------------------
    weights = _solve_linear(A, nf_fit, y_err=err_fit, prior_sigma=prior_sigma)

    return PSFMachineModel(
        weights=weights,
        r_knots=r_knots,
        n_r_knots=n_r_knots,
        n_phi_knots=n_phi_knots,
        rmin=rmin,
        rmax=rmax,
        cut_r=cut_r,
        degree=degree,
        n_stars=0,        # caller can set this; unknown at fit time
        n_pixels=int(np.sum(in_range)),
    )


def evaluate_epsf(
    model: PSFMachineModel,
    cutout_size: int = 25,
    oversampling: int = 4,
    x_0: float = 0.0,
    y_0: float = 0.0,
    clip_negative: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    """Evaluate the ePSF model on an oversampled pixel grid.

    The grid is centred at ``(x_0, y_0)`` (sub-pixel offset from the nearest
    detector pixel centre) and spans ``cutout_size`` native pixels in each
    dimension sampled at ``oversampling`` times the native pixel spacing.

    Parameters
    ----------
    model : PSFMachineModel
        Fitted polar B-spline ePSF model.
    cutout_size : int
        Stamp size in native detector pixels.  Default ``25``.
    oversampling : int
        Oversampling factor (number of sub-pixels per native pixel).
        Default ``4``.
    x_0, y_0 : float
        Sub-pixel offset of the star centre from the nearest pixel centre
        (in native pixels).  Default ``(0, 0)``.
    clip_negative : bool
        Set negative model values to zero.  Default ``True``.
    normalise : bool
        Divide the output image by its sum so it integrates to 1.
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
