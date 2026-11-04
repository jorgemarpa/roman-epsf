"""
roman_epsf.star_selection
=========================
Method-agnostic helpers for selecting point sources from a Roman WFI source
catalog and extracting background-subtracted star cutouts suitable for any
ePSF construction method.

These utilities are intentionally decoupled from the ``photutils`` submodule
so that future ePSF method modules (e.g. WebbPSF, pixel-convolution) can
reuse the same star-selection and extraction logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.nddata import NDData
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from photutils.psf import EPSFStars, extract_stars

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = ["select_stars", "extract_star_cutouts"]


def select_stars(
    cat_df: pd.DataFrame | Table,
    image_shape: tuple[int, int],
    cutout_size: int,
    flux_percentile_range: tuple[float, float] = (60.0, 99.0),
    min_sep_factor: float = 1.2,
    gwcs=None,
) -> pd.DataFrame:
    """Select isolated, bright, unflagged point sources from a Roman source catalog.

    Parameters
    ----------
    cat_df : pd.DataFrame or astropy.table.Table
        Source catalog. Pixel coordinates are resolved by trying column name
        variants ``x_centroid/y_centroid``, ``xcentroid/ycentroid``, ``x/y``,
        or by converting ``ra/dec`` via *gwcs*.
    image_shape : (int, int)
        Detector image shape ``(ny, nx)`` used to enforce boundary margins.
    cutout_size : int
        Stamp size in pixels; determines the edge buffer and isolation radius.
    flux_percentile_range : (float, float)
        Lower and upper percentile of the flux distribution used to retain
        bright but unsaturated sources.  Default ``(60, 99)``.
    min_sep_factor : float
        Minimum separation between any two sources in units of *cutout_size*.
        Default ``1.2``.
    gwcs : GWCS or None
        World-coordinate system used to convert RA/Dec to pixel coordinates
        when the catalog lacks pixel columns.

    Returns
    -------
    pd.DataFrame
        Filtered catalog with additional columns ``x_pix`` and ``y_pix``.
    """
    if isinstance(cat_df, Table):
        cat_df = cat_df.to_pandas()
    else:
        cat_df = cat_df.copy()

    # --- resolve pixel coordinates -------------------------------------------
    if {"x_centroid", "y_centroid"}.issubset(cat_df.columns):
        cat_df["x_pix"] = cat_df["x_centroid"].to_numpy()
        cat_df["y_pix"] = cat_df["y_centroid"].to_numpy()
    elif {"xcentroid", "ycentroid"}.issubset(cat_df.columns):
        cat_df["x_pix"] = cat_df["xcentroid"].to_numpy()
        cat_df["y_pix"] = cat_df["ycentroid"].to_numpy()
    elif {"x", "y"}.issubset(cat_df.columns):
        cat_df["x_pix"] = cat_df["x"].to_numpy()
        cat_df["y_pix"] = cat_df["y"].to_numpy()
    elif {"ra", "dec"}.issubset(cat_df.columns):
        if gwcs is None:
            raise ValueError("Catalog has only ra/dec columns but gwcs=None.")
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(
            ra=cat_df["ra"].to_numpy() * u.deg,
            dec=cat_df["dec"].to_numpy() * u.deg,
        )
        x_pix, y_pix = gwcs.world_to_pixel(coords)
        cat_df["x_pix"] = x_pix
        cat_df["y_pix"] = y_pix
    else:
        raise KeyError(
            f"Cannot resolve pixel coordinates from catalog columns: {list(cat_df.columns)}"
        )

    ny, nx = image_shape
    half_size = (cutout_size - 1) // 2
    edge_buffer = half_size + 5

    # --- boundary mask -------------------------------------------------------
    mask = (
        np.isfinite(cat_df["x_pix"])
        & np.isfinite(cat_df["y_pix"])
        & (cat_df["x_pix"] > edge_buffer)
        & (cat_df["x_pix"] < (nx - 1 - edge_buffer))
        & (cat_df["y_pix"] > edge_buffer)
        & (cat_df["y_pix"] < (ny - 1 - edge_buffer))
    )

    # --- catalog quality flags / morphology ----------------------------------
    if "flags" in cat_df.columns:
        mask &= cat_df["flags"] == 0
    if "is_extended" in cat_df.columns:
        mask &= ~cat_df["is_extended"].astype(bool)
    if "sharpness" in cat_df.columns:
        mask &= cat_df["sharpness"].between(0.3, 1.01)
    if "roundness1" in cat_df.columns:
        mask &= cat_df["roundness1"].between(-0.5, 0.5)

    # --- flux percentile filter ----------------------------------------------
    flux_col = next(
        (c for c in ["segment_flux", "psf_flux", "aper_flux", "flux", "kron_flux"] if c in cat_df.columns),
        None,
    )
    if flux_col is not None:
        f_vals = cat_df.loc[mask, flux_col].dropna()
        if len(f_vals) > 0:
            f_low, f_high = np.nanpercentile(f_vals, list(flux_percentile_range))
            mask &= cat_df[flux_col].between(f_low, f_high)

    candidates = cat_df.loc[mask].copy()

    # --- isolation criterion -------------------------------------------------
    min_sep = cutout_size * min_sep_factor
    cx = candidates["x_pix"].to_numpy()
    cy = candidates["y_pix"].to_numpy()
    ax = cat_df["x_pix"].to_numpy()
    ay = cat_df["y_pix"].to_numpy()

    dx = cx[:, None] - ax[None, :]
    dy = cy[:, None] - ay[None, :]
    dist = np.hypot(dx, dy)
    dist[dist == 0.0] = np.inf
    isolated = np.nanmin(dist, axis=1) > min_sep

    selected = candidates.loc[isolated].copy()
    log.info("select_stars: %d isolated, bright stars selected.", len(selected))
    return selected


def extract_star_cutouts(
    image: np.ndarray,
    dq: np.ndarray | None,
    selected_stars: pd.DataFrame,
    cutout_size: int,
    snr_threshold: float = 5.0,
    min_snr_pixels: int = 5,
    background: str | None = "global",
    bkg2d_kwargs: dict | None = None,
) -> tuple[NDData, EPSFStars, pd.DataFrame, float]:
    """Extract background-subtracted star cutouts for ePSF construction.

    Parameters
    ----------
    image : np.ndarray
        2D detector science image.
    dq : np.ndarray or None
        2D data-quality array.  Bits ``DO_NOT_USE`` and ``SATURATED`` are
        masked when *roman-datamodels* is installed; otherwise all non-zero
        DQ values are masked.
    selected_stars : pd.DataFrame
        Star catalog with ``x_pix`` and ``y_pix`` columns (output of
        :func:`select_stars`).
    cutout_size : int
        Stamp size in pixels (should be odd for ``EPSFBuilder``).
    snr_threshold : float
        Minimum per-pixel SNR for a cutout to be considered valid.
    min_snr_pixels : int
        Minimum number of pixels above *snr_threshold* required to keep a
        cutout.
    background : {'global', '2d', None}
        Background subtraction mode:

        * ``'global'`` – subtract a single sigma-clipped median scalar
          (``astropy.stats.sigma_clipped_stats``).
        * ``'2d'`` – fit and subtract a 2D background surface
          (``photutils.background.Background2D``).  Pass *bkg2d_kwargs* to
          control ``box_size``, ``filter_size``, ``sigma_clip``, etc.
        * ``None`` / ``'none'`` – no background subtraction.
    bkg2d_kwargs : dict or None
        Keyword arguments forwarded to ``photutils.background.Background2D``
        when ``background='2d'``.  Ignored for other modes.

    Returns
    -------
    nddata : astropy.nddata.NDData
        Background-subtracted, bad-pixel-masked image wrapped for
        ``photutils.psf.extract_stars``.
    stars : photutils.psf.EPSFStars
        Valid star cutouts ready for ``EPSFBuilder``.
    filtered_df : pd.DataFrame
        Subset of *selected_stars* corresponding to the returned cutouts.
    background_std : float
        RMS of the background estimate (scalar sigma for ``'global'``,
        median RMS for ``'2d'``, or NaN when background is ``None``).
    """
    image = np.asarray(image, dtype=np.float64)

    # --- build bad-pixel mask ------------------------------------------------
    bad_mask = ~np.isfinite(image)
    if dq is not None:
        dq_arr = np.asarray(dq)
        try:
            from roman_datamodels.dqflags import pixel

            bad_mask |= (dq_arr & (pixel.DO_NOT_USE | pixel.SATURATED)) != 0
        except ImportError:
            bad_mask |= dq_arr != 0

    # --- background subtraction ----------------------------------------------
    bg_mode = (background or "none").lower()

    if bg_mode == "global":
        _, median_val, std_val = sigma_clipped_stats(image, mask=bad_mask, sigma=3.0)
        data_bkgsub = image - median_val
        background_std = float(std_val)

    elif bg_mode == "2d":
        from photutils.background import Background2D, MedianBackground

        kw = dict(box_size=64, filter_size=3, sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground())
        if bkg2d_kwargs:
            kw.update(bkg2d_kwargs)
        bkg = Background2D(image, mask=bad_mask, **kw)
        data_bkgsub = image - bkg.background
        background_std = float(np.nanmedian(bkg.background_rms))

    elif bg_mode in ("none", "null"):
        data_bkgsub = image.copy()
        background_std = float("nan")

    else:
        raise ValueError(
            f"background={background!r} is not valid. Choose 'global', '2d', or None."
        )

    data_clean = np.where(bad_mask, 0.0, data_bkgsub)
    nddata = NDData(data=data_clean, mask=bad_mask)

    # --- extract all candidate cutouts --------------------------------------
    stars_tbl = Table()
    stars_tbl["x"] = selected_stars["x_pix"].to_numpy()
    stars_tbl["y"] = selected_stars["y_pix"].to_numpy()

    candidate_stars = extract_stars(nddata, stars_tbl, size=cutout_size)

    # --- filter low-SNR / empty cutouts -------------------------------------
    valid_mask = []
    for star in candidate_stars:
        valid_pix = (star.weights > 0) & np.isfinite(star.data)
        if not np.any(valid_pix):
            valid_mask.append(False)
            continue
        # Use background_std for SNR; fall back to per-cutout std when not available
        std_ref = background_std if np.isfinite(background_std) else float(np.nanstd(star.data[valid_pix]))
        if std_ref == 0.0:
            valid_mask.append(False)
            continue
        snr_map = star.data / std_ref
        n_snr = int(np.sum(valid_pix & (snr_map > snr_threshold)))
        positive = float(np.sum(star.data[valid_pix])) > 0.0
        valid_mask.append(n_snr >= min_snr_pixels and positive)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    n_removed = int(np.sum(~valid_mask))

    filtered_df = selected_stars.loc[valid_mask].copy()
    stars_tbl_valid = stars_tbl[valid_mask]
    stars = extract_stars(nddata, stars_tbl_valid, size=cutout_size)

    log.info(
        "extract_star_cutouts: %d valid cutouts extracted (%d removed, SNR threshold=%.1f).",
        len(stars),
        n_removed,
        snr_threshold,
    )
    return nddata, stars, filtered_df, background_std
