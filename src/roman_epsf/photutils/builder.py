"""
roman_epsf.photutils.builder
============================
ePSF construction functions that use ``photutils.psf.EPSFBuilder`` as the
fitting engine.  Star selection and cutout extraction delegate to the shared
:mod:`roman_epsf.star_selection` module so any future method module can reuse
the same preprocessing pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars
from astropy.stats import sigma_clipped_stats

from roman_epsf.star_selection import extract_star_cutouts, select_stars

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = ["GridEPSF", "build_global_epsf", "build_grid_epsf", "build_multiepoch_epsf"]


@dataclass
class GridEPSF:
    """Container for a spatially varying grid of ePSF models.

    Attributes
    ----------
    data : np.ndarray
        4D array of shape ``(n_grid_y, n_grid_x, psf_h, psf_w)`` holding the
        oversampled ePSF pixel data for every grid cell.
    models : np.ndarray
        Object array of shape ``(n_grid_y, n_grid_x)`` holding the
        ``photutils.psf.EPSFModel`` instance for each cell.
    x_centers : np.ndarray
        1D array of cell center X coordinates (detector pixels).
    y_centers : np.ndarray
        1D array of cell center Y coordinates (detector pixels).
    x_edges : np.ndarray
        1D array of cell boundary X coordinates (length ``n_grid_x + 1``).
    y_edges : np.ndarray
        1D array of cell boundary Y coordinates (length ``n_grid_y + 1``).
    oversampling : int
        ePSF oversampling factor.
    star_counts : np.ndarray
        2D int array ``(n_grid_y, n_grid_x)`` with the number of stars used in
        each cell.
    """

    data: np.ndarray
    models: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    oversampling: int
    star_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def build_global_epsf(
    image: np.ndarray,
    catalog: pd.DataFrame | Table,
    dq: np.ndarray | None = None,
    gwcs=None,
    cutout_size: int = 25,
    oversampling: int = 3,
    maxiters: int = 5,
    flux_percentile_range: tuple[float, float] = (60.0, 99.0),
    min_sep_factor: float = 1.2,
    snr_threshold: float = 5.0,
    min_snr_pixels: int = 5,
    background: str | None = "global",
    bkg2d_kwargs: dict | None = None,
    **epsf_kwargs,
):
    """Build a single global ePSF model from a detector image and source catalog.

    Parameters
    ----------
    image : np.ndarray
        2D science image (single Roman WFI detector exposure).
    catalog : pd.DataFrame or astropy.table.Table
        Source catalog associated with *image*.
    dq : np.ndarray or None
        2D data-quality array.
    gwcs : GWCS or None
        World-coordinate system; required only when *catalog* lacks pixel
        coordinate columns.
    cutout_size : int
        Star stamp size in pixels (odd preferred).  Default ``25``.
    oversampling : int
        EPSFBuilder oversampling factor.  Default ``3``.
    maxiters : int
        Maximum EPSFBuilder iterations.  Default ``5``.
    flux_percentile_range : (float, float)
        Flux percentile range for source brightness filtering.
    min_sep_factor : float
        Isolation criterion in units of *cutout_size*.
    snr_threshold : float
        Per-pixel SNR threshold for cutout validation.
    min_snr_pixels : int
        Minimum number of pixels above *snr_threshold* to keep a cutout.
    background : {'global', '2d', None}
        Background subtraction mode passed to :func:`~roman_epsf.star_selection.extract_star_cutouts`.
    bkg2d_kwargs : dict or None
        Extra keyword arguments for ``photutils.background.Background2D``
        (used when ``background='2d'``).
    **epsf_kwargs
        Additional keyword arguments forwarded to ``photutils.psf.EPSFBuilder``.

    Returns
    -------
    epsf_model : photutils.psf.EPSFModel
        Fitted global ePSF model.
    fitted_stars : photutils.psf.EPSFStars
        Star cutouts with updated centroid fits.
    selected_stars : pd.DataFrame
        Catalog rows used to build the ePSF.
    """
    selected = select_stars(
        catalog,
        image_shape=image.shape,
        cutout_size=cutout_size,
        flux_percentile_range=flux_percentile_range,
        min_sep_factor=min_sep_factor,
        gwcs=gwcs,
    )

    _, stars, filtered_df, _ = extract_star_cutouts(
        image,
        dq,
        selected,
        cutout_size=cutout_size,
        snr_threshold=snr_threshold,
        min_snr_pixels=min_snr_pixels,
        background=background,
        bkg2d_kwargs=bkg2d_kwargs,
    )

    builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, **epsf_kwargs)
    epsf_model, fitted_stars = builder(stars)

    log.info(
        "build_global_epsf: ePSF shape %s built from %d stars.",
        epsf_model.data.shape,
        len(stars),
    )
    return epsf_model, fitted_stars, filtered_df


def build_grid_epsf(
    image: np.ndarray,
    catalog: pd.DataFrame | Table,
    dq: np.ndarray | None = None,
    gwcs=None,
    n_grid_x: int = 4,
    n_grid_y: int = 4,
    min_stars_per_cell: int = 5,
    cutout_size: int = 25,
    oversampling: int = 3,
    maxiters: int = 5,
    flux_percentile_range: tuple[float, float] = (60.0, 99.0),
    min_sep_factor: float = 1.2,
    snr_threshold: float = 5.0,
    min_snr_pixels: int = 5,
    background: str | None = "global",
    bkg2d_kwargs: dict | None = None,
    **kwargs,
) -> GridEPSF:
    """Build a spatially varying grid of ePSF models across the detector.

    The detector is divided into an ``n_grid_y × n_grid_x`` grid.  An
    independent ePSF is fitted in each cell, warm-started from a global ePSF
    built across the full image.  Cells with fewer than *min_stars_per_cell*
    stars fall back to the global ePSF.

    Parameters
    ----------
    image : np.ndarray
        2D science image.
    catalog : pd.DataFrame or astropy.table.Table
        Source catalog.
    dq : np.ndarray or None
        2D data-quality array.
    gwcs : GWCS or None
        World-coordinate system for RA/Dec catalogs.
    n_grid_x : int
        Number of grid cells along the X axis.  Default ``4``.
    n_grid_y : int
        Number of grid cells along the Y axis.  Default ``4``.
    min_stars_per_cell : int
        Minimum stars required to build a cell ePSF.  Default ``5``.
    cutout_size : int
        Star stamp size in pixels.  Default ``25``.
    oversampling : int
        EPSFBuilder oversampling factor.  Default ``3``.
    maxiters : int
        Maximum EPSFBuilder iterations per cell.  Default ``5``.
    flux_percentile_range : (float, float)
        Flux percentile range for brightness filtering.
    min_sep_factor : float
        Isolation criterion in units of *cutout_size*.
    snr_threshold : float
        Per-pixel SNR threshold for cutout validation.
    min_snr_pixels : int
        Minimum valid pixels per cutout.
    background : {'global', '2d', None}
        Background subtraction mode.
    bkg2d_kwargs : dict or None
        Extra keyword arguments for ``Background2D``.
    **kwargs
        Forwarded to ``photutils.psf.EPSFBuilder`` for cell builders.

    Returns
    -------
    GridEPSF
        Dataclass containing the 4D ePSF array, per-cell models, grid geometry,
        and star counts.
    """
    # --- build global ePSF as warm-start for all cells -----------------------
    global_model, _, selected = build_global_epsf(
        image,
        catalog,
        dq=dq,
        gwcs=gwcs,
        cutout_size=cutout_size,
        oversampling=oversampling,
        maxiters=maxiters,
        flux_percentile_range=flux_percentile_range,
        min_sep_factor=min_sep_factor,
        snr_threshold=snr_threshold,
        min_snr_pixels=min_snr_pixels,
        background=background,
        bkg2d_kwargs=bkg2d_kwargs,
        **kwargs,
    )

    ny, nx = image.shape
    x_edges = np.linspace(0, nx, n_grid_x + 1)
    y_edges = np.linspace(0, ny, n_grid_y + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    psf_h, psf_w = global_model.data.shape
    epsf_data = np.zeros((n_grid_y, n_grid_x, psf_h, psf_w), dtype=np.float64)
    epsf_models = np.empty((n_grid_y, n_grid_x), dtype=object)
    star_counts = np.zeros((n_grid_y, n_grid_x), dtype=int)

    # Re-extract cutouts from the background-subtracted image so cell builders
    # operate on the same cleaned data.
    nddata, all_stars, _, _ = extract_star_cutouts(
        image,
        dq,
        selected,
        cutout_size=cutout_size,
        snr_threshold=snr_threshold,
        min_snr_pixels=min_snr_pixels,
        background=background,
        bkg2d_kwargs=bkg2d_kwargs,
    )

    sigma_clip = SigmaClip()
    cell_builder = EPSFBuilder(
        oversampling=oversampling,
        maxiters=maxiters,
        progress_bar=False,
        sigma_clip=sigma_clip,
        fit_shape=5,
        **kwargs,
    )

    for iy in range(n_grid_y):
        for ix in range(n_grid_x):
            in_cell = (
                (selected["x_pix"] >= x_edges[ix])
                & (selected["x_pix"] < x_edges[ix + 1])
                & (selected["y_pix"] >= y_edges[iy])
                & (selected["y_pix"] < y_edges[iy + 1])
            )
            cell_df = selected[in_cell]

            cell_tbl = Table()
            cell_tbl["x"] = cell_df["x_pix"].to_numpy()
            cell_tbl["y"] = cell_df["y_pix"].to_numpy()

            from photutils.psf import extract_stars as _extract

            cell_stars = _extract(nddata, cell_tbl, size=cutout_size)
            star_counts[iy, ix] = len(cell_stars)

            if len(cell_stars) >= min_stars_per_cell:
                cell_model, _ = cell_builder.build_epsf(cell_stars, epsf=global_model)
                ch, cw = cell_model.data.shape
                mh, mw = min(ch, psf_h), min(cw, psf_w)
                epsf_data[iy, ix, :mh, :mw] = cell_model.data[:mh, :mw]
                epsf_models[iy, ix] = cell_model
            else:
                log.warning(
                    "Cell (%d, %d) has only %d stars (< %d); using global ePSF.",
                    iy, ix, len(cell_stars), min_stars_per_cell,
                )
                epsf_data[iy, ix] = global_model.data.copy()
                epsf_models[iy, ix] = global_model

            log.info(
                "Cell (iy=%d, ix=%d) center=(%.0f, %.0f): %d stars",
                iy, ix, x_centers[ix], y_centers[iy], star_counts[iy, ix],
            )

    return GridEPSF(
        data=epsf_data,
        models=epsf_models,
        x_centers=x_centers,
        y_centers=y_centers,
        x_edges=x_edges,
        y_edges=y_edges,
        oversampling=oversampling,
        star_counts=star_counts,
    )


def build_multiepoch_epsf(
    cutout_list: list[dict],
    oversampling: int = 3,
    maxiters: int = 5,
    snr_threshold: float = 5.0,
    min_snr_pixels: int = 5,
    **epsf_kwargs,
):
    """Build an ePSF model from multi-epoch cutouts of a single star.

    Each epoch provides a slightly different sub-pixel sampling of the PSF.
    The cutouts are wrapped into ``photutils.psf.EPSFStar`` objects and
    combined by ``EPSFBuilder``.

    Parameters
    ----------
    cutout_list : list of dict
        Each dict must contain:

        * ``'data'``: 2D ``np.ndarray`` science cutout (background not yet
          subtracted; a local sigma-clipped median is removed per epoch).
        * ``'dq'``: 2D DQ array or ``None``.
        * ``'cutout_center'``: ``(x, y)`` star position within the cutout
          (0-indexed).
        * ``'origin'``: ``(x, y)`` cutout origin in detector pixel coordinates.
    oversampling : int
        EPSFBuilder oversampling factor.  Default ``3``.
    maxiters : int
        Maximum EPSFBuilder iterations.  Default ``5``.
    snr_threshold : float
        Minimum per-pixel SNR; epochs with fewer than *min_snr_pixels* valid
        pixels above this threshold are skipped.
    min_snr_pixels : int
        Minimum valid pixels above *snr_threshold* to include an epoch.
    **epsf_kwargs
        Additional keyword arguments forwarded to ``EPSFBuilder``.

    Returns
    -------
    epsf_model : photutils.psf.EPSFModel
        Fitted multi-epoch ePSF model.
    fitted_stars : photutils.psf.EPSFStars
        Per-epoch star cutouts with updated centroid fits.
    """
    epoch_stars = []

    for k, epoch in enumerate(cutout_list):
        sci = np.asarray(epoch["data"], dtype=np.float64).copy()
        dq_e = epoch.get("dq", None)
        cutout_center = epoch.get("cutout_center", None)
        origin = epoch.get("origin", (0, 0))

        # bad-pixel mask
        bad = ~np.isfinite(sci)
        if dq_e is not None:
            dq_arr = np.asarray(dq_e)
            try:
                from roman_datamodels.dqflags import pixel

                bad |= (dq_arr.astype(np.uint32) & (pixel.DO_NOT_USE | pixel.SATURATED)) != 0
            except ImportError:
                bad |= dq_arr != 0

        if np.all(bad):
            log.debug("Epoch %d: all pixels masked, skipping.", k)
            continue

        # per-epoch local background subtraction
        _, med_bg, std_bg = sigma_clipped_stats(sci[~bad], sigma=3.0)
        sci_clean = sci - med_bg
        sci_clean[bad] = 0.0
        weights = np.where(bad, 0.0, 1.0)

        # SNR filter
        if std_bg > 0:
            valid = weights > 0
            n_snr = int(np.sum(valid & ((sci_clean / std_bg) > snr_threshold)))
            if n_snr < min_snr_pixels:
                log.debug("Epoch %d: only %d SNR>%.1f pixels, skipping.", k, n_snr, snr_threshold)
                continue

        star = EPSFStar(
            sci_clean,
            weights=weights,
            cutout_center=tuple(cutout_center) if cutout_center is not None else None,
            origin=tuple(origin) if origin is not None else (0, 0),
            id_label=k,
        )
        epoch_stars.append(star)

    if len(epoch_stars) == 0:
        raise ValueError("No valid epoch cutouts passed the SNR filter.")

    multi_stars = EPSFStars(epoch_stars)
    builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, **epsf_kwargs)
    epsf_model, fitted_stars = builder(multi_stars)

    log.info(
        "build_multiepoch_epsf: ePSF shape %s built from %d/%d epochs.",
        epsf_model.data.shape,
        len(epoch_stars),
        len(cutout_list),
    )
    return epsf_model, fitted_stars
