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
from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars, ImagePSF

try:
    from photutils.psf import EPSFModel
except ImportError:
    EPSFModel = ImagePSF  # In photutils >= 2.0, EPSFModel is superseded by ImagePSF
from astropy.stats import sigma_clipped_stats

from roman_epsf.star_selection import extract_star_cutouts, select_stars

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = [
    "GridEPSF",
    "build_global_epsf",
    "build_grid_epsf",
    "build_multiepoch_epsf",
    "create_stpsf_epsf",
]


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


def create_stpsf_epsf(
    detector: str | int = "WFI01",
    filter_name: str = "F146",
    oversampling: int = 3,
    cutout_size: int = 25,
    fov_pixels: int | None = None,
) -> ImagePSF:
    """Generate an oversampled photutils ImagePSF model using stpsf for Roman WFI.

    Parameters
    ----------
    detector : str or int
        Detector name (e.g. 'WFI01', 'wfi01', or 1). Default 'WFI01'.
    filter_name : str
        Filter name (e.g. 'F146'). Default 'F146'.
    oversampling : int
        Oversampling factor of the model. Default 3.
    cutout_size : int
        Size of star cutouts in detector pixels (default 25).
    fov_pixels : int or None
        Field of view in detector pixels for stpsf.calc_psf. Defaults to *cutout_size*.

    Returns
    -------
    ImagePSF
        Normalized photutils ImagePSF model matching the grid expected by EPSFBuilder.
    """
    try:
        from stpsf import roman
    except ImportError as err:
        raise ImportError(
            "The 'stpsf' package is required to generate an initial PSF model from stpsf. "
            "Please install it (e.g. 'pip install stpsf') and ensure STPSF data files are configured."
        ) from err

    # Normalize detector string
    if isinstance(detector, int):
        det_str = f"WFI{detector:02d}"
    elif isinstance(detector, str) and detector.isdigit():
        det_str = f"WFI{int(detector):02d}"
    else:
        det_str = str(detector).upper()
        if not det_str.startswith("WFI"):
            det_str = f"WFI{det_str}"

    filt_str = str(filter_name).upper()
    fov = int(cutout_size) if fov_pixels is None else int(fov_pixels)

    wfi = roman.WFI()
    wfi.detector = det_str
    wfi.filter = filt_str

    # Compute PSF via stpsf
    hdul = wfi.calc_psf(fov_pixels=fov, oversample=oversampling)

    # Prefer oversampled distorted PSF (extension 'OVERDIST') if present, else PrimaryHDU
    if "OVERDIST" in hdul:
        psf_data = np.asarray(hdul["OVERDIST"].data, dtype=np.float64).copy()
    else:
        psf_data = np.asarray(hdul[0].data, dtype=np.float64).copy()

    # The expected grid shape from photutils.psf.EPSFBuilder is:
    # cutout_size * oversampling (rounded up to odd if even)
    target_size = cutout_size * oversampling
    if target_size % 2 == 0:
        target_size += 1
    ny, nx = psf_data.shape

    # Center-crop or pad along Y
    if ny > target_size:
        dy = (ny - target_size) // 2
        psf_data = psf_data[dy : dy + target_size, :]
    elif ny < target_size:
        pad_y = target_size - ny
        py1 = pad_y // 2
        py2 = pad_y - py1
        psf_data = np.pad(psf_data, ((py1, py2), (0, 0)), mode="constant", constant_values=0.0)

    # Center-crop or pad along X
    ny, nx = psf_data.shape
    if nx > target_size:
        dx = (nx - target_size) // 2
        psf_data = psf_data[:, dx : dx + target_size]
    elif nx < target_size:
        pad_x = target_size - nx
        px1 = pad_x // 2
        px2 = pad_x - px1
        psf_data = np.pad(psf_data, ((0, 0), (px1, px2)), mode="constant", constant_values=0.0)

    # Normalize total flux to 1.0
    total = np.sum(psf_data)
    if total > 0:
        psf_data /= total

    origin = ((psf_data.shape[1] - 1) / 2.0, (psf_data.shape[0] - 1) / 2.0)

    model = ImagePSF(
        data=psf_data,
        flux=1.0,
        x_0=0.0,
        y_0=0.0,
        oversampling=oversampling,
        origin=origin,
    )
    return model


def _resolve_detector_and_filter(
    catalog: pd.DataFrame | Table,
    detector: str | int | None = None,
    filter_name: str | None = None,
) -> tuple[str, str]:
    """Resolve detector and filter names from explicit arguments or catalog metadata."""
    resolved_det = detector
    resolved_filt = filter_name

    meta = getattr(catalog, "meta", None)
    if isinstance(meta, dict):
        if resolved_det is None:
            resolved_det = (
                meta.get("detector")
                or meta.get("DETECTOR")
                or (meta.get("instrument", {}).get("detector") if isinstance(meta.get("instrument"), dict) else None)
            )
        if resolved_filt is None:
            resolved_filt = (
                meta.get("filter")
                or meta.get("FILTER")
                or (meta.get("instrument", {}).get("optical_element") if isinstance(meta.get("instrument"), dict) else None)
            )

    if resolved_det is None:
        resolved_det = "WFI01"
        log.info("No detector specified or found in catalog metadata; defaulting to 'WFI01'.")
    elif isinstance(resolved_det, int):
        resolved_det = f"WFI{resolved_det:02d}"
    elif isinstance(resolved_det, str) and resolved_det.isdigit():
        resolved_det = f"WFI{int(resolved_det):02d}"
    else:
        resolved_det = str(resolved_det).upper()

    if resolved_filt is None:
        resolved_filt = "F146"
        log.info("No filter specified or found in catalog metadata; defaulting to 'F146'.")
    else:
        resolved_filt = str(resolved_filt).upper()

    return resolved_det, resolved_filt


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
    initial_epsf: bool | ImagePSF | EPSFModel = False,
    detector: str | int | None = None,
    filter_name: str | None = None,
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
    initial_epsf : bool, ImagePSF, or EPSFModel, optional
        Initial ePSF model used to warm-start the fitting:
        - ``False`` (default): Build ePSF empirically from extracted stars.
        - ``True``: Automatically generate an initial ePSF model using ``stpsf``
          for the specified *detector* and *filter_name*.
        - ``ImagePSF`` or ``EPSFModel``: User-provided model to warm-start fitting.
    detector : str or int, optional
        Detector name (e.g. 'WFI01' or 1). Used when ``initial_epsf=True``.
        If omitted, resolved from ``catalog.meta`` or defaults to 'WFI01'.
    filter_name : str, optional
        Filter name (e.g. 'F146'). Used when ``initial_epsf=True``.
        If omitted, resolved from ``catalog.meta`` or defaults to 'F146'.
    **epsf_kwargs
        Additional keyword arguments forwarded to ``photutils.psf.EPSFBuilder``.

    Returns
    -------
    epsf_model : photutils.psf.EPSFModel or photutils.psf.ImagePSF
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

    init_model: ImagePSF | EPSFModel | None = None
    if initial_epsf is False or initial_epsf is None:
        init_model = None
    elif initial_epsf is True:
        det, filt = _resolve_detector_and_filter(catalog, detector, filter_name)
        log.info(
            "build_global_epsf: generating initial ePSF using stpsf for %s, %s (cutout_size=%d, oversampling=%d)",
            det,
            filt,
            cutout_size,
            oversampling,
        )
        init_model = create_stpsf_epsf(
            detector=det,
            filter_name=filt,
            oversampling=oversampling,
            cutout_size=cutout_size,
        )
    elif isinstance(initial_epsf, (ImagePSF, EPSFModel)):
        init_model = initial_epsf
    else:
        raise TypeError(
            f"initial_epsf must be False, True, or an ImagePSF/EPSFModel instance; got {type(initial_epsf).__name__}"
        )

    builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, **epsf_kwargs)
    epsf_model, fitted_stars = builder.build_epsf(stars, epsf=init_model)

    log.info(
        "build_global_epsf: ePSF shape %s built from %d stars (initial_epsf=%s).",
        epsf_model.data.shape,
        len(stars),
        bool(initial_epsf),
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
    initial_epsf: bool | ImagePSF | EPSFModel = False,
    detector: str | int | None = None,
    filter_name: str | None = None,
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
    initial_epsf : bool, ImagePSF, or EPSFModel, optional
        Initial ePSF model used to warm-start the global and cell fitting.
    detector : str or int, optional
        Detector name. Used when ``initial_epsf=True``.
    filter_name : str, optional
        Filter name. Used when ``initial_epsf=True``.
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
        initial_epsf=initial_epsf,
        detector=detector,
        filter_name=filter_name,
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
