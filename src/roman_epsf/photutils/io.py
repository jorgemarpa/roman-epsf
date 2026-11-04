"""
roman_epsf.photutils.io
=======================
FITS serialization for photutils ePSF models.

* Single ``ImagePSF`` / ``EPSFModel`` → single ``PrimaryHDU`` FITS file.
* :class:`~roman_epsf.photutils.builder.GridEPSF` → multi-extension FITS file:
  ``PrimaryHDU`` (metadata) + one ``ImageHDU`` per cell + one ``BinTableHDU``
  with grid geometry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from photutils.psf import ImagePSF

from roman_epsf.photutils.builder import GridEPSF

log = logging.getLogger(__name__)

__all__ = ["save_epsf", "load_epsf"]

# Header keyword written to the PrimaryHDU to distinguish grid from single FITS
_GRID_FLAG = "EPSFGRID"


def save_epsf(
    model: ImagePSF | GridEPSF,
    filename: str | Path,
    extra_header: dict | None = None,
    overwrite: bool = True,
) -> Path:
    """Save a photutils ePSF model (single or grid) to a FITS file.

    Parameters
    ----------
    model : ImagePSF, EPSFModel, or GridEPSF
        The model to save.
    filename : str or Path
        Output FITS file path.
    extra_header : dict or None
        Additional keyword/value pairs written to the primary header (e.g.
        ``{'INSTRUME': 'WFI', 'FILTER': 'F146', 'DETECTOR': 'WFI04'}``).
    overwrite : bool
        Overwrite existing file.  Default ``True``.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    out_path = Path(filename)

    if isinstance(model, GridEPSF):
        hdul = _grid_to_hdul(model, extra_header)
    else:
        hdul = _single_to_hdul(model, extra_header)

    hdul.writeto(out_path, overwrite=overwrite)
    log.info("save_epsf: wrote %s", out_path)
    return out_path


def load_epsf(filename: str | Path) -> tuple[ImagePSF | GridEPSF, fits.Header]:
    """Load a photutils ePSF model from a FITS file.

    Automatically detects whether the file stores a single model or a grid.

    Parameters
    ----------
    filename : str or Path
        Path to a FITS file created by :func:`save_epsf`.

    Returns
    -------
    model : ImagePSF or GridEPSF
        Reconstructed ePSF model.
    header : astropy.io.fits.Header
        Primary FITS header containing saved metadata.
    """
    with fits.open(filename) as hdul:
        primary_hdr = hdul[0].header.copy()
        is_grid = bool(primary_hdr.get(_GRID_FLAG, False))

        if is_grid:
            model = _hdul_to_grid(hdul)
        else:
            model = _hdul_to_single(hdul)

    log.info("load_epsf: loaded %s (%s)", filename, type(model).__name__)
    return model, primary_hdr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_single_header(hdr: fits.Header, psf_model: ImagePSF) -> None:
    """Populate *hdr* with all parameters needed to reconstruct *psf_model*."""
    hdr["MODELCLS"] = (psf_model.__class__.__name__, "photutils PSF model class")

    ov = np.atleast_1d(psf_model.oversampling)
    ox, oy = (int(ov[0]), int(ov[0])) if len(ov) == 1 else (int(ov[0]), int(ov[1]))
    hdr["OVERSMPX"] = (ox, "PSF oversampling factor along X")
    hdr["OVERSMPY"] = (oy, "PSF oversampling factor along Y")

    if hasattr(psf_model, "origin") and psf_model.origin is not None:
        orig = np.atleast_1d(psf_model.origin)
        hdr["ORIGIN_X"] = (float(orig[0]), "PSF center X origin in data array")
        hdr["ORIGIN_Y"] = (float(orig[1]), "PSF center Y origin in data array")

    hdr["FLUX"] = (float(psf_model.flux.value), "Default model flux scaling factor")
    hdr["X_0"] = (float(psf_model.x_0.value), "Default model X center position")
    hdr["Y_0"] = (float(psf_model.y_0.value), "Default model Y center position")
    hdr["FIX_FLUX"] = (bool(psf_model.flux.fixed), "Whether flux parameter is fixed")
    hdr["FIX_X0"] = (bool(psf_model.x_0.fixed), "Whether x_0 parameter is fixed")
    hdr["FIX_Y0"] = (bool(psf_model.y_0.fixed), "Whether y_0 parameter is fixed")

    interp_kw = getattr(psf_model, "interp_kwargs", None)
    if interp_kw:
        if "kx" in interp_kw:
            hdr["INTERPKX"] = (int(interp_kw["kx"]), "Spline degree in X")
        if "ky" in interp_kw:
            hdr["INTERPKY"] = (int(interp_kw["ky"]), "Spline degree in Y")
        hdr["INTERPKW"] = (json.dumps(interp_kw), "JSON-encoded interp_kwargs")


def _read_single_header(hdr: fits.Header) -> ImagePSF:
    """Reconstruct an ``ImagePSF`` from its FITS header (data not included)."""
    # Called after data has been read; returns a partially constructed model —
    # caller must supply psf_data separately.
    raise NotImplementedError  # placeholder; see _hdul_to_single


def _single_to_hdul(psf_model: ImagePSF, extra_header: dict | None) -> fits.HDUList:
    hdr = fits.Header()
    hdr[_GRID_FLAG] = (False, "True if file stores a GridEPSF")
    _write_single_header(hdr, psf_model)

    if extra_header:
        for k, v in extra_header.items():
            hdr[k] = v

    psf_array = np.asarray(psf_model.data, dtype=np.float64)
    primary = fits.PrimaryHDU(data=psf_array, header=hdr)
    return fits.HDUList([primary])


def _hdul_to_single(hdul: fits.HDUList) -> ImagePSF:
    psf_data = np.asarray(hdul[0].data, dtype=np.float64).copy()
    hdr = hdul[0].header

    ox = int(hdr.get("OVERSMPX", 1))
    oy = int(hdr.get("OVERSMPY", ox))
    oversampling = ox if ox == oy else (ox, oy)

    origin = None
    if "ORIGIN_X" in hdr and "ORIGIN_Y" in hdr:
        origin = (float(hdr["ORIGIN_X"]), float(hdr["ORIGIN_Y"]))

    model = ImagePSF(
        data=psf_data,
        flux=float(hdr.get("FLUX", 1.0)),
        x_0=float(hdr.get("X_0", 0.0)),
        y_0=float(hdr.get("Y_0", 0.0)),
        oversampling=oversampling,
        origin=origin,
    )
    model.flux.fixed = bool(hdr.get("FIX_FLUX", False))
    model.x_0.fixed = bool(hdr.get("FIX_X0", False))
    model.y_0.fixed = bool(hdr.get("FIX_Y0", False))
    return model


def _grid_to_hdul(grid: GridEPSF, extra_header: dict | None) -> fits.HDUList:
    n_grid_y, n_grid_x = grid.data.shape[:2]

    primary_hdr = fits.Header()
    primary_hdr[_GRID_FLAG] = (True, "True if file stores a GridEPSF")
    primary_hdr["NGRIDX"] = (n_grid_x, "Number of grid cells along X")
    primary_hdr["NGRIDY"] = (n_grid_y, "Number of grid cells along Y")
    primary_hdr["OVERSAMP"] = (int(grid.oversampling), "ePSF oversampling factor")

    if extra_header:
        for k, v in extra_header.items():
            primary_hdr[k] = v

    # PrimaryHDU carries no image data — just metadata
    primary = fits.PrimaryHDU(header=primary_hdr)
    hdul = fits.HDUList([primary])

    # One ImageHDU per cell
    for iy in range(n_grid_y):
        for ix in range(n_grid_x):
            cell_hdr = fits.Header()
            cell_hdr["CELL_IY"] = (iy, "Grid cell row index (Y)")
            cell_hdr["CELL_IX"] = (ix, "Grid cell column index (X)")
            cell_hdr["XCENTER"] = (float(grid.x_centers[ix]), "Cell center X [detector px]")
            cell_hdr["YCENTER"] = (float(grid.y_centers[iy]), "Cell center Y [detector px]")
            cell_hdr["NSTARS"] = (int(grid.star_counts[iy, ix]), "Stars used in this cell")
            cell_hdr["OVERSMPX"] = (int(grid.oversampling), "ePSF oversampling factor along X")
            cell_hdr["OVERSMPY"] = (int(grid.oversampling), "ePSF oversampling factor along Y")

            # Copy model-level header keywords from the cell EPSFModel
            cell_model = grid.models[iy, ix]
            if cell_model is not None:
                _write_single_header(cell_hdr, cell_model)

            cell_data = np.asarray(grid.data[iy, ix], dtype=np.float64)
            hdul.append(fits.ImageHDU(data=cell_data, header=cell_hdr, name=f"CELL_{iy}_{ix}"))

    # BinTableHDU with grid geometry arrays
    from astropy.table import Table

    geom = Table()
    geom["x_centers"] = grid.x_centers
    geom["y_centers"] = grid.y_centers
    geom["x_edges"] = grid.x_edges[: len(grid.x_centers)]   # n_grid_x values
    geom["y_edges"] = grid.y_edges[: len(grid.y_centers)]   # n_grid_y values
    # Store edges as separate 1D columns for astropy Table compatibility
    geom_full = fits.BinTableHDU(
        data=fits.BinTableHDU.from_columns(
            [
                fits.Column(name="x_centers", format=f"{n_grid_x}D", array=[grid.x_centers]),
                fits.Column(name="y_centers", format=f"{n_grid_y}D", array=[grid.y_centers]),
                fits.Column(name="x_edges", format=f"{n_grid_x + 1}D", array=[grid.x_edges]),
                fits.Column(name="y_edges", format=f"{n_grid_y + 1}D", array=[grid.y_edges]),
                fits.Column(name="star_counts", format=f"{n_grid_y * n_grid_x}J",
                            array=[grid.star_counts.ravel()]),
            ]
        ).data,
        name="GRID_GEOM",
    )
    hdul.append(geom_full)
    return hdul


def _hdul_to_grid(hdul: fits.HDUList) -> GridEPSF:
    primary_hdr = hdul[0].header
    n_grid_x = int(primary_hdr["NGRIDX"])
    n_grid_y = int(primary_hdr["NGRIDY"])
    oversampling = int(primary_hdr["OVERSAMP"])

    # Read geometry from BinTableHDU
    geom_hdu = hdul["GRID_GEOM"]
    x_centers = np.asarray(geom_hdu.data["x_centers"][0], dtype=np.float64)
    y_centers = np.asarray(geom_hdu.data["y_centers"][0], dtype=np.float64)
    x_edges = np.asarray(geom_hdu.data["x_edges"][0], dtype=np.float64)
    y_edges = np.asarray(geom_hdu.data["y_edges"][0], dtype=np.float64)
    star_counts = np.asarray(geom_hdu.data["star_counts"][0], dtype=int).reshape(n_grid_y, n_grid_x)

    # Determine PSF array shape from the first cell HDU
    first_cell = hdul[f"CELL_0_0"]
    psf_h, psf_w = first_cell.data.shape

    epsf_data = np.zeros((n_grid_y, n_grid_x, psf_h, psf_w), dtype=np.float64)
    epsf_models = np.empty((n_grid_y, n_grid_x), dtype=object)

    for iy in range(n_grid_y):
        for ix in range(n_grid_x):
            cell_hdu = hdul[f"CELL_{iy}_{ix}"]
            cell_data = np.asarray(cell_hdu.data, dtype=np.float64).copy()
            epsf_data[iy, ix] = cell_data

            cell_hdr = cell_hdu.header
            ox = int(cell_hdr.get("OVERSMPX", oversampling))
            oy = int(cell_hdr.get("OVERSMPY", ox))
            ov = ox if ox == oy else (ox, oy)

            origin = None
            if "ORIGIN_X" in cell_hdr and "ORIGIN_Y" in cell_hdr:
                origin = (float(cell_hdr["ORIGIN_X"]), float(cell_hdr["ORIGIN_Y"]))

            cell_model = ImagePSF(
                data=cell_data,
                flux=float(cell_hdr.get("FLUX", 1.0)),
                x_0=float(cell_hdr.get("X_0", 0.0)),
                y_0=float(cell_hdr.get("Y_0", 0.0)),
                oversampling=ov,
                origin=origin,
            )
            epsf_models[iy, ix] = cell_model

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
