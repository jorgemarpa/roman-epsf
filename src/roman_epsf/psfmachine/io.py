"""
roman_epsf.psfmachine.io
=========================
FITS serialisation for :class:`~roman_epsf.psfmachine.builder.PSFMachineModel`.

FITS layout
-----------
``save_model`` writes a two-extension FITS file:

* ``PrimaryHDU`` — zero-size data; header holds all scalar parameters
  (``RMIN``, ``RMAX``, ``CUT_R``, ``NRKNOTS``, ``NPHIKNOTS``, ``DEGREE``,
  ``NSTARS``, ``NPIXELS``) plus any caller-supplied extra keywords.
* ``ImageHDU  WEIGHTS`` — 1D float64 array of fitted spline coefficients.
* ``ImageHDU  R_KNOTS`` — 1D float64 array of radial knot positions.

``save_image`` evaluates the model on an oversampled grid and writes a plain
2D ``ImageHDU`` with oversampling metadata in the header.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from roman_epsf.psfmachine.builder import PSFMachineModel, evaluate_epsf

log = logging.getLogger(__name__)

__all__ = ["save_model", "load_model", "save_image"]


def save_model(
    model: PSFMachineModel,
    filename: str | Path,
    extra_header: dict | None = None,
    overwrite: bool = True,
) -> Path:
    """Save a :class:`PSFMachineModel` to a FITS file.

    Parameters
    ----------
    model : PSFMachineModel
        Fitted polar B-spline ePSF model.
    filename : str or Path
        Output FITS file path.
    extra_header : dict or None
        Additional keyword/value pairs written to the primary header (e.g.
        ``{'INSTRUME': 'WFI', 'FILTER': 'F146', 'DETECTOR': 'WFI04'}``).
    overwrite : bool
        Overwrite an existing file.  Default ``True``.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    out = Path(filename)

    # --- Primary HDU: metadata only ----------------------------------------
    phdr = fits.Header()
    coords = getattr(model, "coords", "polar")

    phdr["EPSFMETH"] = ("psfmachine", "ePSF method module")
    phdr["COORDS"] = (coords, "Coordinate system: polar or cartesian")
    phdr["RMIN"] = (float(model.rmin), "Inner radial limit / domain start [native pixels]")
    phdr["RMAX"] = (float(model.rmax), "Outer radial limit / domain half-width [native pixels]")
    phdr["CUT_R"] = (float(model.cut_r), "Transition radius [native pixels]")
    phdr["NRKNOTS"] = (int(model.n_r_knots), "Number of radial / column knots")
    phdr["NPHIKNTS"] = (int(model.n_phi_knots), "Number of azimuthal knots (0 = cartesian)")
    phdr["DEGREE"] = (int(model.degree), "B-spline degree")
    phdr["NSTARS"] = (int(model.n_stars), "Stars used to build model")
    phdr["NPIXELS"] = (int(model.n_pixels), "Pixels used to fit model")
    phdr["LOGFIT"] = (bool(model.log_fit), "True if model fitted in log10(flux) space")
    if getattr(model, "cutout_size", None) is not None:
        phdr["CUTSIZE"] = (int(model.cutout_size), "Stamp size [native pixels]")

    if extra_header:
        for k, v in extra_header.items():
            phdr[k] = v

    primary = fits.PrimaryHDU(header=phdr)

    # --- Extension 1: spline weights ----------------------------------------
    w_hdr = fits.Header()
    w_hdr["EXTNAME"] = "WEIGHTS"
    w_hdr["COMMENT"] = "Fitted B-spline coefficients"
    weights_hdu = fits.ImageHDU(
        data=np.asarray(model.weights, dtype=np.float64), header=w_hdr
    )

    if coords == "cartesian":
        # Cartesian: save separate column and row knot arrays
        ck_hdr = fits.Header()
        ck_hdr["EXTNAME"] = "COL_KNOTS"
        ck_hdr["COMMENT"] = "Column-axis knot positions [native pixels]"
        col_knots_hdu = fits.ImageHDU(
            data=np.asarray(model.r_knots, dtype=np.float64), header=ck_hdr
        )
        rk_hdr = fits.Header()
        rk_hdr["EXTNAME"] = "ROW_KNOTS"
        rk_hdr["COMMENT"] = "Row-axis knot positions [native pixels]"
        row_knots_hdu = fits.ImageHDU(
            data=np.asarray(model.row_knots, dtype=np.float64), header=rk_hdr
        )
        fits.HDUList([primary, weights_hdu, col_knots_hdu, row_knots_hdu]).writeto(
            out, overwrite=overwrite
        )
    else:
        # Polar: save radial knots only
        k_hdr = fits.Header()
        k_hdr["EXTNAME"] = "R_KNOTS"
        k_hdr["COMMENT"] = "Radial knot positions [native pixels]"
        knots_hdu = fits.ImageHDU(
            data=np.asarray(model.r_knots, dtype=np.float64), header=k_hdr
        )
        fits.HDUList([primary, weights_hdu, knots_hdu]).writeto(out, overwrite=overwrite)

    log.info("save_model: wrote %s (coords=%s, weights shape %s)", out, coords, model.weights.shape)
    return out


def load_model(filename: str | Path) -> tuple[PSFMachineModel, fits.Header]:
    """Load a :class:`PSFMachineModel` from a FITS file created by :func:`save_model`.

    Parameters
    ----------
    filename : str or Path
        Path to the FITS file.

    Returns
    -------
    model : PSFMachineModel
        Reconstructed polar B-spline ePSF model.
    header : astropy.io.fits.Header
        Primary FITS header containing saved metadata.
    """
    with fits.open(filename) as hdul:
        phdr = hdul[0].header.copy()
        weights = np.asarray(hdul["WEIGHTS"].data, dtype=np.float64).copy()

        coords = str(phdr.get("COORDS", "polar")).strip().lower()

        if coords == "cartesian":
            col_knots = np.asarray(hdul["COL_KNOTS"].data, dtype=np.float64).copy()
            row_knots = np.asarray(hdul["ROW_KNOTS"].data, dtype=np.float64).copy()
            r_knots = col_knots
        else:
            r_knots = np.asarray(hdul["R_KNOTS"].data, dtype=np.float64).copy()
            col_knots = r_knots
            row_knots = None

    log_fit = bool(phdr.get("LOGFIT", False))
    cutout_size = int(phdr["CUTSIZE"]) if "CUTSIZE" in phdr else None

    model = PSFMachineModel(
        weights=weights,
        r_knots=col_knots,
        n_r_knots=int(phdr["NRKNOTS"]),
        n_phi_knots=int(phdr["NPHIKNTS"]),
        rmin=float(phdr["RMIN"]),
        rmax=float(phdr["RMAX"]),
        cut_r=float(phdr["CUT_R"]),
        degree=int(phdr["DEGREE"]),
        n_stars=int(phdr.get("NSTARS", 0)),
        n_pixels=int(phdr.get("NPIXELS", 0)),
        log_fit=log_fit,
        coords=coords,
        row_knots=row_knots,
        cutout_size=cutout_size,
    )

    log.info(
        "load_model: loaded %s (coords=%s, weights shape %s)",
        filename, coords, weights.shape,
    )
    return model, phdr


def save_image(
    model: PSFMachineModel,
    filename: str | Path,
    cutout_size: int | None = None,
    oversampling: int = 4,
    x_0: float = 0.0,
    y_0: float = 0.0,
    clip_negative: bool = True,
    normalise: bool = True,
    extra_header: dict | None = None,
    overwrite: bool = True,
) -> Path:
    """Evaluate the ePSF model on an oversampled grid and save as a FITS image.

    The output is a 2D ``PrimaryHDU`` image of shape
    ``(cutout_size * oversampling, cutout_size * oversampling)``.

    Parameters
    ----------
    model : PSFMachineModel
        Fitted polar B-spline ePSF model.
    filename : str or Path
        Output FITS file path.
    cutout_size : int or None
        Stamp size in native detector pixels.  Defaults to *model.cutout_size*
        if present, otherwise ``25``.
    oversampling : int
        Oversampling factor.  Default ``4``.
    x_0, y_0 : float
        Sub-pixel offset of the star centre (native pixels).  Default ``(0, 0)``.
    clip_negative : bool
        Set negative model values to zero.  Default ``True``.
    normalise : bool
        Normalise the image to unit sum before saving.  Default ``True``.
    extra_header : dict or None
        Additional FITS header keywords.
    overwrite : bool
        Overwrite an existing file.  Default ``True``.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    if cutout_size is None:
        cutout_size = model.cutout_size if getattr(model, "cutout_size", None) is not None else 25

    psf_img = evaluate_epsf(
        model,
        cutout_size=cutout_size,
        oversampling=oversampling,
        x_0=x_0,
        y_0=y_0,
        clip_negative=clip_negative,
        normalise=normalise,
    )

    out = Path(filename)
    hdr = fits.Header()
    hdr["EPSFMETH"] = ("psfmachine", "ePSF method module")
    hdr["CUTSIZE"] = (cutout_size, "Stamp size [native pixels]")
    hdr["OVERSAMP"] = (oversampling, "Oversampling factor")
    hdr["X_0"] = (x_0, "Sub-pixel x_0 offset [native pixels]")
    hdr["Y_0"] = (y_0, "Sub-pixel y_0 offset [native pixels]")
    hdr["RMIN"] = (float(model.rmin), "Inner radial limit [native pixels]")
    hdr["RMAX"] = (float(model.rmax), "Outer radial limit [native pixels]")
    hdr["NRKNOTS"] = (int(model.n_r_knots), "Number of radial knots")
    hdr["NPHIKNTS"] = (int(model.n_phi_knots), "Number of azimuthal knots")
    hdr["NSTARS"] = (int(model.n_stars), "Stars used to build model")

    if extra_header:
        for k, v in extra_header.items():
            hdr[k] = v

    fits.PrimaryHDU(data=psf_img.astype(np.float64), header=hdr).writeto(
        out, overwrite=overwrite
    )
    log.info(
        "save_image: wrote %s shape=%s oversampling=%d", out, psf_img.shape, oversampling
    )
    return out
