"""
roman_epsf
==========
ePSF model construction for Roman WFI detectors.

The package is organised around pluggable *method modules*:

``roman_epsf.photutils``
    EPSFBuilder method.  Builders: :func:`build_global_epsf`,
    :func:`build_grid_epsf`, :func:`build_multiepoch_epsf`.
    I/O: :func:`save_epsf`, :func:`load_epsf`.

``roman_epsf.psfmachine``
    PSFMachine-inspired polar B-spline method (pure NumPy/SciPy).
    Builders: :func:`build_pixel_arrays`, :func:`fit_epsf`,
    :func:`evaluate_epsf`.
    I/O: :func:`save_model`, :func:`load_model`, :func:`save_image`.

A shared star-selection layer at :mod:`roman_epsf.star_selection` is
method-agnostic and can be reused by both (or any future) method modules.

Quick start
-----------
**EPSFBuilder — global model from one exposure**::

    from roman_epsf import build_global_epsf, save_epsf
    epsf, fitted_stars, stars_df = build_global_epsf(image, catalog, dq=dq, gwcs=wcs)
    save_epsf(epsf, "my_epsf.fits", extra_header={"FILTER": "F146"})

**PSFMachine — polar B-spline model**::

    from roman_epsf import select_stars
    from roman_epsf.psfmachine import build_pixel_arrays, fit_epsf, evaluate_epsf, save_model
    stars_df = select_stars(catalog, image.shape, cutout_size=25,
                            flux_percentile_range=(20, 98))
    dc, dr, nf, _ = build_pixel_arrays(image, dq, stars_df, cutout_size=25)
    model = fit_epsf(dc, dr, nf, n_r_knots=12, n_phi_knots=15)
    psf_img = evaluate_epsf(model, cutout_size=25, oversampling=4)
    save_model(model, "my_psfmachine.fits")

**Star selection only** (method-agnostic)::

    from roman_epsf import select_stars, extract_star_cutouts
    stars_df = select_stars(catalog, image.shape, cutout_size=25)
    nddata, cutouts, stars_df, std = extract_star_cutouts(image, dq, stars_df, 25)
"""

import logging
from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev"

log = logging.getLogger(__name__)

from .photutils import (  # noqa: E402
    GridEPSF,
    build_global_epsf,
    build_grid_epsf,
    build_multiepoch_epsf,
    load_epsf,
    save_epsf,
)
from .psfmachine import (  # noqa: E402
    PSFMachineModel,
    build_pixel_arrays,
    evaluate_epsf,
    fit_epsf,
    load_model,
    save_image,
    save_model,
)
from .star_selection import extract_star_cutouts, select_stars  # noqa: E402

__all__ = [
    # shared star-selection utilities
    "select_stars",
    "extract_star_cutouts",
    # photutils builders
    "GridEPSF",
    "build_global_epsf",
    "build_grid_epsf",
    "build_multiepoch_epsf",
    "save_epsf",
    "load_epsf",
    # psfmachine builders
    "PSFMachineModel",
    "build_pixel_arrays",
    "fit_epsf",
    "evaluate_epsf",
    "save_model",
    "load_model",
    "save_image",
]
