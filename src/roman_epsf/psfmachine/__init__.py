"""
roman_epsf.psfmachine
=====================
PSFMachine-style polar B-spline ePSF method module (pure NumPy / SciPy).

Public API
----------
* :func:`build_pixel_arrays` — assemble ``delta_col``, ``delta_row``, and
  normalised-flux arrays from cutouts and a catalog.
* :func:`fit_epsf` — fit a polar B-spline model to the pixel arrays.
* :func:`evaluate_epsf` — evaluate the model on an oversampled grid.
* :func:`save_model` / :func:`load_model` — FITS round-trip for the spline
  coefficients and knot parameters.
* :func:`save_image` — evaluate and write an oversampled ePSF image to FITS.
* :class:`PSFMachineModel` — dataclass holding the fitted model.
"""

from .builder import PSFMachineModel, build_pixel_arrays, evaluate_epsf, fit_epsf, refine_centroids
from .io import load_model, save_image, save_model

__all__ = [
    "PSFMachineModel",
    "build_pixel_arrays",
    "fit_epsf",
    "evaluate_epsf",
    "refine_centroids",
    "save_model",
    "load_model",
    "save_image",
]
