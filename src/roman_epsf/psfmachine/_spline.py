"""
roman_epsf.psfmachine._spline
==============================
Pure NumPy / SciPy B-spline utilities for the PSFMachine-style polar ePSF model.

This module replaces the PSFMachine ``patsy``/``scipy.sparse`` design-matrix
machinery with dense NumPy arrays so the ePSF module has no non-standard
astronomy dependencies.

Key references
--------------
* Hedges et al. 2021: https://ui.adsabs.harvard.edu/abs/2021arXiv210608411H
* Original PSFMachine utils: github.com/SSDataLab/psfmachine/blob/master/src/psfmachine/utils.py
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

__all__ = [
    "_wrapped_spline_matrix",
    "_r_bspline_matrix",
    "_make_design_matrix_polar",
    "_xy_bspline_matrix",
    "_cartesian_knots_1d",
    "_make_design_matrix_cartesian",
    "_solve_linear",
]


def _wrapped_spline_matrix(phi: np.ndarray, n_knots: int = 15) -> np.ndarray:
    """Periodic (wrapped) cubic B-spline basis for the angle coordinate.

    The angle vector *phi* must lie in ``[-π, π]``.  The basis handles the
    periodicity by extending the domain to ``[π, 3π]`` and folding the extra
    columns back onto the first half, producing continuous wrap-around.

    This is a dense-array reimplementation of PSFMachine's ``wrapped_spline``.

    Parameters
    ----------
    phi : np.ndarray
        1D array of angle values in ``[-π, π]``.
    n_knots : int
        Number of angular knots (= number of basis columns returned).

    Returns
    -------
    np.ndarray
        Dense matrix of shape ``(len(phi), n_knots)``.
    """
    if not np.all((phi >= -np.pi) & (phi <= np.pi)):
        raise ValueError("All phi values must be in [-pi, pi].")

    x = phi.copy()
    # Extend domain: stack original and shifted-by-2π copy
    x_ext = np.concatenate([x, x + 2.0 * np.pi])

    nt = 2 * n_knots + 1
    t = np.linspace(-np.pi, 3.0 * np.pi, nt)
    dt = float(np.median(np.diff(t)))

    # --- Cox-de Boor recursion (always cubic = degree 3) --------------------
    # Zeroth-order basis (box functions)
    basis = np.array(
        [((x_ext >= t[i]) & (x_ext < t[i + 1])).astype(float) for i in range(nt - 1)]
    )  # shape: (nt-1, 2*N)

    for o in range(1, 4):  # build up to cubic regardless of `order` argument
        new_basis = []
        for i in range(nt - 1):
            a = ((x_ext - t[i]) / (dt * o)) * basis[i]
            if (i + o + 1) < (nt - 1):
                b = (
                    -(x_ext - t[i + o + 1]) / (dt * o)
                ) * basis[(i + 1) % (nt - 1)]
            else:
                b = np.zeros(len(x_ext))
            new_basis.append(a + b)
        basis = np.vstack(new_basis)  # shape: (nt-1, 2*N)

    # Fold: take first n_knots rows for the original x positions …
    half = nt // 2  # == n_knots
    folded = basis[:half, : len(x)].copy()
    # … and add the wrap-around contribution from the extended part
    for i in range(-3, 0):  # last 3 rows (cubic = 3 wrap-around terms)
        folded[i] += basis[half + i, len(x) :]

    return folded.T  # shape: (N, n_knots)


def _r_bspline_matrix(
    r: np.ndarray,
    rmin: float,
    rmax: float,
    r_knots: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """Cubic B-spline design matrix for the radial coordinate.

    Equivalent to patsy's
    ``dmatrix("bs(x, knots=r_knots, degree=3, include_intercept=True)", ...)``
    augmented with boundary-clamp points at *rmin* and *rmax*.

    Parameters
    ----------
    r : np.ndarray
        1D array of radial distances.
    rmin, rmax : float
        Domain boundaries (same units as *r*).
    r_knots : np.ndarray
        Interior knot positions (all used as interior knots; boundary knots
        are added automatically at *rmin* and *rmax*).
    degree : int
        Spline degree.  Default ``3`` (cubic).

    Returns
    -------
    np.ndarray
        Dense matrix of shape ``(len(r), n_r_basis)`` where
        ``n_r_basis = len(r_knots) + degree + 1``.
    """
    # Full knot vector: clamped at boundaries (degree+1 repetitions each side)
    t = np.concatenate([
        np.repeat(rmin, degree + 1),
        r_knots,               # all r_knots treated as interior knots
        np.repeat(rmax, degree + 1),
    ])
    # Augment r with boundary values so the full knot range is always covered;
    # these two extra rows are removed at the end (same trick as PSFMachine).
    r_aug = np.concatenate([r, [rmin, rmax]])
    dm = BSpline.design_matrix(r_aug, t, degree).toarray()
    return dm[:-2]  # drop the two boundary sentinel rows


def _make_design_matrix_polar(
    r: np.ndarray,
    phi: np.ndarray,
    rmin: float,
    rmax: float,
    r_knots: np.ndarray,
    n_phi_knots: int,
    cut_r: float,
    degree: int = 3,
) -> np.ndarray:
    """Build the polar B-spline design matrix.

    This is a dense, pure NumPy/SciPy reimplementation of PSFMachine's
    ``_make_A_polar``.  It enforces radial-only dependence inside *cut_r* so
    that the small-radius region (where fewer pixels are sampled) is not
    over-parameterised with angular terms.

    Parameters
    ----------
    r : np.ndarray
        1D radial distances (same units as *rmin*/*rmax*/*cut_r*).
    phi : np.ndarray
        1D angle values in ``[-π, π]``.
    rmin, rmax : float
        Radial domain boundaries.
    r_knots : np.ndarray
        Radial knot positions (``n_r_knots`` values, sqrt-spaced recommended).
    n_phi_knots : int
        Number of angular basis functions.
    cut_r : float
        Radius below which only radial dependence is modelled.
    degree : int
        Spline degree.  Default ``3``.

    Returns
    -------
    np.ndarray
        Dense design matrix of shape ``(len(r), n_basis)`` where
        ``n_basis = n_phi_knots * (n_r_basis - cut_r_int) + (cut_r_int - 1) + 1``.
    """
    # Radial basis: (N, n_r_basis)
    r_dm = _r_bspline_matrix(r, rmin, rmax, r_knots, degree)
    n_r = r_dm.shape[1]

    # Angular (periodic) basis: (N, n_phi_knots)
    phi_dm = _wrapped_spline_matrix(phi, n_phi_knots)
    n_phi = phi_dm.shape[1]

    # Full tensor product: column i = phi_j * r_k
    # Layout: [phi_0*r_0, phi_1*r_0, ..., phi_{n_phi-1}*r_0, phi_0*r_1, ...]
    X = np.hstack([phi_dm * r_dm[:, k : k + 1] for k in range(n_r)])
    # shape: (N, n_phi * n_r)

    # Determine cut-off index in r_knots.
    # Clamp to >= 2 so the inner pure-radial block r_dm[:, 1:cut_r_int] is
    # never empty (which would happen when cut_r falls between the first two
    # knots, leaving the steepest PSF core with only a constant intercept).
    matches = np.where(r_knots <= cut_r)[0]
    cut_r_int = max(2, int(matches.max())) if len(matches) >= 2 else 2

    # Columns to discard: all phi-modulated terms for r-basis indices < cut_r_int
    cut_cols = set(range(n_phi * cut_r_int))
    keep_cols = np.array(sorted(set(range(X.shape[1])) - cut_cols))

    # Reassemble:
    #   - outer phi×r terms (r > cut_r)
    #   - inner pure-radial terms (r < cut_r, skipping index 0 = intercept)
    #   - global intercept column
    X1 = np.hstack([
        X[:, keep_cols],                    # outer: phi × r terms
        r_dm[:, 1:cut_r_int],              # inner: pure radial (no phi)
        np.ones((len(r), 1), dtype=float), # global intercept
    ])
    return X1


def _xy_bspline_matrix(
    x: np.ndarray,
    xmin: float,
    xmax: float,
    knots: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """Clamped B-spline design matrix for one Cartesian coordinate.

    Equivalent to :func:`_r_bspline_matrix` but for an arbitrary domain
    ``[xmin, xmax]`` that may include negative values.

    Parameters
    ----------
    x : np.ndarray
        1D array of coordinate values.  All entries should lie in
        ``[xmin, xmax]``; values outside are clamped by the boundary knots.
    xmin, xmax : float
        Domain boundaries (``xmin < xmax``).
    knots : np.ndarray
        Interior knot positions (all between ``xmin`` and ``xmax``).
    degree : int
        Spline degree.  Default ``3`` (cubic).

    Returns
    -------
    np.ndarray
        Dense matrix of shape ``(len(x), len(knots) + degree + 1)``.
    """
    t = np.concatenate([
        np.repeat(xmin, degree + 1),
        knots,
        np.repeat(xmax, degree + 1),
    ])
    x_aug = np.concatenate([x, [xmin, xmax]])
    dm = BSpline.design_matrix(x_aug, t, degree).toarray()
    return dm[:-2]


def _cartesian_knots_1d(
    dmax: float,
    n_knots: int,
    cut_r: float,
) -> np.ndarray:
    """Interior knot positions for a 1D B-spline on ``[-dmax, dmax]``.

    Knots within ``(-cut_r, cut_r)`` are sqrt-spaced (denser near 0) to give
    higher resolution in the PSF core.  Knots outside ``cut_r`` are linearly
    spaced.

    Parameters
    ----------
    dmax : float
        Domain half-width.  Knots span ``(-dmax, dmax)`` (boundaries excluded).
    n_knots : int
        Target number of interior knots.  The actual count may differ by ±1
        depending on rounding.
    cut_r : float
        Transition radius: sqrt-spaced inside, linearly spaced outside.

    Returns
    -------
    np.ndarray
        Sorted 1D array of interior knot positions.
    """
    cut_r = min(float(cut_r), dmax * 0.85)  # guard: cut_r must be < dmax
    inner_frac = cut_r / dmax

    # Allocate knots proportionally: more in the inner (PSF core) region
    n_inner_half = max(2, int(round(n_knots * inner_frac / 2.0)))
    n_outer_half = max(1, n_knots // 2 - n_inner_half)

    # Positive inner half: sqrt-spaced on (0, cut_r] — denser near 0
    inner_pos = np.linspace(0.0, cut_r ** 0.5, n_inner_half + 1)[1:] ** 2

    # Positive outer half: linearly spaced on (cut_r, dmax)
    outer_pos = np.linspace(cut_r, dmax, n_outer_half + 2)[1:-1]

    knots_pos = np.concatenate([inner_pos, outer_pos])
    # Mirror to negative side
    return np.sort(np.concatenate([-knots_pos[::-1], knots_pos]))


def _make_design_matrix_cartesian(
    dc: np.ndarray,
    dr: np.ndarray,
    dmax: float,
    col_knots: np.ndarray,
    row_knots: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """2D Cartesian tensor-product B-spline design matrix.

    Builds the full outer product of two 1D B-spline bases — one along the
    column offset axis and one along the row offset axis — on the symmetric
    domain ``[-dmax, dmax]``.

    Sqrt-spaced knots (from :func:`_cartesian_knots_1d`) concentrate basis
    functions near the PSF centre; the linear solver then assigns near-zero
    weights to functions that cover the quiet outer region.

    Parameters
    ----------
    dc : np.ndarray
        Column offsets from star centre (native detector pixels).
    dr : np.ndarray
        Row offsets from star centre (native detector pixels).
    dmax : float
        Domain half-width in native pixels (equal to ``rmax`` in the model).
    col_knots : np.ndarray
        Interior knot positions for the column axis.
    row_knots : np.ndarray
        Interior knot positions for the row axis.
    degree : int
        Spline degree.  Default ``3`` (cubic).

    Returns
    -------
    np.ndarray
        Dense design matrix of shape
        ``(len(dc), n_col_basis * n_row_basis)`` where
        ``n_col_basis = len(col_knots) + degree + 1``.
    """
    dmin = -dmax
    col_dm = _xy_bspline_matrix(dc, dmin, dmax, col_knots, degree)  # (N, n_col)
    row_dm = _xy_bspline_matrix(dr, dmin, dmax, row_knots, degree)  # (N, n_row)

    # Tensor product: col_j * row_k for all (j, k)
    # Layout: all col functions paired with row_0, then row_1, ...
    return np.hstack([col_dm * row_dm[:, k : k + 1] for k in range(row_dm.shape[1])])


def _solve_linear(
    A: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None = None,
    prior_sigma: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the weighted linear system ``A w = y`` for weights ``w``.

    Optionally applies per-observation weights (``1/y_err²``) and a zero-mean
    Gaussian prior with standard deviation *prior_sigma* on each weight.

    Parameters
    ----------
    A : np.ndarray
        Design matrix of shape ``(n_obs, n_basis)``.
    y : np.ndarray
        Observation vector of length ``n_obs``.
    y_err : np.ndarray or None
        Per-observation standard deviations.  When provided, the system is
        weighted by ``1/y_err²``.
    prior_sigma : float or None
        Standard deviation of the isotropic Gaussian prior on *w*.  Adds
        ``diag(1/prior_sigma²)`` to the normal equations.
    mask : np.ndarray of bool or None
        Boolean mask selecting the observations to use.  Default: all True.

    Returns
    -------
    np.ndarray
        Weight vector of length ``n_basis``.
    """
    if mask is None:
        mask = np.ones(len(y), dtype=bool)

    A_m = A[mask]
    y_m = y[mask]

    if y_err is not None:
        w2 = 1.0 / y_err[mask] ** 2
        sigma_w_inv = (A_m * w2[:, None]).T @ A_m
        B = A_m.T @ (y_m * w2)
    else:
        sigma_w_inv = A_m.T @ A_m
        B = A_m.T @ y_m

    if prior_sigma is not None:
        sigma_w_inv += np.diag(np.ones(A.shape[1]) / prior_sigma ** 2)

    return np.linalg.solve(sigma_w_inv, B)
