[![PyPI](https://img.shields.io/pypi/v/roman-epsf.svg)](https://pypi.org/project/roman-epsf)
[![pytest](https://github.com/jorgemarpa/roman-epsf/actions/workflows/pytest.yaml/badge.svg)](https://github.com/jorgemarpa/roman-epsf/actions/workflows/pytest.yaml)
[![mypy](https://github.com/jorgemarpa/roman-epsf/actions/workflows/mypy.yaml/badge.svg)](https://github.com/jorgemarpa/roman-epsf/actions/workflows/mypy.yaml)
[![ruff](https://github.com/jorgemarpa/roman-epsf/actions/workflows/ruff.yaml/badge.svg)](https://github.com/jorgemarpa/roman-epsf/actions/workflows/ruff.yaml)
[![Docs](https://github.com/jorgemarpa/roman-epsf/actions/workflows/deploy-mkdocs.yaml/badge.svg)](https://github.com/jorgemarpa/roman-epsf/actions/workflows/deploy-mkdocs.yaml)

# roman-epsf

`roman-epsf` provides effective Point Spread Function (ePSF) modeling algorithms tailored for the **Nancy Grace Roman Space Telescope Wide Field Instrument (WFI)**. It provides two pluggable, high-precision algorithms for constructing empirical and semi-analytical ePSFs from calibrated Roman WFI exposures, backed by a robust star-selection layer for both sparse and dense fields.

---

## About NASA Roman & the Galactic Bulge Time Domain Survey

The **Nancy Grace Roman Space Telescope** is NASA's next-generation flagship observatory, engineered to explore the infrared universe with unprecedented speed and precision. Its primary imaging camera, the **Wide Field Instrument (WFI)**, combines 18 Sensor Chip Assemblies (SCAs) totaling 300 megapixels to yield a 0.281 deg² field of view—roughly 100 times that of the Hubble Space Telescope at comparable resolution (0.11 arcsec/pixel).

A cornerstone community initiative of Roman is the **Galactic Bulge Time Domain Survey (GBTDS)**. By continuously monitoring ~2 square degrees in the dense, dust-obscured Milky Way bulge with ~15-minute sampling over six 62-day seasons in the near-infrared, the GBTDS will detect thousands of exoplanets via gravitational microlensing, discover stellar oscillations, and measure billions of stellar positions. In such extremely crowded stellar fields with detector undersampling and spatial/temporal variations, empirical ePSF modeling is crucial. `roman-epsf` generates the high-fidelity PSF models required by `roman-lcs` for precision photometry and astrometry.

---

## Modeling Methods

| Module | Algorithm | Best suited for |
|--------|-----------|-----------------|
| `roman_epsf.photutils` | `photutils.psf.EPSFBuilder` — iterative stamp-stacking & sub-pixel centroiding | Bright, isolated stars; high-fidelity core modeling; global and spatial grid ePSF maps. |
| `roman_epsf.psfmachine` | Polar B-spline modeling (NumPy/SciPy, inspired by PSFMachine) | Large star samples in crowded fields; faint or moderately blended stars; compact parameter storage. |

Both methods share the `roman_epsf.star_selection` module for isolation filtering, flux-percentile selection, and Data Quality (DQ) bitmasking.

---

## Installation

`roman-epsf` requires Python ≥ 3.11.

### User Installation with `pip`

Install the latest release from PyPI:

```bash
pip install roman-epsf
```

To include Roman Data Quality (`dq`) handling from `roman-datamodels`:

```bash
pip install "roman-epsf[dq]"
```

Or install from a local clone:

```bash
git clone https://github.com/jorgemarpa/roman-epsf.git
cd roman-epsf
pip install -e ".[dq]"
```

### Development Installation with `uv`

We recommend [uv](https://docs.astral.sh/uv/) for development. When developing across the TRExS Roman toolchain, `roman-epsf` can link to a local editable copy of `roman-cuts`:

```bash
# Clone repository
git clone https://github.com/jorgemarpa/roman-epsf.git
cd roman-epsf

# Sync environment with all development and optional extras
uv sync --all-extras

# Run tests
uv run pytest -v

# Run linting and type checking
uv run ruff check .
uv run mypy src/

# Build and preview documentation locally
uv run mkdocs serve
```

---

## Quickstart Guide

### 1. Star Selection

Select suitable PSF stars from a catalog and extract background-subtracted cutouts:

```python
from roman_epsf import select_stars, extract_star_cutouts

# Filter stars by isolation and flux distribution
stars_df = select_stars(
    catalog=catalog,              # pd.DataFrame with coordinates
    image_shape=image.shape,      # (ny, nx)
    cutout_size=25,
    flux_percentile_range=(60, 99),
    min_sep_factor=1.2,           # Isolation radius factor
    gwcs=wcs,                     # Optional GWCS if catalog contains RA/Dec
)

# Extract calibrated star cutouts with Data Quality masking
nddata, cutouts, stars_df, bkg_std = extract_star_cutouts(
    image=image,
    dq=dq,
    stars_df=stars_df,
    cutout_size=25,
    background="global",          # "global", "2d", or None
)
```

### 2. `photutils` Method — Iterative Stamp-Stacking

#### Global ePSF from an Exposure

```python
from roman_epsf import build_global_epsf, save_epsf, load_epsf

# Build global ePSF model across an exposure
epsf, fitted_stars, stars_df = build_global_epsf(
    image=image,
    catalog=catalog,
    dq=dq,
    cutout_size=25,
    oversampling=3,
    background="global",
)

# Save to standard FITS format
save_epsf(epsf, "roman_epsf.fits", extra_header={"FILTER": "F146", "SCA": 2})

# Reload from FITS
loaded_epsf, header = load_epsf("roman_epsf.fits")
```

#### Spatially Varying Grid ePSF

```python
from roman_epsf import build_grid_epsf, save_epsf

# Build a 4x4 spatial grid of ePSF models across the detector
grid = build_grid_epsf(
    image=image,
    catalog=catalog,
    dq=dq,
    n_grid_x=4,
    n_grid_y=4,
    cutout_size=25,
    oversampling=3,
)

# Save multi-cell grid model to FITS
save_epsf(grid, "roman_grid_epsf.fits")
```

#### Multi-Epoch ePSF

```python
from roman_epsf import build_multiepoch_epsf

# Build composite ePSF from multiple dithered cutouts
# cutout_list: list of dicts with keys 'data', 'dq', 'cutout_center', 'origin'
epsf, fitted_stars = build_multiepoch_epsf(cutout_list, oversampling=3)
```

### 3. `psfmachine` Method — Polar B-Spline Modeling

The polar B-spline method represents the PSF using continuous 2D radial and azimuthal B-splines, providing stable convergence even in dense star fields:

```python
from roman_epsf import select_stars
from roman_epsf.psfmachine import (
    build_pixel_arrays,
    fit_epsf,
    evaluate_epsf,
    save_model,
    load_model,
)

# Select stars (can include fainter sources)
stars_df = select_stars(catalog, image.shape, cutout_size=25, flux_percentile_range=(20, 98))

# Assemble normalized pixel coordinate arrays
dx, dy, norm_flux, star_idx = build_pixel_arrays(image, dq, stars_df, cutout_size=25)

# Fit continuous polar B-spline model
model = fit_epsf(
    dx=dx,
    dy=dy,
    norm_flux=norm_flux,
    n_r_knots=12,
    n_phi_knots=15,
    rmin=0.5,
    rmax=12.0,
)

# Evaluate on an oversampled grid for photometry
psf_array = evaluate_epsf(model, cutout_size=25, oversampling=4)

# Save compact B-spline weights to FITS
save_model(model, "roman_psm_weights.fits")
loaded_model, header = load_model("roman_psm_weights.fits")
```

---

## FITS Serialization Formats

`roman-epsf` outputs self-documenting FITS files compatible with standard astronomy tooling:
- **Single ePSF**: `PrimaryHDU` containing the 2D oversampled PSF image, with keywords `OVERSMPX`, `OVERSMPY`, `ORIGIN_X`, `ORIGIN_Y`, and `EPSFGRID=False`.
- **Grid ePSF**: `PrimaryHDU` metadata + one `ImageHDU` per spatial grid cell (`CELL_IY_IX`) + `BinTableHDU` (`GRID_GEOM`) storing cell centers and star counts (`EPSFGRID=True`).
- **PSFMachine Spline**: `PrimaryHDU` metadata + `ImageHDU "WEIGHTS"` (B-spline coefficients) + `ImageHDU "R_KNOTS"` (radial knot locations).

---

## Changelog

See [CHANGELOG.md](https://github.com/jorgemarpa/roman-epsf/blob/main/CHANGELOG.md) for a complete release history.
