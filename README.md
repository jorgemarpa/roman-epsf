# roman-epsf

ePSF model construction for Roman WFI detectors.

`roman-epsf` provides two pluggable methods for building effective PSF (ePSF)
models from calibrated Roman WFI L2 exposures, along with a shared star-selection
layer that both methods share.

---

## Methods

| Module | Algorithm | Best for |
|--------|-----------|----------|
| `roman_epsf.photutils` | `photutils.psf.EPSFBuilder` — iterative stamp-stacking | Bright, isolated stars; high-fidelity centroiding |
| `roman_epsf.psfmachine` | Polar B-spline (pure NumPy/SciPy, PSFMachine-inspired) | Large star samples including faint/blended sources |

---

## Installation

```bash
# from the repos directory
pip install -e roman-epsf/
```

With optional DQ flag support from the Roman pipeline:

```bash
pip install -e "roman-epsf/[dq]"
```

**Requirements:** Python ≥ 3.11, NumPy, SciPy ≥ 1.10, Astropy, photutils ≥ 2.0,
pandas, and [`roman-cuts`](https://github.com/jorgemarpa/roman-cuts) for data access.

---

## Quick start

### `photutils` method — global ePSF from one exposure

```python
from roman_epsf import build_global_epsf, save_epsf, load_epsf

# image: 2D np.ndarray, catalog: pd.DataFrame with x_centroid/y_centroid
epsf, fitted_stars, stars_df = build_global_epsf(
    image, catalog,
    dq=dq,
    gwcs=wcs,
    cutout_size=25,
    oversampling=3,
    background="global",        # or "2d" for Background2D surface
)

# Save to FITS and reload
save_epsf(epsf, "roman_epsf.fits", extra_header={"FILTER": "F146"})
loaded_epsf, hdr = load_epsf("roman_epsf.fits")
```

### `photutils` method — spatially varying grid

```python
from roman_epsf import build_grid_epsf, save_epsf

grid = build_grid_epsf(image, catalog, dq=dq, n_grid_x=4, n_grid_y=4)
# grid.data  shape: (n_grid_y, n_grid_x, psf_h, psf_w)
# grid.models: object array of EPSFModel per cell
save_epsf(grid, "roman_grid_epsf.fits")
```

### `psfmachine` method — polar B-spline ePSF

```python
from roman_epsf import select_stars
from roman_epsf.psfmachine import build_pixel_arrays, fit_epsf, evaluate_epsf, save_model, load_model

# Star selection can be looser: fainter and less isolated sources are fine
stars_df = select_stars(
    catalog, image.shape, cutout_size=25,
    flux_percentile_range=(20, 98),
    min_sep_factor=0.8,
)

# Assemble pixel arrays across all star cutouts
dc, dr, norm_flux, star_idx = build_pixel_arrays(image, dq, stars_df, cutout_size=25)

# Fit polar B-spline model
model = fit_epsf(dc, dr, norm_flux, n_r_knots=12, n_phi_knots=15, rmin=0.5, rmax=12.0)

# Evaluate on an oversampled grid (compatible with photutils.psf.ImagePSF)
psf_img = evaluate_epsf(model, cutout_size=25, oversampling=4)

# Save compact B-spline weights to FITS
save_model(model, "roman_psm.fits")
loaded_model, hdr = load_model("roman_psm.fits")
```

### Multi-epoch ePSF

```python
from roman_epsf import build_multiepoch_epsf

# cutout_list: list of dicts with keys data, dq, cutout_center, origin
epsf, fitted_stars = build_multiepoch_epsf(cutout_list, oversampling=3)
```

---

## Star selection

Both methods share the same star-selection utilities in `roman_epsf.star_selection`:

```python
from roman_epsf import select_stars, extract_star_cutouts

stars_df = select_stars(
    catalog, image.shape, cutout_size=25,
    flux_percentile_range=(60, 99),   # adjust per method
    min_sep_factor=1.2,               # isolation radius = factor × cutout_size
    gwcs=wcs,                         # optional; used if catalog has only RA/Dec
)

nddata, cutouts, stars_df, bkg_std = extract_star_cutouts(
    image, dq, stars_df, cutout_size=25,
    background="global",              # "global", "2d", or None
    bkg2d_kwargs={"box_size": 128},   # forwarded to photutils.background.Background2D
)
```

---

## Package layout

```
roman-epsf/
├── src/roman_epsf/
│   ├── __init__.py            # top-level re-exports
│   ├── star_selection.py      # shared: select_stars, extract_star_cutouts
│   ├── photutils/
│   │   ├── builder.py         # build_global_epsf, build_grid_epsf, build_multiepoch_epsf
│   │   └── io.py              # save_epsf, load_epsf
│   └── psfmachine/
│       ├── builder.py         # build_pixel_arrays, fit_epsf, evaluate_epsf, PSFMachineModel
│       ├── _spline.py         # B-spline primitives (polar, clamped, periodic)
│       └── io.py              # save_model, load_model, save_image
├── notebooks/
│   ├── demo_photutils_epsf.ipynb    # photutils workflow demo
│   └── demo_psfmachine_epsf.ipynb  # PSFMachine workflow demo
└── tests/
    ├── test_star_selection.py
    ├── test_builder.py        # photutils builders + FITS I/O
    └── test_psfmachine.py     # PSFMachine builders, splines, FITS I/O
```

---

## FITS file formats

### `photutils` single ePSF

`PrimaryHDU` containing the 2D oversampled PSF array, with header keywords
`OVERSMPX`, `OVERSMPY`, `ORIGIN_X/Y`, and model parameters. `EPSFGRID=False`.

### `photutils` grid ePSF

`PrimaryHDU` (metadata) + one `ImageHDU` per grid cell (`CELL_IY_IX`) +
`BinTableHDU "GRID_GEOM"` storing cell centres, edges, and star counts.
`EPSFGRID=True`.

### `psfmachine` model

`PrimaryHDU` (scalar parameters: `RMIN`, `RMAX`, `CUT_R`, `NRKNOTS`, `NPHIKNTS`,
`DEGREE`, `NSTARS`, `NPIXELS`) + `ImageHDU "WEIGHTS"` (B-spline coefficients) +
`ImageHDU "R_KNOTS"` (radial knot positions).

---

## Development

```bash
pip install -e "roman-epsf/[dq]"   # editable + optional dep
pytest roman-epsf/tests/
```

Linting and type-checking:

```bash
ruff check src/ tests/
mypy src/
```
