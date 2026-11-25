# Changelog

All notable changes to `roman-epsf` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-24

### Added
- Initial public release of `roman-epsf` for Roman Space Telescope WFI effective PSF (ePSF) modeling.
- Pluggable `photutils` iterative stamp-stacking algorithm:
  - Global single-exposure ePSF construction (`build_global_epsf`).
  - Spatially varying grid ePSF construction across detector coordinates (`build_grid_epsf`).
  - Multi-epoch ePSF construction across dithered exposures (`build_multiepoch_epsf`).
- Polar B-spline ePSF algorithm inspired by `PSFMachine` (`roman_epsf.psfmachine`):
  - Star pixel array aggregation (`build_pixel_arrays`).
  - Smooth polar B-spline fitting with radial/azimuthal knot optimization (`fit_epsf`).
  - Oversampled grid evaluation compatible with `photutils.psf.ImagePSF` (`evaluate_epsf`).
- Shared star selection and cutout extraction layer (`roman_epsf.star_selection`):
  - Catalog-driven isolation and flux-percentile filtering (`select_stars`).
  - Data Quality (DQ) bitmask filtering and 2D background estimation (`extract_star_cutouts`).
- Support for initial ePSF seeding from `stpsf` WebbPSF-Roman models.
- Standardized FITS serialization for single ePSFs, multi-cell grid models, and B-spline coefficient representations.
- Modern PEP 621 packaging with `uv` and `hatchling` build backend.
- MkDocs documentation suite with Material theme and API docstrings.
- Comprehensive GitHub Actions CI suite (`pytest`, `ruff`, `mypy`, `deploy-mkdocs`, `pypi-publish`).
