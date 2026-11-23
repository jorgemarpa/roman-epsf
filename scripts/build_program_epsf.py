#!/usr/bin/env python3
"""
build_program_epsf.py
=====================
CLI script to build global photutils ePSF models for Roman WFI detector exposures
from a specific MAST Program ID (PID) observation and filter.

Based on the workflow in `roman-epsf/notebooks/demo_photutils_epsf.ipynb`.

Key Features:
-------------
- Supports running for a single detector, a list of detectors, or all 18 detectors.
- Designed for distributed parallelization with GNU `parallel` or `xargs`.
- Automatically selects the second "Observation" number from the MAST product filename
  encodings by default (or uses a user-defined `--observation`).
- Saves ePSF FITS models into structured directories: `<output_dir>/pid_<PID>/<FILTER>/`.
- Embeds complete provenance and observational metadata in the primary FITS header.
- Optional warm-starting using physical PSF models from `stpsf`.

Usage Examples:
---------------
1. Single detector (ideal for distributed parallel workers):
   $ python build_program_epsf.py --pid 1055 --filter F146 --detector 2

2. Distributed across multiple cores with GNU parallel:
   $ parallel -j 4 python build_program_epsf.py --pid 1055 --filter F146 --detector {} ::: {1..18}

3. Distributed with xargs:
   $ seq 1 18 | xargs -n 1 -P 4 -I {} python build_program_epsf.py --pid 1055 --filter F146 --detector {}

4. Process all detectors sequentially:
   $ python build_program_epsf.py --pid 1055 --filter F146 --detector all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# MAST data access
from astroquery.mast import MastMissions, Observations
from roman_cuts.com import query_mast_files
from roman_cuts.com.io import _open_asdf, open_parquet

# roman-epsf
from roman_epsf import build_global_epsf, load_epsf, save_epsf, select_stars

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("build_program_epsf")


# Fixed ePSF construction parameters
CUTOUT_SIZE: int = 25
BACKGROUND: str = "2d"
FLUX_PERCENTILE_RANGE: Tuple[float, float] = (50.0, 99.0)
MIN_SEP_FACTOR: float = 1.0
INITIAL_EPSF: bool = True


def parse_detectors(detector_arg: Optional[str]) -> List[int]:
    """Parse detector argument into a list of detector integers (1-18).

    Accepts:
      - None or 'all': [1, 2, ..., 18]
      - Single integer/string: '2', 'wfi02', 'WFI02' -> [2]
      - Comma-separated: '1,2,5' -> [1, 2, 5]
      - Range: '1-4' -> [1, 2, 3, 4]
    """
    if detector_arg is None or detector_arg.strip().lower() in ("all", "*"):
        return list(range(1, 19))

    detectors = []
    for part in detector_arg.split(","):
        part = part.strip().upper()
        if not part:
            continue
        if part.startswith("WFI"):
            part = part[3:]
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            detectors.extend(range(start, end + 1))
        else:
            detectors.append(int(part))

    # Validate range
    out = sorted(list(set(detectors)))
    for d in out:
        if d < 1 or d > 18:
            raise ValueError(f"Invalid detector index '{d}'. Must be between 1 and 18.")
    return out


def extract_observation_from_filename(filename: str) -> Optional[int]:
    """Extract the 3-digit observation number (OOO) from a Roman filename.

    Encoding format:
    r{PPPPP}{CC}{AAA}{SSS}{OOO}{VVV}_{eeee}_{wfiss}_{filter}_{suffix}
    Characters 14..17 correspond to {OOO}.
    """
    base = os.path.basename(filename)
    if base.startswith("r") and len(base) >= 17:
        try:
            return int(base[14:17])
        except ValueError:
            return None
    return None


def init_mast_session() -> MastMissions:
    """Initialize authenticated MAST mission session."""
    default_env = "/teams/romcom/data/jmartinez/my_crds_path.env"
    if os.path.exists(default_env):
        load_dotenv(default_env, override=True)
        log.info("Loaded default environment credentials from: %s", default_env)
    else:
        load_dotenv(override=True)

    token = os.environ.get("MAST_API_TOKEN", "")
    mission = MastMissions(mission="roman")
    if token:
        mission.login(token=token)
        log.info("Authenticated successfully with MAST using token.")
    else:
        log.warning("No MAST_API_TOKEN found in environment; querying public/anonymous datasets.")

    try:
        Observations.enable_cloud_dataset()
        log.info("Enabled MAST S3 cloud dataset access.")
    except Exception as exc:
        log.warning("Could not enable cloud dataset: %s", exc)

    return mission


def determine_observation_number(
    mission: MastMissions,
    pid: int,
    filter_name: str,
    probe_detector: int = 1,
) -> int:
    """Query MAST to determine the default observation number (second in sorted list)."""
    log.info(
        "No observation number specified. Querying MAST for PID=%d, filter=%s to discover observations...",
        pid,
        filter_name,
    )
    products, _ = query_mast_files(
        program_id=pid,
        filter_name=filter_name,
        mission=mission,
        file_type="l2",
        detector=probe_detector,
        observation=None,
    )

    if len(products) == 0:
        # Fallback probe across detector 2
        log.info("No products found on detector %d; probing detector 2...", probe_detector)
        products, _ = query_mast_files(
            program_id=pid,
            filter_name=filter_name,
            mission=mission,
            file_type="l2",
            detector=2,
            observation=None,
        )

    if len(products) == 0:
        raise RuntimeError(
            f"No L2 data products found on MAST for Program {pid} and filter {filter_name}. "
            "Verify program ID and filter name."
        )

    obs_numbers = set()
    for fn in products["filename"]:
        obs = extract_observation_from_filename(str(fn))
        if obs is not None:
            obs_numbers.add(obs)

    sorted_obs = sorted(list(obs_numbers))
    if not sorted_obs:
        raise RuntimeError("Could not parse any observation numbers from product filenames.")

    # Rule: use as default the second "Observation" value from the list
    if len(sorted_obs) >= 2:
        chosen_obs = sorted_obs[1]
        log.info(
            "Found %d unique observations: %s. Defaulting to second observation: %d",
            len(sorted_obs),
            sorted_obs,
            chosen_obs,
        )
    else:
        chosen_obs = sorted_obs[0]
        log.info(
            "Only 1 unique observation found (%s). Using observation: %d",
            sorted_obs,
            chosen_obs,
        )

    return chosen_obs


def process_detector(
    detector: int,
    pid: int,
    filter_name: str,
    observation: int,
    mission: MastMissions,
    output_dir: Path,
    exposure_idx: int = 1,
    oversampling: int = 3,
    maxiters: int = 5,
    skip_existing: bool = False,
    overwrite: bool = True,
) -> Optional[Path]:
    """Build and save global ePSF model for a single detector."""
    det_str = f"WFI{detector:02d}"
    out_subdir = output_dir / f"pid_{pid}" / filter_name.upper()
    out_subdir.mkdir(parents=True, exist_ok=True)

    fits_filename = f"roman_epsf_pid{pid}_wfi{detector:02d}_{filter_name.lower()}_obs{observation:03d}_global.fits"
    fits_path = out_subdir / fits_filename

    if skip_existing and fits_path.exists():
        log.info("[%s] Output file already exists (%s); skipping.", det_str, fits_path)
        return fits_path

    log.info(
        "[%s] Querying L2 images and L4 catalogs (PID=%d, Filter=%s, Obs=%d)...",
        det_str,
        pid,
        filter_name,
        observation,
    )

    # 1. Query L2 and L4 files
    products_l2, _ = query_mast_files(
        program_id=pid,
        filter_name=filter_name,
        mission=mission,
        file_type="l2",
        detector=detector,
        observation=observation,
    )
    products_l4, _ = query_mast_files(
        program_id=pid,
        filter_name=filter_name,
        mission=mission,
        file_type="l4",
        detector=detector,
        observation=observation,
    )

    if len(products_l2) == 0:
        log.error("[%s] No L2 products found for observation %d. Skipping.", det_str, observation)
        return None
    if len(products_l4) == 0:
        log.error("[%s] No L4 products found for observation %d. Skipping.", det_str, observation)
        return None

    products_l2.sort("product_key")
    products_l4.sort("product_key")

    # Select exposure index
    actual_exp_idx = exposure_idx
    if actual_exp_idx >= len(products_l2):
        actual_exp_idx = 0
        log.warning(
            "[%s] Requested exposure index %d out of bounds (%d available); falling back to 0.",
            det_str,
            exposure_idx,
            len(products_l2),
        )

    asdf_filename = str(products_l2["filename"][actual_exp_idx])
    prefix = asdf_filename.replace("_cal.asdf", "")
    matching_l4 = [f for f in products_l4["filename"] if prefix in str(f)]
    parquet_filename = str(matching_l4[0]) if matching_l4 else str(products_l4["filename"][actual_exp_idx])

    log.info("[%s] Selected L2 Image  : %s", det_str, asdf_filename)
    log.info("[%s] Matched L4 Catalog : %s", det_str, parquet_filename)

    # 2. Load Image Data & WCS
    token = os.environ.get("MAST_API_TOKEN", "")
    with _open_asdf(asdf_filename, mission=mission) as af:
        image_data = np.asarray(af["roman"]["data"], dtype=np.float64).copy()
        dq = np.asarray(af["roman"]["dq"]).copy() if "dq" in af["roman"] else None
        gwcs = af["roman"]["meta"].get("wcs", None)

    # 3. Load Catalog
    if os.path.exists(parquet_filename):
        cat_df = pd.read_parquet(parquet_filename)
    else:
        cat_df = open_parquet(parquet_filename, token=token)

    log.info("[%s] Image shape: %s | Catalog sources: %d", det_str, image_data.shape, len(cat_df))

    # 4. Star Selection
    selected = select_stars(
        cat_df,
        image_shape=image_data.shape,
        cutout_size=CUTOUT_SIZE,
        flux_percentile_range=FLUX_PERCENTILE_RANGE,
        min_sep_factor=MIN_SEP_FACTOR,
        gwcs=gwcs,
    )
    log.info("[%s] Selected %d isolated point sources for ePSF construction.", det_str, len(selected))

    if len(selected) < 3:
        log.error("[%s] Insufficient stars selected (%d < 3) to build ePSF. Skipping.", det_str, len(selected))
        return None

    # 5. Build global ePSF model (always warm-starts with stpsf and 2d background)
    log.info(
        "[%s] Fitting global ePSF (cutout_size=%d, oversampling=%d, maxiters=%d, initial_epsf=stpsf, background=%s)...",
        det_str,
        CUTOUT_SIZE,
        oversampling,
        maxiters,
        BACKGROUND,
    )
    epsf_model, fitted_stars, _ = build_global_epsf(
        image=image_data,
        catalog=selected,
        dq=dq,
        gwcs=gwcs,
        cutout_size=CUTOUT_SIZE,
        oversampling=oversampling,
        maxiters=maxiters,
        flux_percentile_range=(0.0, 100.0),  # already filtered
        min_sep_factor=1.0,
        background=BACKGROUND,
        initial_epsf=INITIAL_EPSF,
        detector=det_str,
        filter_name=filter_name,
        progress_bar=False,
    )

    # 6. Save model with header metadata
    extra_header = {
        "INSTRUME": "WFI",
        "DETECTOR": det_str,
        "FILTER": filter_name.upper(),
        "PROGRAM": int(pid),
        "OBSNUM": int(observation),
        "CUTSIZE": int(CUTOUT_SIZE),
        "OVERSAMP": int(oversampling),
        "NSTARS": int(len(fitted_stars)),
        "ASDF_REF": os.path.basename(asdf_filename),
        "PARQ_REF": os.path.basename(parquet_filename),
        "INIT_MOD": "stpsf",
        "BKG_MODE": BACKGROUND,
    }

    saved_path = save_epsf(
        model=epsf_model,
        filename=fits_path,
        extra_header=extra_header,
        overwrite=overwrite,
    )
    log.info("[%s] Successfully saved ePSF model to: %s", det_str, saved_path)
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description="Build global ePSF models for Roman WFI detector exposures from MAST observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pid",
        "-p",
        type=int,
        default=1055,
        help="Roman Program (Proposal) ID, e.g. 1055.",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        default="F146",
        help="WFI Optical element / filter name (e.g. F146, F158, F062).",
    )
    parser.add_argument(
        "--detector",
        "-d",
        type=str,
        default="all",
        help="Detector to build ePSF for: 'all' (1-18), single integer '2', 'WFI02', or comma-separated list '1,2,3'.",
    )
    parser.add_argument(
        "--observation",
        "-o",
        type=int,
        default=None,
        help="Observation number (OOO). If omitted, defaults to the 2nd observation in the MAST sorted list.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/ePSF",
        help="Base directory for saving models. Models are written to <output-dir>/pid_<PID>/<FILTER>/.",
    )
    parser.add_argument(
        "--exposure-idx",
        "-e",
        type=int,
        default=1,
        help="Index of exposure to select from the sorted L2 product list (default: 1, matching demo notebook).",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=3,
        help="ePSF oversampling factor.",
    )
    parser.add_argument(
        "--maxiters",
        type=int,
        default=5,
        help="Maximum EPSFBuilder iterations.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if the output FITS model already exists.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing FITS files.",
    )

    args = parser.parse_args()

    # 1. Parse detectors
    detectors = parse_detectors(args.detector)
    filter_name = args.filter.upper()
    output_dir = Path(args.output_dir).resolve()

    log.info("==================================================")
    log.info("Roman ePSF Global Model Batch Builder")
    log.info("Program ID  : %d", args.pid)
    log.info("Filter      : %s", filter_name)
    log.info("Detectors   : %s (%d total)", detectors, len(detectors))
    log.info("Output Dir  : %s", output_dir)
    log.info("==================================================")

    # 2. Initialize MAST session
    mission = init_mast_session()

    # 3. Determine observation number if not explicitly given
    if args.observation is not None:
        observation = args.observation
        log.info("Using user-defined observation number: %d", observation)
    else:
        observation = determine_observation_number(
            mission=mission,
            pid=args.pid,
            filter_name=filter_name,
            probe_detector=detectors[0],
        )

    # 4. Process each detector
    successes = []
    failures = []

    for det in detectors:
        try:
            saved = process_detector(
                detector=det,
                pid=args.pid,
                filter_name=filter_name,
                observation=observation,
                mission=mission,
                output_dir=output_dir,
                exposure_idx=args.exposure_idx,
                oversampling=args.oversampling,
                maxiters=args.maxiters,
                skip_existing=args.skip_existing,
                overwrite=not args.no_overwrite,
            )
            if saved is not None:
                successes.append(det)
            else:
                failures.append(det)
        except Exception as exc:
            log.exception("Error processing detector WFI%02d: %s", det, exc)
            failures.append(det)

    log.info("==================================================")
    log.info("Run Summary:")
    log.info("Successful detectors (%d): %s", len(successes), successes)
    if failures:
        log.warning("Failed/skipped detectors (%d): %s", len(failures), failures)
        sys.exit(1)
    else:
        log.info("All requested detectors processed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
