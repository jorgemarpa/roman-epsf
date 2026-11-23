"""Tests for build_program_epsf.py CLI helper functions."""

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_program_epsf import extract_observation_from_filename, parse_detectors


def test_parse_detectors():
    assert parse_detectors("all") == list(range(1, 19))
    assert parse_detectors(None) == list(range(1, 19))
    assert parse_detectors("2") == [2]
    assert parse_detectors("WFI04") == [4]
    assert parse_detectors("wfi07") == [7]
    assert parse_detectors("1,3,5") == [1, 3, 5]
    assert parse_detectors("1-4,8") == [1, 2, 3, 4, 8]


def test_extract_observation_from_filename():
    filename = "r0105501001001006001_0001_wfi02_f146_cal.asdf"
    assert extract_observation_from_filename(filename) == 6

    filename2 = "r0034001002001001001_0001_wfi01_f146_cal.asdf"
    assert extract_observation_from_filename(filename2) == 1

    assert extract_observation_from_filename("invalid_filename.asdf") is None
