"""The product format is a contract between two repos: pin the shared file and its behaviour."""
import hashlib
from pathlib import Path

import pytest

from oceanml3d.inference.product_contract import (
    PRODUCT_FORMAT_VERSION,
    check_manifest,
    parse_variable_name,
    validate_manifest,
)

CONTRACT = Path(__file__).resolve().parents[1] / "oceanml3d" / "inference" / "product_contract.py"
PINNED_SHA256 = "dd44f0e198d0122b88207509801ed5c60a7e2fa36172151d89d78a622fe88846"


def test_contract_file_is_pinned():
    """If this fails, product_contract.py changed: copy it to oceanml3d-eval, bump the version
    and update the hash in docs/product_format.md of BOTH repos."""
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == PINNED_SHA256


def _manifest(**over):
    doc = {"format_version": PRODUCT_FORMAT_VERSION, "name": "xp", "path": "/tmp/daily",
           "pattern": r"xp_(\\d{4})-(\\d{2})-(\\d{2})\\.nc",
           "variables": {"u": {"standard_name": "eastward_sea_water_velocity"}},
           "coords": {"lat": "lat", "lon": "lon", "time": "time"}}
    doc.update(over)
    return doc


def test_valid_manifest():
    assert validate_manifest(_manifest()) == []


@pytest.mark.parametrize("over,expected", [
    ({"variables": {}}, "non-empty mapping"),
    ({"pattern": "([unclosed"}, "not a valid regex"),
    ({"coords": {"lat": "lat"}}, "'coords' must map"),
    ({"first_date": "01/02/2019"}, "YYYY-MM-DD"),
    ({"depth_m": "fifteen"}, "must be a number"),
    ({"format_version": PRODUCT_FORMAT_VERSION + 1}, "newer than this reader"),
    ({"typo_key": 1}, "unknown keys"),
    ({"variables": {"thetao_d12": {"depth_index": 3}}}, "contradicts the name suffix"),
])
def test_invalid_manifests(over, expected):
    assert any(expected in p for p in validate_manifest(_manifest(**over)))


def test_missing_required_key_raises():
    with pytest.raises(ValueError, match="missing required key 'path'"):
        check_manifest({k: v for k, v in _manifest().items() if k != "path"})


def test_strict_flags_non_canonical_base():
    assert validate_manifest(_manifest(variables={"foo_d01": {}})) == []
    assert any("not canonical" in p for p in validate_manifest(_manifest(variables={"foo_d01": {}}), strict=True))


def test_parse_variable_name():
    assert parse_variable_name("thetao_d12") == ("thetao", 12)
    assert parse_variable_name("u") == ("u", None)
