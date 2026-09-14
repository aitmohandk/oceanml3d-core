import pytest

from oceanml3d.catalog import Catalog


def test_resolve_key_and_root(tmp_path):
    c = Catalog({"ssh": "sub/ssh.nc", "era5": {"path": "/abs/era5.nc"}}, root=tmp_path)
    assert c.resolve("ssh") == tmp_path / "sub" / "ssh.nc"
    assert str(c.resolve("era5")) == "/abs/era5.nc"
    assert str(c.resolve("/other/file.nc")) == "/other/file.nc"
    with pytest.raises(KeyError):
        c.resolve("unknown_key")
