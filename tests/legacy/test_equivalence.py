"""Tier A of the porting harness: compare each ported item with the original's stored output.

Every test skips with an explicit message when its golden file is missing, so the suite stays green
on a fresh clone and turns red the moment a ported implementation drifts.
"""
from pathlib import Path

import numpy as np
import pytest

from oceanml3d.dynamics import get_dynamics

GOLDEN = Path(__file__).resolve().parent / "golden"

# tolerance per case, with the reason it is not zero
TOLERANCES = {
    "lorenz63": (1e-9, "same RK4, same order of operations; float64 both sides"),
    "lorenz96_2scale": (1e-9, "same RK4 and clamp; numpy vs torch float64 reassociation only"),
    "gaspari_cohn": (1e-12, "closed form, Horner vs expanded polynomial"),
}


def load(case: str) -> dict:
    path = GOLDEN / f"{case}.npz"
    if not path.exists():
        pytest.skip(f"no golden file for '{case}': run tests/legacy/generate_golden.py --fm <legacy repo> "
                    f"--case {case} on a machine where the original code runs")
    return dict(np.load(path, allow_pickle=False))


def test_lorenz63_matches_the_original():
    g = load("lorenz63")
    tol, _ = TOLERANCES["lorenz63"]
    ours = get_dynamics("lorenz63", dt=float(g["dt"])).simulate(len(g["trajectory"]), x0=g["x0"])
    assert np.abs(ours - g["trajectory"]).max() < tol


def test_lorenz96_2scale_matches_the_original():
    g = load("lorenz96_2scale")
    tol, _ = TOLERANCES["lorenz96_2scale"]
    dyn = get_dynamics("lorenz96_2scale", NO=int(g["NO"]), J=int(g["J"]), forcing=float(g["F"]),
                       dt=float(g["dt"]), clip_range=50.0)
    ours = dyn.simulate(len(g["trajectory"]), x0=g["x0"])
    assert np.abs(ours - g["trajectory"]).max() < tol


def test_gaspari_cohn_matches_the_original():
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from oceanml3d.models.ocean.assimilation.model import gaspari_cohn

    g = load("gaspari_cohn")
    tol, _ = TOLERANCES["gaspari_cohn"]
    assert np.abs(gaspari_cohn(g["z"]) - g["values"]).max() < tol


def test_every_case_has_a_documented_tolerance():
    """A golden file without a tolerance entry means someone added a case without saying how close
    'equivalent' has to be."""
    for f in GOLDEN.glob("*.npz"):
        assert f.stem in TOLERANCES, f"add {f.stem} to TOLERANCES with the reason the tolerance is not zero"


def test_harness_detects_drift(tmp_path, monkeypatch):
    """Self-test: the comparison must fail when the implementation moves, otherwise a green suite
    would prove nothing while the golden files are absent."""
    import tests.legacy.test_equivalence as mod

    dyn = get_dynamics("lorenz63", dt=0.01)
    ref = dyn.simulate(50, x0=np.ones(3))
    np.savez(tmp_path / "lorenz63.npz", x0=np.ones(3), trajectory=ref, dt=0.01)
    monkeypatch.setattr(mod, "GOLDEN", tmp_path)
    mod.test_lorenz63_matches_the_original()                      # identical -> passes

    np.savez(tmp_path / "lorenz63.npz", x0=np.ones(3), trajectory=ref * 1.0001, dt=0.01)
    with pytest.raises(AssertionError):
        mod.test_lorenz63_matches_the_original()                  # 1e-4 drift -> caught
