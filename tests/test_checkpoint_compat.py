"""Backward compatibility tests: verify checkpoints reproduce stored results."""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oceanml3d.legacy.evaluation.metrics import rmse
from oceanml3d.legacy.models.solver import TweedieSolver

EXP_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments")


def _evaluate_checkpoint(exp_dir, device="cpu"):
    """Load checkpoint_stage2.pt and evaluate on test data."""
    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.exists(results_path):
        return None, "no results.json"

    with open(results_path) as f:
        expected = json.load(f)

    ckpt_path = os.path.join(exp_dir, "checkpoint_stage2.pt")
    if not os.path.exists(ckpt_path):
        return None, "no checkpoint_stage2.pt"

    hc = expected["config"]["hidden_channels"]
    model = TweedieSolver(
        state_dim=3, hidden_channels=hc,
        time_emb_dim=64, use_obs=True, use_energy=True,
        nu=1.0, K_inner=5, N_outer=10, dropout=0.1,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    # Load datasets
    datasets_path = os.path.join(EXP_DIR, "datasets.pt")
    if not os.path.exists(datasets_path):
        return None, "no datasets.pt"
    datasets = torch.load(datasets_path, map_location=device)

    # Evaluate
    def _eval(ds):
        rmses = []
        for i in range(len(ds)):
            w = ds[i]
            obs = w["obs"].unsqueeze(0).to(device)
            pred = model(obs).detach().cpu().numpy()[0]
            truth = w["true_state"].numpy()
            rmses.append(rmse(pred, truth))
        all_rmse = np.stack(rmses, axis=0)
        return float(np.mean(all_rmse))

    cs1_rmse = _eval(datasets["test_cs1"])
    cs2_rmse = _eval(datasets["test_cs2"])

    return {"cs1": cs1_rmse, "cs2": cs2_rmse}, expected


def test_b1_small_unet_checkpoint():
    """B1_small_unet: checkpoint reproduces stored CS1=0.097, CS2=0.110."""
    exp_dir = os.path.join(EXP_DIR, "B1_small_unet")
    results, expected = _evaluate_checkpoint(exp_dir)
    if results is None:
        pytest.skip(f"Checkpoint or datasets not available: {expected}")

    expected_cs1 = expected["fm_cs1"]["mean"]
    expected_cs2 = expected["fm_cs2"]["mean"]

    assert abs(results["cs1"] - expected_cs1) < 0.01, \
        f"CS1 RMSE mismatch: got {results['cs1']:.4f}, expected {expected_cs1:.4f}"
    assert abs(results["cs2"] - expected_cs2) < 0.01, \
        f"CS2 RMSE mismatch: got {results['cs2']:.4f}, expected {expected_cs2:.4f}"
    print(f"  B1_small_unet: CS1={results['cs1']:.4f} (expected {expected_cs1:.4f}), "
          f"CS2={results['cs2']:.4f} (expected {expected_cs2:.4f})")


def test_c1_longer_train_checkpoint():
    """C1_longer_train: checkpoint reproduces stored CS1=0.175, CS2=0.179."""
    exp_dir = os.path.join(EXP_DIR, "C1_longer_train")
    results, expected = _evaluate_checkpoint(exp_dir)
    if results is None:
        pytest.skip(f"Checkpoint or datasets not available: {expected}")

    expected_cs1 = expected["fm_cs1"]["mean"]
    expected_cs2 = expected["fm_cs2"]["mean"]

    assert abs(results["cs1"] - expected_cs1) < 0.01, \
        f"CS1 RMSE mismatch: got {results['cs1']:.4f}, expected {expected_cs1:.4f}"
    assert abs(results["cs2"] - expected_cs2) < 0.01, \
        f"CS2 RMSE mismatch: got {results['cs2']:.4f}, expected {expected_cs2:.4f}"
    print(f"  C1_longer_train: CS1={results['cs1']:.4f} (expected {expected_cs1:.4f}), "
          f"CS2={results['cs2']:.4f} (expected {expected_cs2:.4f})")


def test_lightning_module_produces_same_output():
    """Lit4DVarNetFM.forward() == TweedieSolver.forward() for same weights."""
    from omegaconf import OmegaConf

    from oceanml3d.legacy.conf.schema import ExperimentConfig
    from oceanml3d.legacy.training.lightning_module import LitModel as Lit4DVarNetFM

    cfg = OmegaConf.structured(ExperimentConfig())
    model = TweedieSolver(state_dim=3, hidden_channels=[32, 64, 128], time_emb_dim=64)
    lit = Lit4DVarNetFM(model, cfg, stage=1)

    x = torch.randn(1, 100, 3)
    with torch.no_grad():
        torch.manual_seed(42)
        out_model = model(x)
        torch.manual_seed(42)
        out_lit = lit(x)

    assert torch.allclose(out_model, out_lit), "LitModule forward != model forward"


def _nosc(variables, norm_stats):
    from oceanml3d.models.ocean.nosc.model import NOSCUNet
    from oceanml3d.training.weights import patch_weight

    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    return NOSCUNet(variables, 5, w, widths=(8, 16), optimizer_kw={"lr": 1e-3, "t_max": 1},
                    norm_stats=norm_stats, trunk="nosc")


def test_norm_stats_travel_with_the_weights(variables, tmp_path):
    """`predict_step` denormalises with `self.norm_stats`, and those come from the *train* split. A
    prediction run that recomputes them from whatever splits, domain or variables the current config
    declares can therefore denormalise with statistics the model was never trained under -- and
    write a product that looks entirely normal. Storing them in the checkpoint makes the pair
    inseparable."""
    n = len(variables)
    stats = (np.linspace(-1, 1, n).astype(np.float32), np.linspace(0.5, 3, n).astype(np.float32))
    trained = _nosc(variables, stats)

    ckpt = {"state_dict": trained.state_dict()}
    trained.on_save_checkpoint(ckpt)
    assert ckpt["oceanml3d"]["variables"] == list(variables.names)

    wrong = (np.zeros(n, dtype=np.float32), np.full(n, 7.0, dtype=np.float32))
    loaded = _nosc(variables, wrong)
    loaded.on_load_checkpoint(ckpt)
    loaded.load_state_dict(ckpt["state_dict"])
    assert np.allclose(loaded.norm_stats[0], stats[0])
    assert np.allclose(loaded.norm_stats[1], stats[1])

    # and it must survive a real round-trip under weights_only=True, which is how predict loads
    path = tmp_path / "m.ckpt"
    torch.save(ckpt, path)
    back = torch.load(path, map_location="cpu", weights_only=True)
    assert back["oceanml3d"]["norm_stats"]["mean"] == ckpt["oceanml3d"]["norm_stats"]["mean"]


def test_a_checkpoint_from_another_channel_layout_is_refused(variables):
    n = len(variables)
    model = _nosc(variables, (np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)))
    ckpt = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(ckpt)
    ckpt["oceanml3d"]["variables"] = ["something", "else"]
    with pytest.raises(ValueError, match="different channel layout"):
        model.on_load_checkpoint(ckpt)
