"""Validation / test metrics: pooled RMSE in physical units on ``eval_domain``, loss-independent.

The checkpoint used to be selected on ``val/loss`` -- the training objective, which changes with every
ablation and says nothing in degC or m/s. These tests pin what replaced it.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")


def _dm(variables, catalog, eval_domain=None):
    from oceanml3d.data.datamodule import OceanDataModule

    dm = OceanDataModule(
        variables, catalog, domain={"lat": [-5, 5], "lon": [-5, 5]},
        splits={"train": ["2019-01-01", "2019-01-20"], "val": ["2019-01-21", "2019-01-30"],
                "test": ["2019-01-31", "2019-02-09"]},
        patch={"time": 5, "lat": 16, "lon": 16}, stride={"time": 1, "lat": 12, "lon": 12},
        batch_size=2, num_workers=0, eval_domain=eval_domain)
    dm.setup("fit")
    return dm


def _model(variables, dm, **kw):
    from oceanml3d.models.ocean.nosc.model import NOSCUNet
    from oceanml3d.training.weights import patch_weight

    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    return NOSCUNet(variables, 5, w, widths=(8, 16), optimizer_kw={"lr": 1e-3, "t_max": 1},
                    norm_stats=dm.norm_stats(), **kw)


def test_val_and_test_batches_carry_the_eval_mask_and_predict_does_not(variables, catalog):
    dm = _dm(variables, catalog, eval_domain={"lat": [-3, 3], "lon": [-3, 3]})
    x, mask = next(iter(dm.val_dataloader()))
    assert x.ndim == 5 and mask.shape == (x.shape[0], x.shape[3], x.shape[4])
    coords = dm.eval_datasets["val"].patches.coords(0)
    inside = (np.abs(coords["lat"]) <= 3)[:, None] & (np.abs(coords["lon"]) <= 3)[None, :]
    np.testing.assert_array_equal(mask[0].numpy().astype(bool), inside)
    assert 0 < mask[0].mean() < 1
    assert isinstance(next(iter(dm.predict_dataloader())), torch.Tensor)
    assert isinstance(next(iter(dm.train_dataloader())), torch.Tensor)


def test_metrics_are_pooled_rmse_in_physical_units():
    """Two batches with different errors: the RMSE of the pool, not the mean of batch RMSEs."""
    from oceanml3d.models.ocean.base import BaseOceanModel
    from oceanml3d.variables import VariableSet

    vs = VariableSet.from_config({
        "obs": {"source": "s", "role": "input"},
        "t0": {"source": "s", "role": "target", "group": "temperature"},
        "t1": {"source": "s", "role": "target", "group": "temperature"},
        "u": {"source": "s", "role": "target", "group": "currents"},
    })
    std = np.array([1.0, 2.0, 3.0, 0.5], np.float32)
    m = BaseOceanModel(vs, 1, np.ones((1, 2, 2), np.float32), norm_stats=(np.zeros(4, np.float32), std))
    tgt = torch.zeros(1, 3, 1, 2, 2)
    m._accumulate_errors(torch.full((1, 3, 1, 2, 2), 1.0), tgt, "val")      # normalised error 1
    m._accumulate_errors(torch.full((1, 3, 1, 2, 2), 3.0), tgt, "val")      # normalised error 3
    got = m.eval_metrics(m._eval_sums["val"])
    pooled = np.sqrt((1 + 9) / 2)
    assert got["nrmse"] == pytest.approx(pooled)
    assert got["rmse_t1"] == pytest.approx(pooled * 3.0)
    assert got["rmse_u"] == pytest.approx(pooled * 0.5)
    assert got["rmse_temperature"] == pytest.approx(pooled * np.sqrt((2.0 ** 2 + 3.0 ** 2) / 2))


def test_the_eval_mask_and_nan_targets_are_excluded():
    from oceanml3d.models.ocean.base import BaseOceanModel
    from oceanml3d.variables import VariableSet

    vs = VariableSet.from_config({"obs": {"source": "s", "role": "input"},
                                  "t": {"source": "s", "role": "target"}})
    m = BaseOceanModel(vs, 1, np.ones((1, 2, 2), np.float32))
    out = torch.tensor([[[[[1.0, 100.0], [1.0, 1.0]]]]])
    tgt = torch.tensor([[[[[0.0, 0.0], [float("nan"), 0.0]]]]])
    m._eval_mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])              # hides the 100
    m._accumulate_errors(out, tgt, "test")
    assert m.eval_metrics(m._eval_sums["test"])["nrmse"] == pytest.approx(1.0)


def test_training_logs_the_metrics_and_checkpoints_on_nrmse(variables, catalog, tmp_path):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    dm = _dm(variables, catalog, eval_domain={"lat": [-4, 4], "lon": [-4, 4]})
    model = _model(variables, dm)
    ckpt = ModelCheckpoint(dirpath=tmp_path, monitor="val/nrmse", mode="min")
    trainer = Trainer(max_epochs=2, accelerator="cpu", logger=False, callbacks=[ckpt],
                      limit_train_batches=2, limit_val_batches=2, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=dm)
    logged = trainer.callback_metrics
    for key in ("val/nrmse", "val/rmse_u_drifter", "val/rmse_v_drifter", "val/loss"):
        assert key in logged and torch.isfinite(logged[key]), key
    assert ckpt.best_model_path and ckpt.best_model_score is not None
    trainer.test(model, datamodule=dm, verbose=False)
    assert "test/nrmse" in trainer.callback_metrics


def test_overlapping_val_and_test_are_refused_unless_declared():
    from omegaconf import OmegaConf

    from oceanml3d.config_schema import validate_config

    base = OmegaConf.load(__import__("pathlib").Path(__file__).parents[1] / "config/training/default.yaml")
    cfg = OmegaConf.create({
        "training": base,
        "data": {"domain": {"lat": [0, 10], "lon": [0, 10]}, "eval_domain": {"lat": [1, 9], "lon": [1, 9]},
                 "patch": {"time": 5, "lat": 8, "lon": 8}, "stride": {"time": 1, "lat": 4, "lon": 4},
                 "splits": {"train": {"time": ["2010-01-01", "2017-12-15"]},
                            "val": {"time": ["2018-01-01", "2018-12-31"]},
                            "test": {"time": ["2018-12-20", "2020-01-10"]}}},
        "export": {"enabled": False},
    })
    assert any("test starts inside the val window" in p for p in validate_config(cfg))
    cfg.data.splits_overlap_ok = True
    assert not any("val window" in p for p in validate_config(cfg))
    cfg.data.eval_domain = {"lat": [1, 12], "lon": [1, 9]}
    assert any("eval_domain.lat" in p for p in validate_config(cfg))


def test_the_shipped_osse_splits_are_disjoint_with_gaps():
    import pandas as pd
    import yaml

    d = yaml.safe_load((__import__("pathlib").Path(__file__).parents[1] / "config/data/osse3d_gs21.yaml").read_text())
    s = {k: [pd.Timestamp(x) for x in v["time"]] for k, v in d["splits"].items()}
    window = d["patch"]["time"]
    assert (s["val"][0] - s["train"][1]).days > window
    assert (s["test"][0] - s["val"][1]).days > window
    assert (pd.Timestamp("2019-01-01") - s["test"][0]).days >= window - 1, "2019 must be fully covered"
    assert not d.get("splits_overlap_ok", False)


# --- patch/stride `auto`: fitted to the grid, whatever the resolution ------------------------------

@pytest.mark.parametrize("cells,patch,stride", [(145, 144, 136), (49, 48, 40), (16, 16, 8)])
def test_auto_patch_matches_the_hand_set_native_values_and_fits_quarter_degree(cells, patch, stride):
    from omegaconf import OmegaConf

    from oceanml3d.cli import resolve_auto_patch

    cfg = OmegaConf.create({"data": {"patch": {"time": 11, "lat": "auto", "lon": "auto"},
                                     "stride": {"time": 1, "lat": "auto", "lon": "auto"},
                                     "patch_multiple": 16},
                            "training": {"rec_weight": {"crop": {"time": 0, "lat": 4, "lon": 4}}}})
    OmegaConf.set_struct(cfg, True)                    # as Hydra hands it over
    resolve_auto_patch(cfg, None, None, sizes={"lat": cells, "lon": cells})
    assert (cfg.data.patch.lat, cfg.data.stride.lat) == (patch, stride)
    assert (cfg.data.patch.lon, cfg.data.stride.lon) == (patch, stride)
    assert cfg.data.patch.time == 11 and cfg.data.stride.time == 1


def test_auto_patch_refuses_a_domain_smaller_than_the_multiple():
    from omegaconf import OmegaConf

    from oceanml3d.cli import resolve_auto_patch

    cfg = OmegaConf.create({"data": {"patch": {"time": 5, "lat": "auto", "lon": 16},
                                     "stride": {"time": 1, "lat": "auto", "lon": 8}},
                            "training": {"rec_weight": {"crop": {"lat": 4}}}})
    with pytest.raises(SystemExit, match="fewer than data.patch_multiple"):
        resolve_auto_patch(cfg, None, None, sizes={"lat": 12, "lon": 40})

