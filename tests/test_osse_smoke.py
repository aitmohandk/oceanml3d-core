"""OSSE-3D path end to end on synthetic data (torch parts skipped without torch)."""
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from oceanml3d.cli import CONFIG_DIR, build_catalog, build_variables, prepare_observations
from oceanml3d.data.open import open_variable_set


@pytest.fixture(scope="session")
def osse_dir(tmp_path_factory):
    from make_synthetic_data import make, make_osse

    out = tmp_path_factory.mktemp("osse")
    make(out, days=40, n=48)
    make_osse(out, days=40, n=48, n_depth=3)
    return out


@pytest.fixture(scope="session")
def osse_cfg(osse_dir):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="main", overrides=["experiment=osse3d_smoke", f"paths.root={osse_dir}",
                                                      "data.splits.train.time=[2019-01-01,2019-01-20]",
                                                      "data.splits.val.time=[2019-01-21,2019-01-30]",
                                                      "data.splits.test.time=[2019-01-31,2019-02-09]"])


def test_prepare_and_open(osse_dir, osse_cfg):
    catalog = build_catalog(osse_cfg)
    prepare_observations(osse_cfg, catalog)
    assert (osse_dir / "synthetic_pseudo_obs.nc").exists()
    variables = build_variables(osse_cfg)
    assert len(variables.targets) == 10 and len(variables.inputs) == 12 and len(variables.names) == 24
    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5), "time": slice("2019-01-01", "2019-01-10")})
    assert list(da.channel.values)[-1] == "vo_d02"
    argo = da.sel(channel="argo_thetao_d01").values
    assert 0 < np.isfinite(argo).mean() < 0.2                     # sparse in-situ input
    assert np.array_equal(da.sel(channel="argo_mask_d01").values > 0, np.isfinite(argo))   # presence channel
    sst = da.sel(channel="sst_in").values
    assert 0.5 < np.isfinite(sst).mean() < 0.9                    # cloud-masked SST
    assert set(np.unique(da.sel(channel="mask_jason3").values)) <= {0.0, 1.0}
    assert np.isfinite(da.sel(channel="thetao_d02").values).all()  # dense truth target


def test_osse_model_forward(osse_cfg, osse_dir):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from oceanml3d.cli import build_datamodule, build_model

    variables, catalog = build_variables(osse_cfg), build_catalog(osse_cfg)
    dm = build_datamodule(osse_cfg, variables, catalog)
    dm.setup("fit")
    for ablation in [None, "heads", "attention", "uncertainty", "temporal_conv3d", "gradsolver"]:
        cfg = osse_cfg
        if ablation:
            with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
                cfg = compose(config_name="main", overrides=["experiment=osse3d_smoke", f"paths.root={osse_dir}", f"ablation={ablation}"])
        model = build_model(cfg, variables, dm.norm_stats())
        batch = next(iter(dm.train_dataloader()))
        out = model(batch)
        assert out.shape[1:] == (10, 5, 32, 32)
        assert torch.isfinite(model.step(batch, "train"))


@pytest.mark.slow
def test_osse_train_predict_export_produces_a_valid_non_trivial_product(osse_cfg, osse_dir, tmp_path):
    """The whole gridded path in one go: prepare-obs, train, predict, stitch, export, validate.

    Every step of this chain was covered in isolation and none of them together, so nothing caught a
    product that was well-formed and empty of information. Two things are checked beyond "files
    exist": the manifest satisfies the contract shared byte for byte with `oceanml3d-eval`, and each
    target varies in space -- a trunk stuck at MONAI's zero initialisation exports its per-channel
    mean, which is finite, correctly shaped and spatially constant.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    import xarray as xr
    import yaml
    from pytorch_lightning import Trainer

    from oceanml3d.cli import build_datamodule, build_model
    from oceanml3d.inference.export import export_variable_names
    from oceanml3d.inference.predict import predict_and_export
    from oceanml3d.inference.product_contract import validate_manifest

    variables, catalog = build_variables(osse_cfg), build_catalog(osse_cfg)
    prepare_observations(osse_cfg, catalog)
    dm = build_datamodule(osse_cfg, variables, catalog)
    dm.setup("fit")
    model = build_model(osse_cfg, variables, dm.norm_stats())

    trainer = Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, limit_train_batches=2, limit_val_batches=1)
    trainer.fit(model, datamodule=dm)

    manifest_path = predict_and_export(model, dm, trainer, tmp_path, "osse_smoke", depth_m=None)
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text())
    assert validate_manifest(manifest, strict=True) == []

    daily = sorted((tmp_path / "daily").glob("osse_smoke_*.nc"))
    assert daily, "the manifest was written but no daily file was"
    names = export_variable_names(variables)
    with xr.open_dataset(daily[0]) as ds:
        for short, std, units in names.values():
            assert short in ds, f"{short} missing from the product"
            values = ds[short].values
            assert np.isfinite(values).any(), f"{short} is entirely non-finite"
            assert float(np.nanstd(values)) > 0, f"{short} is spatially constant (a dead trunk exports its mean)"
            if std:
                assert ds[short].attrs.get("standard_name") == std
            if units:
                assert ds[short].attrs.get("units") == units

    # and the statistics that denormalised it are the ones training used
    ckpt = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(ckpt)
    assert ckpt["oceanml3d"]["variables"] == list(variables.names)
    assert torch.isfinite(torch.as_tensor(ckpt["oceanml3d"]["norm_stats"]["std"])).all()
