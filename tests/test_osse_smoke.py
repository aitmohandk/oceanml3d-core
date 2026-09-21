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
    # 11 inputs, not 12: `bathy` is set aside (`bathy: null`), and `ablation=bathy` puts it back.
    assert len(variables.targets) == 10 and len(variables.inputs) == 11 and len(variables.names) == 23
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


def test_auto_patch_is_read_from_the_data(osse_dir):
    """`auto` resolves from the grid on disk (here 20 x 20 cells on the smoke domain) and trains."""
    from oceanml3d.cli import build_datamodule
    from oceanml3d.config_schema import validate_config

    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="main", overrides=[
            "experiment=osse3d_smoke", f"paths.root={osse_dir}",
            "data.splits.train.time=[2019-01-01,2019-01-20]",
            "data.splits.val.time=[2019-01-21,2019-01-30]",
            "data.splits.test.time=[2019-01-31,2019-02-09]",
            "data.patch.lat=auto", "data.patch.lon=auto", "data.stride.lat=auto", "data.stride.lon=auto",
            "data.patch_multiple=8"])
    assert validate_config(cfg) == [] or not any("patch" in p for p in validate_config(cfg))
    variables, catalog = build_variables(cfg), build_catalog(cfg)
    prepare_observations(cfg, catalog)
    dm = build_datamodule(cfg, variables, catalog)
    n = open_variable_set(variables, catalog, {k: slice(*v) for k, v in cfg.data.domain.items()}).sizes["lat"]
    assert cfg.data.patch.lat == (n // 8) * 8 and cfg.data.stride.lat == cfg.data.patch.lat - 2 * 4
    dm.setup("fit")
    assert tuple(dm.datasets["train"][0].shape[-2:]) == (cfg.data.patch.lat, cfg.data.patch.lon)


# --- a derived file that does not match the truth is rebuilt, not reused ----------------------------
# The run that exposed this reused a virtual ARGO file with no date in common with the truth: 42 lines
# of "100% absent", then training went ahead on an ARGO input that was zero everywhere.

def _fresh(tmp_path_factory):
    from make_synthetic_data import make, make_osse

    out = tmp_path_factory.mktemp("osse_stale")
    make(out, days=40, n=48)
    make_osse(out, days=40, n=48, n_depth=3)
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="main", overrides=["experiment=osse3d_smoke", f"paths.root={out}",
                                                     "data.splits.train.time=[2019-01-01,2019-01-20]",
                                                     "data.splits.val.time=[2019-01-21,2019-01-30]",
                                                     "data.splits.test.time=[2019-01-31,2019-02-09]"])
    # A coverage table, as scripts/prepare/argo_profiles.py writes it: without one, prepare-obs uses
    # the synthetic ARGO file as it is and there is nothing to rebuild from.
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 60
    table = pd.DataFrame({"time": pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 39, n), "D"),
                          "lat": rng.uniform(-9, 9, n), "lon": rng.uniform(-9, 9, n),
                          "profile_id": [f"19000{i}_{i}" for i in range(n)],
                          "d00": 1.0, "d01": 1.0, "d02": 1.0})
    (out / "argo").mkdir(exist_ok=True)
    table.to_csv(out / "argo" / "argo_profiles_gs.csv", index=False)
    return out, cfg, build_catalog(cfg)


def test_a_virtual_argo_file_on_another_time_axis_is_rebuilt(tmp_path_factory, capsys):
    import pandas as pd
    import xarray as xr

    out, cfg, catalog = _fresh(tmp_path_factory)
    prepare_observations(cfg, catalog)
    argo = out / "synthetic_argo_virtual.nc"
    good = xr.open_dataset(argo).load()
    good.close()
    good.assign_coords(time=good.time + pd.Timedelta(days=3650)).to_netcdf(argo)   # a decade off
    capsys.readouterr()

    prepare_observations(cfg, catalog)
    log = capsys.readouterr().out
    assert "rebuilding" in log and "time axis" in log, log
    with xr.open_dataset(argo) as rebuilt:
        assert pd.DatetimeIndex(rebuilt.time.values).equals(pd.DatetimeIndex(good.time.values))


def test_a_valid_virtual_argo_file_is_reused_and_the_log_says_so(tmp_path_factory, capsys):
    out, cfg, catalog = _fresh(tmp_path_factory)
    prepare_observations(cfg, catalog)
    capsys.readouterr()
    prepare_observations(cfg, catalog)
    assert "virtual ARGO: reusing" in capsys.readouterr().out


def test_a_file_sharing_no_date_with_the_task_stops_the_run(tmp_path_factory):
    import pandas as pd
    import xarray as xr

    out, cfg, catalog = _fresh(tmp_path_factory)
    prepare_observations(cfg, catalog)
    argo = out / "synthetic_argo_virtual.nc"
    ds = xr.open_dataset(argo).load()
    ds.close()
    ds.assign_coords(time=ds.time + pd.Timedelta(days=3650)).to_netcdf(argo)
    variables = build_variables(cfg)
    with pytest.raises(ValueError, match="share no date with the rest of the task") as err:
        open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5),
                                               "time": slice("2019-01-01", "2019-01-10")})
    assert "synthetic_argo_virtual.nc" in str(err.value) and "delete it" in str(err.value)
