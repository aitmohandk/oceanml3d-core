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
