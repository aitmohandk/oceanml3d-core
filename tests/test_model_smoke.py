import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")


@pytest.fixture
def datamodule(variables, catalog):
    from oceanml3d.data.datamodule import OceanDataModule

    dm = OceanDataModule(
        variables, catalog, domain={"lat": [-5, 5], "lon": [-5, 5]},
        splits={"train": ["2019-01-01", "2019-01-20"], "val": ["2019-01-21", "2019-01-30"], "test": ["2019-01-31", "2019-02-09"]},
        patch={"time": 5, "lat": 16, "lon": 16}, stride={"time": 1, "lat": 12, "lon": 12}, batch_size=2, num_workers=0)
    dm.setup("fit")
    return dm


@pytest.mark.parametrize("trunk", ["monai", "nosc"])
def test_nosc_forward_shapes(variables, datamodule, trunk):
    from oceanml3d.models.ocean.nosc.model import NOSCUNet
    from oceanml3d.training.weights import patch_weight

    if trunk == "monai":
        pytest.importorskip("monai")
    w = patch_weight("triangular", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    model = NOSCUNet(variables, 5, w, widths=(8, 16, 32), optimizer_kw={"lr": 1e-3, "t_max": 1},
                     norm_stats=datamodule.norm_stats(), trunk=trunk)
    batch = next(iter(datamodule.train_dataloader()))
    out = model(batch)
    assert out.shape == (2, 2, 5, 16, 16)
    loss = model.step(batch, "train")
    assert torch.isfinite(loss)


def test_nosc_rejects_a_checkpoint_from_the_other_trunk(variables, datamodule):
    """Old checkpoints carry no `trunk` hyperparameter, so Lightning would rebuild them as MONAI and
    fail on an unreadable size mismatch. `net.inc.` only exists in UNetNosc."""
    from oceanml3d.models.ocean.nosc.model import NOSCUNet
    from oceanml3d.training.weights import patch_weight

    pytest.importorskip("monai")
    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    nosc = NOSCUNet(variables, 5, w, widths=(8, 16), optimizer_kw={"lr": 1e-3, "t_max": 1}, trunk="nosc")
    ckpt = {"state_dict": nosc.state_dict()}
    with pytest.raises(RuntimeError, match="ablation=trunk_nosc"):
        NOSCUNet(variables, 5, w, widths=(8, 16), optimizer_kw={"lr": 1e-3, "t_max": 1}).on_load_checkpoint(ckpt)
    nosc.on_load_checkpoint(ckpt)                       # the right trunk accepts it


@pytest.mark.parametrize("trunk", ["monai", "nosc"])
def test_the_trunk_can_actually_fit_a_batch(variables, datamodule, trunk):
    """A guard against a model that runs, exports, and has learnt nothing.

    It is not hypothetical for the MONAI trunk: MONAI wraps the output convolution and every
    resblock's second convolution in ``zero_module``, so a freshly built trunk is *exactly* the zero
    map (pinned in ``test_monai_unet2d::test_zero_initialisation_is_inherited_from_monai``). Every
    other test here would pass on a model stuck in that state -- shapes are right, the loss is
    finite, the export writes its files. Overfitting one batch is the cheapest thing that cannot.

    The loss is computed directly rather than through ``model.step`` so that no ``self.log`` call
    happens outside a Trainer, and masked so that NaN targets do not poison the gradient.
    """
    from oceanml3d.models.ocean.nosc.model import NOSCUNet
    from oceanml3d.training.weights import patch_weight

    if trunk == "monai":
        pytest.importorskip("monai")
    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    model = NOSCUNet(variables, 5, w, widths=(8, 16), optimizer_kw={"lr": 1e-3, "t_max": 1},
                     norm_stats=datamodule.norm_stats(), trunk=trunk)
    batch = next(iter(datamodule.train_dataloader()))
    tgt = model.targets(batch)
    mask = torch.isfinite(tgt)
    assert mask.any(), "the synthetic fixture has no finite target: the test would be vacuous"

    def mse():
        return ((model(batch)[mask] - tgt[mask]) ** 2).mean()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = float(mse())
    for _ in range(30):
        opt.zero_grad()
        loss = mse()
        loss.backward()
        opt.step()
    last = float(mse())

    assert last < 0.9 * first, f"{trunk}: loss went {first:.4f} -> {last:.4f}; the model is not learning"
    assert float(model(batch).std()) > 0, f"{trunk}: the output is spatially constant after training"


@pytest.mark.slow
def test_train_predict_export(variables, datamodule, tmp_path):
    from pytorch_lightning import Trainer

    from oceanml3d.inference.export import export_variable_names
    from oceanml3d.inference.predict import predict_and_export
    from oceanml3d.models.ocean.nosc.model import NOSCUNet
    from oceanml3d.training.weights import patch_weight

    xr = pytest.importorskip("xarray")
    np = pytest.importorskip("numpy")

    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    model = NOSCUNet(variables, 5, w, widths=(8, 16), optimizer_kw={"lr": 1e-3, "t_max": 1},
                     norm_stats=datamodule.norm_stats())
    trainer = Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, limit_train_batches=2)
    trainer.fit(model, datamodule=datamodule)
    manifest = predict_and_export(model, datamodule, trainer, tmp_path, "smoke", depth_m=15)
    assert manifest.exists()
    files = sorted((tmp_path / "daily").glob("smoke_*.nc"))
    assert len(files) == 10

    # The two assertions above pass on a degenerate product, and "all zeros" is not the signature to
    # look for: `predict_step` denormalises, so a trunk stuck at the zero map exports `mean` -- a
    # spatially *constant*, perfectly finite field. Zero spatial variance is the tell.
    with xr.open_dataset(files[0]) as ds:
        for short, _std, _units in export_variable_names(variables).values():
            assert short in ds, f"{short} missing from the exported product"
            values = ds[short].values
            assert np.isfinite(values).any(), f"{short} is entirely non-finite"
            assert float(np.nanstd(values)) > 0, f"{short} is spatially constant (a dead trunk exports its mean)"


def test_fourdvarnet_forward(variables, catalog):
    from oceanml3d.data.datamodule import OceanDataModule
    from oceanml3d.models.ocean.fourdvarnet.model import FourDVarNet
    from oceanml3d.training.weights import patch_weight
    from oceanml3d.variables import VariableSet

    vs = VariableSet.from_config({
        "ssh_obs": {"source": "ssh", "var_name": "zos", "role": "input"},
        "u10": {"source": "era5", "role": "input"},
        "zos": {"source": "ssh", "var_name": "zos", "role": "target"}})
    dm = OceanDataModule(vs, catalog, domain={"lat": [-5, 5], "lon": [-5, 5]},
                         splits={"train": ["2019-01-01", "2019-01-20"], "val": ["2019-01-21", "2019-01-30"], "test": ["2019-01-31", "2019-02-09"]},
                         patch={"time": 3, "lat": 16, "lon": 16}, stride={"time": 1, "lat": 12, "lon": 12}, batch_size=2, num_workers=0)
    dm.setup("fit")
    w = patch_weight("constant", {"time": 3, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    model = FourDVarNet(vs, 3, w, obs_map={"zos": "ssh_obs"}, n_step=3, prior_hidden=8, grad_hidden=8,
                        optimizer_kw={"lr": 1e-3, "t_max": 1}, norm_stats=dm.norm_stats())
    batch = next(iter(dm.train_dataloader()))
    assert model(batch).shape == (2, 1, 3, 16, 16)
    loss = model.step(batch, "train")
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    assert model(batch).shape == (2, 1, 3, 16, 16)


@pytest.mark.parametrize("name,kw", [("passthrough", {"source": {"u_drifter": "ugos", "v_drifter": "vgos"}}),
                                     ("linear", {}), ("climatology", {})])
def test_baseline_models_share_the_interface(variables, datamodule, name, kw):
    """The trivial baselines exercise the same contract as nosc_unet: if an interface silently
    assumed a U-Net, these would break."""
    from oceanml3d.registry import get_model
    from oceanml3d.training.weights import patch_weight

    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    model = get_model(name)(variables, 5, w, optimizer_kw={"lr": 1e-2, "t_max": 1},
                            norm_stats=datamodule.norm_stats(), **kw)
    batch = next(iter(datamodule.train_dataloader()))
    out = model(batch)
    assert out.shape == (2, 2, 5, 16, 16)
    loss = model.step(batch, "train")
    loss.backward()
    assert torch.isfinite(loss)
    assert model.predict_step(batch, 0).shape == out.shape


def test_passthrough_rejects_unknown_source(variables):
    import numpy as np

    from oceanml3d.models.ocean.baselines.model import Passthrough

    w = np.ones((5, 16, 16), np.float32)
    with pytest.raises(ValueError, match="not input variables"):
        Passthrough(variables, 5, w, source={"u_drifter": "not_a_variable"})
