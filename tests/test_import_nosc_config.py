import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_nosc_config import convert  # noqa: E402

LEGACY = yaml.safe_load("""
paths: {ssh: /Odyssey/private/data/SSH_L4_CMEMS_4th.nc}
multivar:
  ssh:   {var_path: "${paths.ssh}", var_name: zos, input_arch: prior_input, output_arch: no_output, broadcast_time: False}
  sst:   {var_path: /Odyssey/private/data/SST_L4_4th.nc, var_name: analysed_sst, input_arch: prior_input, output_arch: no_output, broadcast_time: False, transfo: sst_transfo}
  bathy: {var_path: /elsewhere/bathy_4th.nc, var_name: deptho, input_arch: prior_input, output_arch: no_output, broadcast_time: True}
  u_drifter: {var_path: /Odyssey/private/data/drifters.nc, var_name: u, input_arch: no_input, output_arch: full_output, broadcast_time: False, head_group: currents, mystery_key: 7}
domain:
  train: {lat: {_target_: builtins.slice, _args_: [-70, 70]}, lon: {_target_: builtins.slice, _args_: [-180, 180]}}
  test:  {lat: {_target_: builtins.slice, _args_: [-60, 60]}, lon: {_target_: builtins.slice, _args_: [-170, 170]}}
datamodule:
  domains:
    train: {time: {_target_: builtins.slice, _args_: ['2010-01-01', '2017-12-31']}}
    val:   {time: {_target_: builtins.slice, _args_: ['2018-01-01', '2018-12-31']}}
    test:  {time: {_target_: builtins.slice, _args_: ['2019-01-01', '2019-12-31']}}
  xrds_kw: {patch_dims: {time: 11, lat: 560, lon: 1440}, strides: {time: 1, lat: 552, lon: 1432}}
  dl_kw: {batch_size: 2, num_workers: 8}
  norm_stats: [[0.1], [0.2]]
trainer: {max_epochs: 150, gradient_clip_val: 0.5, limit_train_batches: 102}
model:
  _target_: contrib.multivar.multivar_models_unet_mae.MultivarUNet_mae
  grad_loss_weight: 20.0
  loss_group_mode: group_mean
  opt_fn: {lr: 0.0001}
  rec_weight: {_target_: contrib.multivar.multivar_utils.get_multivar_mapping_wei, crop: {time: 0, lat: 4, lon: 4}, offset: 1}
  solver: {model_channels: 64, channel_mult: [1, 2, 4], dropout: 0.1}
entrypoints: [{_target_: pytorch_lightning.seed_everything, seed: 333}]
unknown_block: {a: 1}
""")

CATALOG = {"datasets": {"ssh_l4_duacs_4th": {"path": "ssh_l4/SSH_L4_CMEMS_4th.nc"},
                        "sst_l4_odyssea_4th": "sst/SST_L4_4th.nc",
                        "drifters_aoml_15m_4th": "drifters/drifters.nc"}}


def test_roles_sources_and_transforms():
    data, xp, notes = convert(LEGACY, CATALOG)
    v = data["variables"]
    assert v["ssh"] == {"source": "ssh_l4_duacs_4th", "role": "input", "var_name": "zos"}
    assert v["sst"]["transform"] == "log_grad"
    assert v["bathy"]["role"] == "static" and v["bathy"]["source"].startswith("TODO_")
    assert v["u_drifter"]["role"] == "target" and v["u_drifter"]["group"] == "currents"


def test_domain_splits_patches_and_training():
    data, xp, _ = convert(LEGACY, CATALOG)
    assert data["domain"] == {"lat": [-70, 70], "lon": [-180, 180]}
    assert data["eval_domain"] == {"lat": [-60, 60], "lon": [-170, 170]}
    assert data["splits"]["val"] == {"time": ["2018-01-01", "2018-12-31"]}
    assert data["patch"]["lat"] == 560 and data["stride"]["lon"] == 1432
    t = xp["training"]
    assert t["loss"] == "mae" and t["loss_combine"] == "group_mean" and t["grad_loss_weight"] == 20.0
    assert t["rec_weight"] == {"kind": "triangular", "crop": {"time": 0, "lat": 4, "lon": 4}, "kw": {"offset": 1}}
    assert t["trainer"]["max_epochs"] == 150 and t["batch_size"] == 2
    assert xp["model"]["widths"] == [64, 128, 256] and xp["model"]["dropout"] == 0.1


def test_everything_unmapped_is_reported():
    _, _, notes = convert(LEGACY, CATALOG)
    joined = "\n".join(notes)
    assert "mystery_key" in joined            # unknown per-variable key
    assert "unknown_block" in joined          # unknown top-level key
    assert "norm_stats" in joined             # behaviour change flagged
    assert "entrypoints" in joined
    assert "TODO_bathy_4th" in joined         # data source missing from the catalog
