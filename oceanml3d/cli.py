"""``oceanml3d`` command line: train / predict / export / list-models.

Everything is driven by one Hydra config tree (``config/``): an experiment picks a
``data`` group (variables + domain + splits), a ``model`` group, a ``training`` group and
a ``paths`` group (site-specific catalog). Example::

    oceanml3d train experiment=nosc_15m_duacs paths=odyssey
    oceanml3d predict experiment=nosc_15m_duacs ckpt=outputs/.../best.ckpt
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from oceanml3d.catalog import Catalog
from oceanml3d.config_schema import check_config, validate_config
from oceanml3d.variables import VariableSet

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


def build_variables(cfg: DictConfig) -> VariableSet:
    return VariableSet.from_config({k: v for k, v in OmegaConf.to_container(cfg.data.variables, resolve=True).items() if v is not None})


def build_catalog(cfg: DictConfig) -> Catalog:
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    return Catalog(paths.get("datasets", {}), root=paths.get("root"))


def build_datamodule(cfg: DictConfig, variables: VariableSet, catalog: Catalog):
    from oceanml3d.data.augment import build_augmentations
    from oceanml3d.data.datamodule import OceanDataModule

    resolve_auto_patch(cfg, catalog, variables)      # no-op once resolved; any caller gets numbers
    d = cfg.data
    return OceanDataModule(
        variables, catalog,
        domain=OmegaConf.to_container(d.domain, resolve=True),
        splits=OmegaConf.to_container(d.splits, resolve=True),
        patch=OmegaConf.to_container(d.patch, resolve=True),
        stride=OmegaConf.to_container(d.stride, resolve=True),
        batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
        norm_stats=OmegaConf.to_container(d.norm_stats, resolve=True) if d.get("norm_stats") else None,
        chunks=OmegaConf.to_container(d.chunks, resolve=True) if d.get("chunks") else None,
        cache=str(Path(OmegaConf.to_container(cfg.paths, resolve=True).get("root", ".")) / "cache" / f"{cfg.experiment_name}.zarr")
        if cfg.training.get("cache", False) else None,
        jitter=bool(cfg.training.get("moving_patches", False)),
        augmentations=build_augmentations(OmegaConf.to_container(d.augment, resolve=True) if d.get("augment") else None, variables),
        eval_domain=OmegaConf.to_container(d.eval_domain, resolve=True) if d.get("eval_domain") else None,
    )


def _is_auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == "auto"


def grid_sizes(cfg: DictConfig, catalog: Catalog, variables: VariableSet) -> dict[str, int]:
    """Number of (lat, lon) cells of the task's domain, read -- lazily -- from the first target."""
    from oceanml3d.data.open import open_variable

    domain = {k: slice(*v) for k, v in OmegaConf.to_container(cfg.data.domain, resolve=True).items()}
    da = open_variable(variables.targets[0], catalog, domain)
    return {d: int(da.sizes[d]) for d in ("lat", "lon")}


def resolve_auto_patch(cfg: DictConfig, catalog: Catalog, variables: VariableSet,
                       sizes: dict[str, int] | None = None) -> None:
    """Replace ``auto`` in ``data.patch`` / ``data.stride`` (lat, lon) by numbers fitted to the grid.

    The patch size is a number of *cells*, so it is tied to the resolution: 144 x 144 is the whole
    Gulf Stream box at 1/12 deg (145 cells) and does not fit at all at 0.25 deg (49). ``auto``
    derives it from the data actually on disk, whatever ``OCEANML3D_TARGET_RES`` produced it:

    * ``patch = largest multiple of data.patch_multiple (16) <= cells``: the whole domain, trimmed
      to what the U-Net's down-sampling divides. 145 -> 144, 49 -> 48.
    * ``stride = patch - 2 x rec_weight.crop``: the largest stride whose overlap still covers the
      cropped borders, so the export has no seams (the rule ``validate_config`` enforces).
      144 -> 136, 48 -> 40 with the default crop of 4.

    That reproduces the hand-set native values exactly. The resolved numbers are written back into
    the config, so the run directory records what was used.
    """
    d = cfg.data
    wanted = [dim for dim in ("lat", "lon") if _is_auto(d.patch.get(dim)) or _is_auto(d.stride.get(dim))]
    if not wanted:
        return
    sizes = sizes or grid_sizes(cfg, catalog, variables)
    mult = int(d.get("patch_multiple", 16))
    crop = OmegaConf.to_container(cfg.training.rec_weight.get("crop", {}), resolve=True)
    with open_dict(cfg):
        for dim in wanted:
            if _is_auto(d.patch.get(dim)):
                p = (sizes[dim] // mult) * mult
                if p < mult:
                    raise SystemExit(f"data.patch.{dim}=auto: the domain has {sizes[dim]} cells along "
                                     f"{dim}, fewer than data.patch_multiple={mult}. Enlarge the domain, "
                                     f"refine the resolution, or lower patch_multiple.")
                d.patch[dim] = p
            if _is_auto(d.stride.get(dim)):
                d.stride[dim] = max(1, int(d.patch[dim]) - 2 * int(crop.get(dim, 0)))
    print("[oceanml3d] patch/stride from the grid (" + ", ".join(
        f"{dim}: {sizes[dim]} cells -> patch {d.patch[dim]}, stride {d.stride[dim]}" for dim in wanted) + ")")


def check_export_rim(cfg: DictConfig, catalog: Catalog, variables: VariableSet) -> None:
    """Refuse a crop whose uncovered rim reaches into ``eval_domain``.

    ``training.rec_weight.crop`` is in *cells*, and no neighbouring patch covers the outer edge of
    the domain, so the exported product is NaN over a rim ``crop x grid step`` wide. At 1/12 deg the
    default 4 cells is 0.33 deg, well inside the 1 deg rim ``eval_domain`` excludes. At 0.5 deg it is
    2 deg: the product covered 34-42 N of a 32-44 N box, and a band the metrics score was empty.
    ``auto`` fitted the patch and the stride to the resolution; the crop it did not.
    """
    import numpy as np

    from oceanml3d.data.open import open_variable

    ev = cfg.data.get("eval_domain")
    crop = OmegaConf.to_container(cfg.training.rec_weight.get("crop", {}), resolve=True)
    if not ev or not any(int(crop.get(dim, 0)) for dim in ("lat", "lon")):
        return
    dom = OmegaConf.to_container(cfg.data.domain, resolve=True)
    da = open_variable(variables.targets[0], catalog, {k: slice(*v) for k, v in dom.items()})
    problems, rims = [], []
    for dim in ("lat", "lon"):
        cells = int(crop.get(dim, 0))
        if not cells or dim not in da.coords or da.sizes.get(dim, 0) < 2 or dim not in ev:
            continue
        step = float(np.median(np.abs(np.diff(np.asarray(da[dim].values, float)))))
        rim = cells * step
        margin = min(float(ev[dim][0]) - float(dom[dim][0]), float(dom[dim][1]) - float(ev[dim][1]))
        rims.append(f"{dim} {rim:.3g} deg ({cells} cells x {step:.3g})")
        if rim > margin + 0.5 * step:
            fit = max(0, int((margin + 0.5 * step) // step))
            problems.append(f"{dim}: the crop leaves a {rim:.3g} deg rim of the product empty ({cells} cells "
                            f"x {step:.3g} deg), wider than the {margin:.3g} deg that eval_domain excludes. "
                            f"Either training.rec_weight.crop.{dim}={fit} (the stride follows when it is "
                            f"`auto`), or an eval_domain at least {rim:.3g} deg inside the domain, or a finer "
                            f"resolution.")
    if rims:
        print("[oceanml3d] exported product empty over the outer rim: " + ", ".join(rims))
    if problems:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(problems))


def build_model(cfg: DictConfig, variables: VariableSet, norm_stats):
    import inspect

    from oceanml3d.registry import get_model
    from oceanml3d.training.weights import patch_weight

    m = OmegaConf.to_container(cfg.model, resolve=True)
    name = m.pop("name")
    cls = get_model(name)
    accepted = set(inspect.signature(cls.__init__).parameters) | set(inspect.signature(cls.__mro__[1].__init__).parameters)
    for k in [k for k in m if k not in accepted]:
        print(f"[oceanml3d] model '{name}' ignores config key '{k}' (inherited from the experiment body)")
        m.pop(k)
    if any(_is_auto(cfg.data.patch.get(k)) for k in ("lat", "lon")):
        raise ValueError("data.patch is still 'auto': build the datamodule first (it resolves it)")
    window = int(cfg.data.patch.time)
    weight = patch_weight(cfg.training.rec_weight.kind, OmegaConf.to_container(cfg.data.patch, resolve=True),
                          OmegaConf.to_container(cfg.training.rec_weight.crop, resolve=True),
                          **OmegaConf.to_container(cfg.training.rec_weight.get("kw", {}), resolve=True))
    opt = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    return cls(variables, window, weight, loss=cfg.training.loss, optimizer=opt.pop("name"),
                           optimizer_kw=opt, norm_stats=norm_stats,
                           loss_combine=cfg.training.get("loss_combine", "flat_sum"),
                           loss_group_weights=OmegaConf.to_container(cfg.training.get("loss_group_weights", {}), resolve=True),
                           grad_loss_weight=cfg.training.get("grad_loss_weight", 0.0), **m)


def build_trainer(cfg: DictConfig, stage: str | None = None):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    from oceanml3d.training.callbacks import VersioningCallback

    t = cfg.training.trainer
    if stage and cfg.training.get("stage_trainer", {}).get(stage):
        t = OmegaConf.merge(t, cfg.training.stage_trainer[stage])
    out = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out = f"{out}/{stage}" if stage else out
    # Selected on `val/nrmse` by default, not `val/loss`: the loss changes definition with every
    # ablation, the RMSE does not (see BaseOceanModel._eval_step). `training.monitor` overrides.
    monitor = cfg.training.get("monitor", "val/nrmse")
    callbacks = [
        VersioningCallback(cfg), LearningRateMonitor(),
        ModelCheckpoint(monitor=monitor, mode="min", save_top_k=cfg.training.save_top_k, save_last=True,
                        filename="{epoch:03d}-{" + monitor + ":.5f}", auto_insert_metric_name=False),
    ]
    es = cfg.training.get("early_stopping")
    if es:
        callbacks.append(EarlyStopping(monitor=monitor, mode="min", **OmegaConf.to_container(es, resolve=True)))
    return Trainer(callbacks=callbacks, logger=_loggers(out, TensorBoardLogger, CSVLogger),
                   default_root_dir=out, **OmegaConf.to_container(t, resolve=True))


def _loggers(out: str, tensorboard_cls, csv_cls) -> list:
    """CSV always; TensorBoard when it is installed.

    TensorBoard is a viewer, not a dependency of training, and an image built without it used to
    fail a queued job at `build_trainer` -- after the data had been opened and the model built --
    with `Neither tensorboard nor tensorboardX is available`. The CSV log holds every scalar
    TensorBoard would have shown (`metrics.csv` next to the checkpoints).
    """
    loggers = [csv_cls(out, name="", version="")]
    try:
        loggers.append(tensorboard_cls(out, name="", version=""))
    except ModuleNotFoundError as exc:
        print(f"[oceanml3d] TensorBoard logging off ({exc.__class__.__name__}: neither tensorboard nor "
              f"tensorboardX is installed); metrics go to {out}/metrics.csv only")
    return loggers


@hydra.main(config_path=CONFIG_DIR, config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cmd = cfg.command
    if cmd == "list-models":
        from oceanml3d.registry import list_models
        print("\n".join(list_models()))
        return
    if cmd == "show-config":
        print(OmegaConf.to_yaml(cfg, resolve=True))
        problems = validate_config(cfg, build_catalog(cfg), build_variables(cfg))
        print("\n# validation: " + ("OK" if not problems else "\n# - ".join([""] + problems)).strip())
        return
    if cmd == "validate":
        catalog, variables = build_catalog(cfg), build_variables(cfg)
        resolve_auto_patch(cfg, catalog, variables)
        check_config(cfg, catalog, variables)
        check_export_rim(cfg, catalog, variables)
        print(f"configuration '{cfg.experiment_name}' is valid")
        return
    if cmd == "prepare-obs":
        prepare_observations(cfg, build_catalog(cfg))
        return
    if cmd == "eofs":
        compute_eofs(cfg, build_catalog(cfg))
        return

    import pytorch_lightning as L
    L.seed_everything(cfg.seed, workers=True)
    variables = build_variables(cfg)
    catalog = build_catalog(cfg)
    check_config(cfg, variables=variables)          # data files may not exist yet: checked after prepare
    if cfg.data.get("prepare"):
        prepare_observations(cfg, catalog)
    resolve_auto_patch(cfg, catalog, variables)
    check_config(cfg, catalog, variables)
    check_export_rim(cfg, catalog, variables)
    dm = build_datamodule(cfg, variables, catalog)
    dm.setup("fit")
    model = build_model(cfg, variables, dm.norm_stats())
    trainer = build_trainer(cfg)

    if cmd == "train":
        stages = list(cfg.training.get("stages") or getattr(model, "stages", ("main",)))
        for i, stage in enumerate(stages):
            if len(stages) > 1:
                print(f"[oceanml3d] stage {i + 1}/{len(stages)}: {stage}")
                model.set_stage(stage)
                trainer = build_trainer(cfg, stage=stage)
            trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("ckpt") if i == 0 else None)
        # "best" refers to the checkpoint callback of the trainer that ran last, which for a
        # multi-stage pipeline is the last stage -- exactly what should be tested. The old
        # `if len(stages) == 1` tested the in-memory weights of the final epoch instead, which is
        # not what the single-stage branch did, and nothing said so.
        best = getattr(trainer.checkpoint_callback, "best_model_path", "")
        trainer.test(model, datamodule=dm, ckpt_path="best" if best else None)
        if cfg.export.enabled:
            _export(cfg, model, dm, trainer)
    elif cmd == "predict":
        if not cfg.get("ckpt"):
            sys.exit("predict requires ckpt=<path>")
        import torch
        ckpt = torch.load(cfg.ckpt, map_location="cpu", weights_only=True)
        model.on_load_checkpoint(ckpt)              # channel layout + normalisation statistics
        model.load_state_dict(ckpt["state_dict"])
        if not ckpt.get("oceanml3d", {}).get("norm_stats"):
            print(f"[oceanml3d] warning: {cfg.ckpt} carries no normalisation statistics (written "
                  "before they were stored in checkpoints). Falling back to statistics recomputed "
                  "from this configuration; if its splits, domain or variables differ from the "
                  "training run, the exported product is denormalised with the wrong numbers.")
        _export(cfg, model, dm, trainer)
    else:
        sys.exit(f"unknown command '{cmd}' (train|predict|list-models|show-config)")


def prepare_observations(cfg: DictConfig, catalog: Catalog) -> None:
    """Observation-system simulation declared in ``data.prepare``.

    Idempotent (existing outputs are skipped) and **rank-safe**. This runs from ``main()``, before a
    Trainer exists, so Lightning's ``prepare_data`` hook -- which would be the idiomatic place -- is
    not available: ``dm.setup("fit")`` needs the files before ``trainer.fit`` is ever called. Under
    DDP the script is relaunched once per rank, so all of them reach this point at the same time.

    One process simulates and the others wait for the result. Skipping on existence is not enough on
    its own: the existence test passes the moment the first writer creates the file, well before it
    has finished filling it. The writes are therefore atomic (see :mod:`oceanml3d.io`), so a waiting
    rank sees either nothing or a complete file.
    """
    from oceanml3d.io import global_rank_env, is_global_zero, wait_for
    from oceanml3d.obs.argo_virtual import build_virtual_argo
    from oceanml3d.obs.pseudo_obs import prepare_pseudo_obs

    prep = OmegaConf.to_container(cfg.data.prepare, resolve=True)
    po = prep.get("pseudo_obs") or {}
    entries = [po] if "truth" in po else [v for v in po.values() if v]
    va = prep.get("virtual_argo")

    declared = [catalog.resolve(e["output"]) for e in entries]
    if va:
        declared.append(catalog.resolve(va["output"]))
    if not is_global_zero():
        print(f"[oceanml3d] rank {global_rank_env()}: waiting for the observation simulation "
              f"produced by rank 0 ({len(declared)} file(s))")
        wait_for(declared, what="simulated observations")
        return

    for e in entries:
        output, truth = catalog.resolve(e["output"]), catalog.resolve(e["truth"])
        stale = stale_derived_file(output, truth)
        if stale:
            print(f"[oceanml3d] rebuilding {output}: {stale}")
        out = prepare_pseudo_obs(truth, e["truth_var"], output,
                                 missions=e.get("missions"),
                                 real_mask=catalog.resolve(e["real_mask"]) if e.get("real_mask") else None,
                                 noise_std=e.get("noise_std", 0.0), seed=e.get("seed", 1234),
                                 historical=e.get("historical", False), per_mission=e.get("per_mission", False),
                                 depth_index=e.get("depth_index"), clouds=e.get("clouds"),
                                 skip_if_exists=not stale)
        print(f"pseudo-obs: {out}")
    if va:
        import pandas as pd
        output, truth = catalog.resolve(va["output"]), catalog.resolve(va["truth"])
        profiles_path = catalog.resolve(va["profiles"])
        if not profiles_path.exists():
            if not output.exists():
                print(f"virtual ARGO skipped: profile table {profiles_path} not found (see scripts/prepare/argo_profiles.py)")
                return
            # Nothing to rebuild from, so the file is used as it is -- but not unseen.
            stale = stale_derived_file(output, truth)
            print(f"virtual ARGO: no profile table ({profiles_path}); using {output} as it is"
                  + (f" -- WARNING {stale}" if stale else ""))
            return
        # Said either way. The run that trained on an empty ARGO input reused a file built against
        # something else and printed nothing about it: neither "virtual ARGO:" nor "skipped".
        stale = stale_derived_file(output, truth, newer_than=profiles_path) if output.exists() else "absent"
        if not stale:
            print(f"virtual ARGO: reusing {output} (same time axis and grid as {truth.name}, newer than "
                  f"{profiles_path.name})")
            return
        if stale != "absent":
            print(f"[oceanml3d] rebuilding {output}: {stale}")
        profiles = pd.read_csv(profiles_path) if profiles_path.suffix == ".csv" else pd.read_parquet(profiles_path)
        out = build_virtual_argo(profiles, truth, va["truth_var"], list(va["depth_indices"]),
                                 output, noise_std=va.get("noise_std", 0.0), seed=va.get("seed", 0))
        print(f"virtual ARGO: {out}")


def stale_derived_file(output: Path, truth: Path, newer_than: Path | None = None) -> str | None:
    """Why an existing ``prepare-obs`` output must be rebuilt, or ``None`` if it can be reused.

    These files are skipped when they exist, which is what makes the step idempotent -- and what let
    a virtual ARGO file with no date in common with the truth be reused silently, so a whole training
    run saw an ARGO input that was zero everywhere. Existence is not validity: the file has to be on
    the truth's time axis and grid (a truth re-prepared at another resolution or over another period
    leaves every derived file behind), and newer than the table it was simulated from.
    """
    import pandas as pd
    import xarray as xr

    from oceanml3d.data.open import normalise_dims

    output, truth = Path(output), Path(truth)
    if not output.exists():
        return None
    if newer_than is not None and Path(newer_than).exists() and \
            Path(newer_than).stat().st_mtime > output.stat().st_mtime:
        return f"{Path(newer_than).name} is newer than it"
    try:
        with xr.open_dataset(output) as d:
            got = normalise_dims(d)
            got_t = pd.DatetimeIndex(got.time.values).normalize()
            got_grid = (got.sizes.get("lat"), got.sizes.get("lon"))
        ref = normalise_dims(xr.open_zarr(truth) if truth.suffix == ".zarr" else xr.open_dataset(truth))
        ref_t = pd.DatetimeIndex(ref.time.values).normalize()
        ref_grid = (ref.sizes.get("lat"), ref.sizes.get("lon"))
    except Exception as exc:  # noqa: BLE001 -- unreadable is a reason to rebuild, not to crash
        return f"it cannot be checked ({type(exc).__name__}: {exc})"
    if got_grid != ref_grid:
        return f"its grid is {got_grid[0]}x{got_grid[1]}, the truth's is {ref_grid[0]}x{ref_grid[1]}"
    if not got_t.equals(ref_t):
        def span(t):
            return f"{t[0].date()}..{t[-1].date()} ({len(t)} days)" if len(t) else "empty"
        return f"its time axis is {span(got_t)}, the truth's is {span(ref_t)}"
    return None


def compute_eofs(cfg: DictConfig, catalog: Catalog) -> None:
    """Vertical EOF bases for every multi-level target group (needed by ablation=vertical_modes)."""
    from oceanml3d.training.vertical_modes import compute_vertical_eofs

    variables = build_variables(cfg)
    n_modes = int(cfg.model.get("mode_specs", {}) and next(iter(cfg.model.mode_specs.values())).n_modes or 8)
    train = cfg.data.splits.train.time
    root = Path(OmegaConf.to_container(cfg.paths, resolve=True).get("root", "."))
    for group, names in variables.target_groups.items():
        specs = [variables[n] for n in names]
        if len(specs) < 2 or specs[0].depth_index is None:
            continue
        s0 = specs[0]
        out = root / "eofs" / f"{s0.var_name}_{cfg.data.get('tag', 'gs21')}.npz"
        compute_vertical_eofs(catalog.resolve(s0.source), s0.var_name, [s.depth_index for s in specs], n_modes, out,
                              time_slice=slice(train[0], train[1]),
                              domain={k: slice(*v) for k, v in OmegaConf.to_container(cfg.data.domain, resolve=True).items()})
        print(f"EOFs for group '{group}': {out}")


def _export(cfg: DictConfig, model, dm, trainer) -> None:
    """Write the product. Rank 0 only.

    Under DDP every rank reaches this point. Prediction itself is never sharded (see
    :func:`oceanml3d.inference.predict.predict_field`), but letting four ranks run the pass and
    write the same NetCDF files concurrently is pointless at best. The barrier keeps the others
    from exiting while rank 0 is still writing.
    """
    from oceanml3d.inference.predict import predict_and_export

    if not getattr(trainer, "is_global_zero", True):
        trainer.strategy.barrier("oceanml3d-export")
        return

    e = cfg.export
    out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "product"
    time_slice = slice(*e.time) if e.get("time") else None
    attrs = {"experiment": cfg.experiment_name, "model": cfg.model.name, "git_hash": _git_hash()}
    path = predict_and_export(model, dm, trainer, out, cfg.experiment_name, e.split, time_slice, e.get("depth_m"), attrs)
    print(f"product manifest written to {path}")
    if getattr(trainer, "world_size", 1) > 1:
        trainer.strategy.barrier("oceanml3d-export")


def _git_hash() -> str:
    from oceanml3d.training.callbacks import git_hash
    return git_hash()


if __name__ == "__main__":
    main()
