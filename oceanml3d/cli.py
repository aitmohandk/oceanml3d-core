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
from omegaconf import DictConfig, OmegaConf

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
    )


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
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    from oceanml3d.training.callbacks import VersioningCallback

    t = cfg.training.trainer
    if stage and cfg.training.get("stage_trainer", {}).get(stage):
        t = OmegaConf.merge(t, cfg.training.stage_trainer[stage])
    out = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out = f"{out}/{stage}" if stage else out
    callbacks = [
        VersioningCallback(cfg), LearningRateMonitor(),
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=cfg.training.save_top_k,
                        filename="{epoch:03d}-{val/loss:.5f}", auto_insert_metric_name=False),
    ]
    loggers = [CSVLogger(out, name="", version=""), TensorBoardLogger(out, name="", version="")]
    return Trainer(callbacks=callbacks, logger=loggers, default_root_dir=out, **OmegaConf.to_container(t, resolve=True))


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
        check_config(cfg, build_catalog(cfg), build_variables(cfg))
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
    check_config(cfg, catalog, variables)
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
        out = prepare_pseudo_obs(catalog.resolve(e["truth"]), e["truth_var"], catalog.resolve(e["output"]),
                                 missions=e.get("missions"),
                                 real_mask=catalog.resolve(e["real_mask"]) if e.get("real_mask") else None,
                                 noise_std=e.get("noise_std", 0.0), seed=e.get("seed", 1234),
                                 historical=e.get("historical", False), per_mission=e.get("per_mission", False),
                                 depth_index=e.get("depth_index"), clouds=e.get("clouds"))
        print(f"pseudo-obs: {out}")
    if va and not catalog.resolve(va["output"]).exists():
        import pandas as pd
        profiles_path = catalog.resolve(va["profiles"])
        if not profiles_path.exists():
            print(f"virtual ARGO skipped: profile table {profiles_path} not found (see scripts/prepare/argo_profiles.py)")
            return
        profiles = pd.read_csv(profiles_path) if profiles_path.suffix == ".csv" else pd.read_parquet(profiles_path)
        out = build_virtual_argo(profiles, catalog.resolve(va["truth"]), va["truth_var"], list(va["depth_indices"]),
                                 catalog.resolve(va["output"]), noise_std=va.get("noise_std", 0.0), seed=va.get("seed", 0))
        print(f"virtual ARGO: {out}")


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
