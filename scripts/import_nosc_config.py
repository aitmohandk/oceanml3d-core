"""Translate a legacy NOSC ``config/xp/*.yaml`` into an oceanml3d data+experiment pair.

Migration is where science silently changes: this tool does the mechanical part (variables and
their roles, domain, splits, patch/stride, loss, optimiser, model widths) and — more importantly —
*reports every key it did not understand*, so nothing is dropped without you seeing it.

    python scripts/import_nosc_config.py --xp <NOSC>/config/xp/unet_uv_....yaml \\
        --out-data config/data/imported.yaml --out-experiment config/experiment/imported.yaml \\
        [--catalog config/paths/odyssey.yaml]

With ``--catalog`` the absolute ``var_path`` values are matched against the catalog and replaced
by catalog keys; unmatched paths are emitted as ``TODO_<n>`` keys listed in the report.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

ROLE = {("prior_input", "no_output"): "input", ("no_input", "full_output"): "target",
        ("prior_input", "full_output"): "both", ("no_input", "no_output"): "aux"}
TRANSFORM = {"sst_transfo": "log_grad", "log_grad": "log_grad"}
KNOWN_TOP = {"multivar", "multivar_selector", "domain", "datamodule", "trainer", "model", "entrypoints",
             "paths", "defaults", "_target_", "hydra", "params", "xp"}


def _slice_args(node: Any) -> list | None:
    if isinstance(node, dict) and node.get("_target_") == "builtins.slice":
        return list(node.get("_args_", []))
    return node if isinstance(node, list) else None


def _deref(value: Any) -> Any:
    """Resolve trivial ``${paths.x}`` interpolations to the literal text (kept for the report)."""
    return value


def convert(xp: dict, catalog: dict | None = None) -> tuple[dict, dict, list[str]]:
    notes: list[str] = []
    paths = xp.get("paths", {})
    by_path: dict[str, str] = {}
    if catalog:
        for key, entry in (catalog.get("datasets") or {}).items():
            p = entry["path"] if isinstance(entry, dict) else entry
            by_path[Path(p).name] = key

    def source_key(var_path: str) -> str:
        raw = str(var_path)
        m = re.fullmatch(r"\$\{paths\.([^}]+)\}", raw)
        if m and m.group(1) in paths:
            raw = str(paths[m.group(1)])
        name = Path(raw).name
        if name in by_path:
            return by_path[name]
        key = "TODO_" + re.sub(r"\W+", "_", Path(raw).stem)[:40]
        notes.append(f"data source not in the catalog: '{raw}' -> catalog key '{key}' (add it to config/paths/<site>.yaml)")
        return key

    variables: dict[str, dict] = {}
    for name, info in (xp.get("multivar") or {}).items():
        role = ROLE.get((info.get("input_arch", "no_input"), info.get("output_arch", "no_output")))
        if role is None:
            notes.append(f"variable '{name}': unknown arch pair {info.get('input_arch')}/{info.get('output_arch')}, skipped")
            continue
        if info.get("broadcast_time"):
            role = "static"
        spec: dict[str, Any] = {"source": source_key(info.get("var_path", "")), "role": role}
        if info.get("var_name") and info["var_name"] != name:
            spec["var_name"] = info["var_name"]
        if info.get("depth_index") is not None:
            spec["depth_index"] = int(info["depth_index"])
        if info.get("depth_level") is not None:
            notes.append(f"variable '{name}': depth_level={info['depth_level']} was a *nearest* lookup; "
                         "converted to nothing - set an exact depth_index instead")
        if info.get("head_group"):
            spec["group"] = info["head_group"]
        for k in ("transfo", "transform"):
            if info.get(k):
                spec["transform"] = TRANSFORM.get(info[k], info[k])
        if info.get("sst_transfo"):        # NOSC flag: log|grad(SST)| applied at load time
            spec["transform"] = "log_grad"
        for k in set(info) - {"var_path", "var_name", "input_arch", "output_arch", "broadcast_time",
                              "depth_index", "depth_level", "head_group", "transfo", "transform", "sst_transfo"}:
            notes.append(f"variable '{name}': unmapped key '{k}': {info[k]!r}")
        variables[name] = spec

    dm = xp.get("datamodule", {})
    xk = dm.get("xrds_kw", {})
    dom = (xp.get("domain") or {}).get("train", xp.get("domain", {}))
    data = {
        "variables": variables,
        "domain": {k: _slice_args(v) for k, v in dom.items() if _slice_args(v)},
        "splits": {k: {"time": _slice_args(v.get("time"))} for k, v in (dm.get("domains") or {}).items()},
        "patch": dict(xk.get("patch_dims", {})),
        "stride": dict(xk.get("strides", {})),
        "norm_stats": None,
    }
    if (xp.get("domain") or {}).get("test"):
        data["eval_domain"] = {k: _slice_args(v) for k, v in xp["domain"]["test"].items() if _slice_args(v)}
    if dm.get("norm_stats"):
        notes.append("datamodule.norm_stats was hard-coded; oceanml3d recomputes it on the train split "
                     "(set data.norm_stats explicitly to reproduce the old run bit-for-bit)")

    model = xp.get("model", {})
    solver = model.get("solver", {})
    training: dict[str, Any] = {"batch_size": (dm.get("dl_kw") or {}).get("batch_size", 1),
                                "num_workers": (dm.get("dl_kw") or {}).get("num_workers", 4)}
    target = str(model.get("_target_", ""))
    training["loss"] = "mae" if "mae" in target else "mse"
    if model.get("loss_group_mode"):
        training["loss_combine"] = model["loss_group_mode"]
    if model.get("grad_loss_weight"):
        training["grad_loss_weight"] = model["grad_loss_weight"]
    opt = model.get("opt_fn", {})
    if opt:
        training["optimizer"] = {"name": "cosine_adam", "lr": opt.get("lr", 1e-4),
                                 "t_max": "${training.trainer.max_epochs}"}
    rw = model.get("rec_weight", {})
    if rw:
        training["rec_weight"] = {"kind": "triangular" if "mapping_wei" in str(rw.get("_target_")) else "constant",
                                  "crop": dict(rw.get("crop", {})), "kw": {"offset": rw.get("offset", 1)}}
    trainer = {k: v for k, v in (xp.get("trainer") or {}).items()
               if k in ("max_epochs", "gradient_clip_val", "limit_train_batches", "accelerator", "devices", "precision")}
    if trainer:
        training["trainer"] = trainer

    mult = solver.get("model_channels")
    widths = [int(mult) * int(m) for m in solver.get("channel_mult", [])] if mult else None
    model_cfg: dict[str, Any] = {"name": "nosc_unet"}
    if widths:
        model_cfg["widths"] = widths
    if solver.get("dropout") is not None:
        model_cfg["dropout"] = solver["dropout"]
    if solver.get("attention_resolutions"):
        model_cfg["attention_levels"] = [int(r) for r in solver["attention_resolutions"]]
        notes.append("attention_resolutions -> model.attention_levels: NOSC counts downsampling factors, "
                     "oceanml3d counts encoder levels; check the value")
    if "Headed" in str(solver.get("_target_", "")):
        model_cfg["head"] = "grouped"
    if "VerticalModes" in str(solver.get("_target_", "")):
        model_cfg["head"] = "vertical_modes"
    if "Uncertainty" in target:
        training["loss_combine"] = "uncertainty"

    for key in set(xp) - KNOWN_TOP:
        notes.append(f"top-level key '{key}' not translated: {str(xp[key])[:80]}")
    if xp.get("entrypoints"):
        notes.append("entrypoints: mask/pseudo-obs generation is now declared in data.prepare "
                     "(`oceanml3d command=prepare-obs`); training is the command itself")

    experiment = {"defaults": [{"override /data": "<generated data file>"},
                               {"override /model": model_cfg["name"]}],
                  "model": {k: v for k, v in model_cfg.items() if k != "name"}, "training": training,
                  "export": {"enabled": True, "split": "test", "time": None}}
    return data, experiment, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xp", required=True)
    p.add_argument("--out-data", required=True)
    p.add_argument("--out-experiment", required=True)
    p.add_argument("--catalog")
    a = p.parse_args()
    xp = yaml.safe_load(Path(a.xp).read_text())
    catalog = yaml.safe_load(Path(a.catalog).read_text()) if a.catalog else None
    data, experiment, notes = convert(xp, catalog)
    name = Path(a.out_data).stem
    experiment["defaults"][0] = {"override /data": name}
    Path(a.out_data).write_text("# @package data\n# Imported from " + a.xp + " by scripts/import_nosc_config.py\n"
                                + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    body = yaml.safe_dump(experiment, sort_keys=False, allow_unicode=True, default_flow_style=False)
    body = body.replace("\n- override", "\n  - override")
    Path(a.out_experiment).write_text("# @package _global_\n# Imported from " + a.xp + "\n" + body)
    print(f"wrote {a.out_data} and {a.out_experiment}")
    print("\nREVIEW THESE POINTS:" if notes else "\nnothing left unmapped.")
    for n in notes:
        print(f"  - {n}")
    print("\nThen: oceanml3d command=validate experiment=" + Path(a.out_experiment).stem)


if __name__ == "__main__":
    main()
