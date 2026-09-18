import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from oceanml3d.cli import CONFIG_DIR, build_variables
from oceanml3d.config_schema import validate_config


def _cfg(*overrides):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="main", overrides=["experiment=osse3d_gs21_multivar_unet", *overrides])


def test_shipped_experiments_are_valid():
    for xp in ["nosc_15m_duacs", "nosc_15m_neurost_sst", "nosc_00m_duacs", "osse3d_gs21_multivar_unet",
               "osse3d_gs21_surface_only", "fourdvarnet_ssh_osse", "smoke", "osse3d_smoke"]:
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            cfg = compose(config_name="main", overrides=[f"experiment={xp}"])
        assert validate_config(cfg, variables=build_variables(cfg)) == [], xp


@pytest.mark.parametrize("override,expected", [
    ("data.stride.lat=99999", "gaps in coverage"),
    ("training.loss=huber", "training.loss='huber'"),
    ("training.loss_combine=magic", "training.loss_combine='magic'"),
    ("training.rec_weight.kind=cosine", "rec_weight.kind"),
    ("training.rec_weight.crop.lat=1000", "removes the whole patch"),
    ("export.split=validation", "export.split"),
    ("export.time=[2005-01-01,2005-12-31]", "outside the 'test' split"),
    ("data.splits.val.time=[2011-01-01,2011-12-31]", "leakage"),
    ("data.splits.test.time=[2020-01-01,2019-01-01]", "is not before stop"),
])
def test_catches_common_mistakes(override, expected):
    cfg = _cfg(override)
    assert any(expected in p for p in validate_config(cfg, variables=build_variables(cfg))), override


def test_reports_missing_data_files(tmp_path):
    from oceanml3d.catalog import Catalog

    cfg = _cfg()
    variables = build_variables(cfg)
    problems = validate_config(cfg, Catalog({}, root=tmp_path), variables)
    assert any("not in the catalog" in p for p in problems)


def test_norm_stats_length_mismatch():
    cfg = _cfg()
    with open_dict(cfg):
        cfg.data.norm_stats = [[0.0, 1.0], [1.0, 1.0]]
    assert any("norm_stats has 2 entries" in p for p in validate_config(cfg, variables=build_variables(cfg)))
    assert OmegaConf.is_config(cfg)


def test_multi_stage_pipeline_is_declarative():
    """A model declares its stages; the config can override them and set per-stage trainer options."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    import numpy as np

    from oceanml3d.models.ocean.baselines.model import LinearBaseline
    from oceanml3d.variables import VariableSet

    vs = VariableSet.from_config({"a": {"source": "s", "role": "input"}, "b": {"source": "s", "role": "target"}})
    model = LinearBaseline(vs, 3, np.ones((3, 4, 4), np.float32))
    assert model.stages == ("main",)
    model.set_stage("main")
    with pytest.raises(ValueError, match="declares stages"):
        model.set_stage("prior")

    class TwoStage(LinearBaseline):
        stages = ("prior", "conditional")

    m2 = TwoStage(vs, 3, np.ones((3, 4, 4), np.float32))
    m2.set_stage("conditional")
    assert m2.current_stage == "conditional"


def test_overlap_smaller_than_twice_the_crop_is_rejected():
    """The cropped border of a patch contributes nothing, so the neighbour has to cover it. Both
    shipped tasks sit exactly on the equality (144 - 136 = 8 = 2 x 4), so any edit to patch, stride
    or crop broke the export -- with blank seams, and nothing downstream complaining."""
    cfg = _cfg("data.stride.lat=142")           # overlap 2, crop 4 -> needs 8
    problems = validate_config(cfg, variables=build_variables(cfg))
    assert any("uncovered seams" in x for x in problems), problems

    cfg = _cfg("data.stride.lat=136")           # overlap 8 == 2 x 4: the shipped setting
    assert validate_config(cfg, variables=build_variables(cfg)) == []


GRIDDED_EXPERIMENTS = ["nosc_15m_duacs", "nosc_15m_neurost_sst", "nosc_00m_duacs",
                       "osse3d_gs21_multivar_unet", "osse3d_gs21_surface_only",
                       "fourdvarnet_ssh_osse", "lorenz96_unet", "lorenz96_enkf",
                       "smoke", "osse3d_smoke"]


def _catalog_from(site: str):
    """The site's catalog with every path pointed at a directory that does not exist, so only the
    *key* lookups matter and the test does not need the data."""
    import yaml

    from oceanml3d.catalog import Catalog

    doc = yaml.safe_load(open(f"{CONFIG_DIR}/paths/{site}.yaml"))
    return Catalog(doc.get("datasets", {}), root="/nonexistent-on-purpose")


def test_every_shipped_experiment_resolves_against_the_local_catalog():
    """`config/data/osse3d_gs21.yaml` referenced seven keys that no shipped site file defined, so
    that experiment could not run anywhere as delivered -- and nothing said so, because the
    validator only resolved `spec.source` and ignored masks and the `prepare` block entirely."""
    catalog = _catalog_from("local")
    for xp in GRIDDED_EXPERIMENTS:
        with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
            cfg = compose(config_name="main", overrides=[f"experiment={xp}"])
        unknown = [x for x in validate_config(cfg, catalog, build_variables(cfg))
                   if "not in the catalog" in x]
        assert unknown == [], f"{xp}: {unknown}"


def test_a_site_file_does_not_invent_keys_of_its_own():
    """Site files differ in paths, not in vocabulary: a key only one of them knows is a typo, or a
    key the others silently lack."""
    import glob
    import os

    reference = set(_catalog_from("local").entries)
    for path in sorted(glob.glob(f"{CONFIG_DIR}/paths/*.yaml")):
        site = os.path.splitext(os.path.basename(path))[0]
        extra = set(_catalog_from(site).entries) - reference
        assert extra == set(), f"{site}.yaml defines keys local.yaml does not: {sorted(extra)}"


def test_an_unknown_mask_key_is_reported():
    catalog = _catalog_from("local")
    # `+` because Hydra composes in struct mode: `mask` is not declared on this variable
    cfg = _cfg("+data.variables.ssh_obs.mask=not_a_key")
    problems = validate_config(cfg, catalog, build_variables(cfg))
    assert any("mask 'not_a_key' is not in the catalog" in x for x in problems), problems


def test_a_prepare_output_need_not_exist_but_must_be_declared():
    """`prepare-obs` writes to catalog keys, so they must be declared before it runs -- but
    requiring the file to exist would make validation impossible before the first run."""
    catalog = _catalog_from("local")
    cfg = _cfg()
    problems = validate_config(cfg, catalog, build_variables(cfg))
    # Only the *outputs* are exempt. `truth` and `profiles` are inputs to the simulation and are
    # expected to be missing here, since this catalog deliberately points at nothing -- as is
    # `variable 'ssh_obs': source`, which reads one of those outputs.
    assert not any(".output" in x and "does not exist" in x for x in problems), problems
    assert any("data.prepare.pseudo_obs.truth" in x and "does not exist" in x for x in problems), \
        "inputs to the simulation must still be checked for existence"

    cfg = _cfg("data.prepare.pseudo_obs.ssh.output=not_a_key")
    problems = validate_config(cfg, catalog, build_variables(cfg))
    assert any("pseudo_obs.output 'not_a_key' is not in the catalog" in x for x in problems), problems


def test_every_site_env_file_has_a_matching_paths_file_or_is_a_template():
    """A site is two files: jobs/env/<site>.sh (environment) and config/paths/<site>.yaml (data).
    A site script pointing at a catalog that does not exist fails only once a job is queued."""
    import glob
    import os
    import re

    for path in sorted(glob.glob("jobs/env/*.sh")):
        site = os.path.splitext(os.path.basename(path))[0]
        text = open(path).read()
        m = re.search(r'OCEANML3D_PATHS="\$\{OCEANML3D_PATHS:-(\w+)\}"', text)
        assert m, f"{path} must set OCEANML3D_PATHS"
        declared = m.group(1)
        assert declared == site, f"{path} points at paths={declared}, expected {site}"
        # The paths file may legitimately not exist yet for a site nobody has configured; what must
        # not happen is a site script silently pointing at another site's catalog.
