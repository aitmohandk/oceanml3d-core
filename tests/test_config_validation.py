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
