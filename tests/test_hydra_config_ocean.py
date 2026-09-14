from hydra import compose, initialize_config_dir

from oceanml3d.cli import CONFIG_DIR
from oceanml3d.variables import VariableSet


def test_compose_experiments():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        for xp in ["nosc_15m_duacs", "nosc_15m_neurost_sst", "nosc_00m_duacs", "smoke"]:
            cfg = compose(config_name="main", overrides=[f"experiment={xp}"], return_hydra_config=True)
            vs = VariableSet.from_config(cfg.data.variables)
            assert vs.targets and cfg.model.name == "nosc_unet"
            assert cfg.training.optimizer.t_max == cfg.training.trainer.max_epochs
