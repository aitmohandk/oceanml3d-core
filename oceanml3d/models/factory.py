"""Build a model from a Hydra config: the single ``model_type`` dispatch.

Lives in the package rather than in ``train.py`` because ``evaluation.neural_inference`` needs it
to rebuild a model exactly as training did. While it sat in the root script, importing it from the
package meant ``from train import model_factory`` — which only resolves when the repository root
happens to be on ``sys.path``, i.e. never in an installed distribution.

``train.py`` re-exports it, so ``from train import model_factory`` keeps working.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from oceanml3d.models.direct_unet import DirectUNet
from oceanml3d.models.solver import TweedieSolver
from oceanml3d.models.vanilla_cfm import VanillaCFM


def model_factory(cfg: DictConfig, device: torch.device):
    model_type = cfg.model.get("model_type", "tweedie")
    if model_type == "tweedie":
        model = TweedieSolver(
            state_dim=cfg.model.state_dim,
            hidden_channels=cfg.model.hidden_channels,
            time_emb_dim=cfg.model.time_emb_dim,
            use_obs=cfg.model.use_obs,
            use_energy=cfg.model.use_energy,
            nu=cfg.model.nu,
            K_inner=cfg.model.K_inner,
            N_outer=cfg.model.N_outer,
            dropout=cfg.model.dropout,
        )
    elif model_type == "direct_unet":
        dc = cfg.model.direct_unet
        param_dim = cfg.model.get("param_dim", 4)
        model = DirectUNet(
            state_dim=cfg.model.state_dim,
            hidden_channels=dc.hidden_channels,
            dropout=dc.dropout,
            param_dim=param_dim,
            cond_extra_dim=dc.get("cond_extra_dim", 1 + param_dim),
        )
    elif model_type == "monai_direct_unet":
        from oceanml3d.models.monai_unet_adapter import MonaiDirectUNet
        mdu = cfg.model.monai_direct_unet
        param_dim = cfg.model.get("param_dim", 4)
        model = MonaiDirectUNet(
            state_dim=cfg.model.state_dim,
            hidden_channels=mdu.hidden_channels,
            dropout=mdu.get("dropout", 0.1),
            param_dim=param_dim,
            cond_extra_dim=mdu.get("cond_extra_dim", 1 + param_dim),
            num_res_blocks=mdu.get("num_res_blocks", 2),
            norm_num_groups=mdu.get("norm_num_groups", 32),
        )
    elif model_type == "vanilla_cfm":
        vc = cfg.model.vanilla_cfm
        param_dim = cfg.model.get("param_dim", 4)
        model = VanillaCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=vc.hidden_channels,
            time_emb_dim=vc.time_emb_dim,
            N_outer=vc.N_outer,
            sigma_prior=vc.sigma_prior,
            dropout=vc.dropout,
            train_tau_0_only=vc.get("train_tau_0_only", False),
            param_dim=param_dim,
            cond_extra_dim=vc.get("cond_extra_dim", 1 + param_dim),
        )
    elif model_type == "joint_cfm":
        from oceanml3d.models.vanilla_cfm import JointCFM
        jc = cfg.model.joint_cfm
        vc = cfg.model.vanilla_cfm
        model = JointCFM(
            state_dim=cfg.model.state_dim,
            param_dim=jc.param_dim,
            hidden_channels=vc.hidden_channels,
            time_emb_dim=vc.time_emb_dim,
            N_outer=vc.N_outer,
            sigma_prior=vc.sigma_prior,
            dropout=vc.dropout,
            param_loss_weight=jc.param_loss_weight,
            param_flow_channels=jc.get("param_flow_channels", None),
            train_tau_0_only=jc.train_tau_0_only,
            param_ref=jc.get("param_ref", None),
            param_flow_pool=jc.get("param_flow_pool", "mean"),
        )
    elif model_type == "joint_cfm_coupled":
        from oceanml3d.models.vanilla_cfm import JointCFMCoupled
        jcc = cfg.model.joint_cfm_coupled
        vc = cfg.model.vanilla_cfm
        model = JointCFMCoupled(
            state_dim=cfg.model.state_dim,
            param_dim=jcc.param_dim,
            hidden_channels=vc.hidden_channels,
            time_emb_dim=vc.time_emb_dim,
            N_outer=vc.N_outer,
            sigma_prior=vc.sigma_prior,
            dropout=vc.dropout,
            param_loss_weight=jcc.param_loss_weight,
            param_flow_channels=jcc.get("param_flow_channels", None),
            param_ref=jcc.get("param_ref", None),
            param_flow_pool=jcc.get("param_flow_pool", "mean"),
        )
    elif model_type == "joint_direct_unet":
        from oceanml3d.models.direct_unet import JointDirectUNet
        jdu = cfg.model.joint_direct_unet
        dc = cfg.model.direct_unet
        model = JointDirectUNet(
            state_dim=cfg.model.state_dim,
            param_dim=jdu.param_dim,
            hidden_channels=dc.hidden_channels,
            dropout=dc.dropout,
            param_loss_weight=jdu.param_loss_weight,
            param_head_channels=jdu.get("param_head_channels", None),
            param_ref=jdu.get("param_ref", None),
            param_head_pool=jdu.get("param_head_pool", "mean"),
            param_head_backbone=jdu.get("param_head_backbone", "cnn"),
        )
    elif model_type == "param_head":
        from oceanml3d.models.param_head import StateParamModel
        ph = cfg.model.param_head
        model = StateParamModel(
            state_dim=cfg.model.state_dim,
            param_dim=ph.param_dim,
            state_checkpoint=ph.get("state_checkpoint", None),
            state_model_type=ph.get("state_model_type", "direct_unet"),
            state_hidden_channels=ph.get("state_hidden_channels", None),
            state_cond_extra_dim=ph.get("state_cond_extra_dim", 0),
            param_head_channels=ph.get("param_head_channels", None),
            param_ref=ph.get("param_ref", None),
            param_head_pool=ph.get("param_head_pool", "mean"),
            state_source=ph.get("state_source", "l1b"),
            augment_derivatives=ph.get("augment_derivatives", False),
            device=device,
        )
    elif model_type == "param_head_unet":
        from oceanml3d.models.param_head import StateParamModel
        ph = cfg.model.param_head_unet
        model = StateParamModel(
            state_dim=cfg.model.state_dim,
            param_dim=ph.param_dim,
            state_checkpoint=ph.get("state_checkpoint", None),
            state_model_type=ph.get("state_model_type", "direct_unet"),
            state_hidden_channels=ph.get("state_hidden_channels", None),
            state_cond_extra_dim=ph.get("state_cond_extra_dim", 0),
            param_head_channels=ph.get("param_head_channels", None),
            param_ref=ph.get("param_ref", None),
            param_head_pool=ph.get("param_head_pool", "mean"),
            state_source=ph.get("state_source", "l1b"),
            backbone="unet",
            unet_hidden_channels=ph.get("hidden_channels", None),
            device=device,
        )
    elif model_type == "predict_state_cfm":
        from oceanml3d.models.vanilla_cfm import PredictStateCFM
        psc = cfg.model.predict_state_cfm
        param_dim = cfg.model.get("param_dim", 4)
        model = PredictStateCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=psc.hidden_channels,
            time_emb_dim=psc.time_emb_dim,
            N_outer=psc.N_outer,
            sigma_prior=psc.sigma_prior,
            dropout=psc.dropout,
            train_tau_0_only=psc.get("train_tau_0_only", False),
            param_dim=param_dim,
            cond_extra_dim=psc.cond_extra_dim,
        )
    elif model_type == "tweedie_cfm":
        from oceanml3d.models.vanilla_cfm import TweedieCFM
        tc = cfg.model.tweedie_cfm
        param_dim = cfg.model.get("param_dim", 4)
        model = TweedieCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=tc.hidden_channels,
            time_emb_dim=tc.time_emb_dim,
            K_inner=tc.K_inner,
            N_outer=tc.N_outer,
            sigma_prior=tc.sigma_prior,
            dropout=tc.dropout,
            train_tau_0_only=tc.train_tau_0_only,
            cond_extra_dim=tc.cond_extra_dim,
        )
    elif model_type == "sda_prior":
        from oceanml3d.models.sda import UnconditionalPriorCFM
        sp = cfg.model.sda_prior
        model = UnconditionalPriorCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=sp.hidden_channels,
            time_emb_dim=sp.time_emb_dim,
            N_outer=sp.N_outer,
            sigma_prior=sp.sigma_prior,
            dropout=sp.dropout,
        )
    elif model_type == "sda_prior_cond":
        from oceanml3d.models.sda import ConditionalPriorCFM
        sp = cfg.model.sda_prior
        model = ConditionalPriorCFM(
            state_dim=cfg.model.state_dim,
            param_dim=cfg.model.get("param_dim", 8),
            hidden_channels=sp.hidden_channels,
            time_emb_dim=sp.time_emb_dim,
            N_outer=sp.N_outer,
            sigma_prior=sp.sigma_prior,
            dropout=sp.dropout,
        )
    elif model_type == "fourdvarnet":
        from oceanml3d.models.fourdvarnet import FourDVarNetSolver
        fdv = cfg.model.fdv
        model = FourDVarNetSolver(
            state_dim=cfg.model.state_dim,
            hidden_channels=fdv.hidden_channels,
            time_emb_dim=fdv.time_emb_dim,
            N_outer=fdv.N_outer,
            dropout=fdv.dropout,
            update_input=fdv.update_input,
            R_var=fdv.get("R_var", 0.5),
            prior_weight=fdv.get("prior_weight", 1.0),
            clip_range=fdv.get("clip_range", 50.0),
            trainable_prior_weight=fdv.get("trainable_prior_weight", True),
            aux_var_cost_weight=fdv.get("aux_var_cost_weight", 0.0),
            prior_tau_conditioning=fdv.get("prior_tau_conditioning", False),
        )
    elif model_type == "fourdvarnet_cfm":
        from oceanml3d.models.fourdvarnet import FourDVarNetPredictStateCFM
        fc = cfg.model.fdv_cfm
        model = FourDVarNetPredictStateCFM(
            state_dim=cfg.model.state_dim,
            hidden_channels=fc.hidden_channels,
            time_emb_dim=fc.time_emb_dim,
            N_outer=fc.N_outer,
            K_inner=fc.K_inner,
            sigma_prior=fc.sigma_prior,
            dropout=fc.dropout,
            train_tau_0_only=fc.train_tau_0_only,
            update_input=fc.update_input,
            clip_range=fc.get("clip_range", 50.0),
            R_var=fc.get("R_var", 0.5),
            obs_weight=fc.get("obs_weight", 1.0),
            min_obs_weight=fc.get("min_obs_weight", 1e-3),
            trainable_obs_weight=fc.get("trainable_obs_weight", True),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)
