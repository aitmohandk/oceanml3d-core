import torch

from oceanml3d.models.vanilla_cfm import JointCFM


class _MockBatch:
    def __init__(self, B=2, T=50, D=3, params=None, true_params=None):
        self.states = torch.randn(B, T, D)
        self.obs = torch.randn(B, T, D)
        self.obs_mask = torch.ones(B, T, dtype=torch.bool)
        self.forcing = torch.randn(B, T)
        self.params = params if params is not None else torch.randn(B, 4)
        self.true_params = true_params
        self.batch_size = B


class TestJointCFM:
    def test_forward_returns_state_velocity_param_velocity_and_state_estimate(self):
        """`forward` returns `(v_state, v_param, x_hat_1)`, and the param branch is optional.

        This test used to unpack two values and call the second one `param_feats`. The signature
        now carries the analytic state estimate as a third output and gates the param velocity on
        a `param_tau` interpolant being supplied, so the old unpacking could not run at all.
        """
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8])
        B, T = 2, 30
        x_t = torch.randn(B, T, 3)
        batch = _MockBatch(B=B, T=T, D=3)
        tau = torch.rand(B)

        v_state, v_param, x_hat_1 = model(x_t, batch, tau)
        assert v_state.shape == (B, T, 3)
        assert x_hat_1.shape == (B, T, 3)
        assert v_param is None, "no param interpolant was supplied, so there is no param velocity"

        # the analytic state estimate documented on the class: x_hat_1 = x_t + (1 - tau) * u_theta
        torch.testing.assert_close(x_hat_1, x_t + (1.0 - tau).view(-1, 1, 1) * v_state)

        v_state, v_param, _ = model(x_t, batch, tau, param_tau=torch.randn(B, 3))
        assert v_param.shape == (B, 3), "the param flow pools over time, one velocity per sample"

    def test_sample_params_from_state_shape_and_positivity(self):
        """`estimate_params` never existed on this class; the ens-then-head entry point is
        `sample_params_from_state`, which integrates the param flow against a given state estimate.

        (`sample(..., return_params=True)` is the joint path and is covered by
        `test_sample_with_params_shape`.)
        """
        torch.manual_seed(0)
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], N_outer=3)
        B, T = 2, 30
        batch = _MockBatch(B=B, T=T, D=3)
        params = model.sample_params_from_state(batch, torch.randn(B, T, 3))
        assert params.shape == (B, 3)
        assert (params > 0).all(), f"denormalised params must stay positive, got {params}"

    def test_loss_finite_without_params(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8])
        batch = _MockBatch(B=2, T=30, D=3)
        loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_loss_finite_with_params(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8])
        true_params = torch.tensor([[10.0, 28.0, 2.66], [9.5, 27.0, 2.5]])
        batch = _MockBatch(B=2, T=30, D=3, true_params=true_params)
        loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_loss_with_params_differs_from_without(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8])
        base_params = torch.tensor([[10.0, 28.0, 2.66]])
        batch_no = _MockBatch(B=1, T=30, D=3)
        batch_yes = _MockBatch(B=1, T=30, D=3, true_params=base_params)
        loss_no = model.compute_cfm_loss(batch_no)
        loss_yes = model.compute_cfm_loss(batch_yes)
        assert not torch.allclose(loss_no, loss_yes)

    def test_sample_shape(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], N_outer=3)
        batch = _MockBatch(B=2, T=30, D=3)
        samples = model.sample(batch)
        assert samples.shape == (2, 30, 3)

    def test_sample_with_params_shape(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], N_outer=3)
        batch = _MockBatch(B=2, T=30, D=3)
        samples, params = model.sample(batch, return_params=True)
        assert samples.shape == (2, 30, 3)
        assert params.shape == (2, 3)
        assert (params > 0).all()

    def test_sample_finite(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], N_outer=3)
        model.eval()
        batch = _MockBatch(B=1, T=30, D=3)
        with torch.no_grad():
            samples = model.sample(batch)
        assert torch.isfinite(samples).all()

    def test_tau0_mode_loss(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], train_tau_0_only=True)
        batch = _MockBatch(B=2, T=30, D=3)
        loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss)

    def test_tau0_mode_sample(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], N_outer=3, train_tau_0_only=True)
        batch = _MockBatch(B=1, T=30, D=3)
        samples = model.sample(batch)
        assert samples.shape == (1, 30, 3)
        assert torch.isfinite(samples).all()

    def test_param_loss_weight_zero_ignores_params(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8], param_loss_weight=0.0)
        from oceanml3d.data.dataloader import FlowMatchingBatch
        B, T = 1, 30
        batch_no_base = _MockBatch(B=B, T=T, D=3)
        batch_no = FlowMatchingBatch(
            batch_no_base.states, batch_no_base.obs, batch_no_base.obs_mask,
            batch_no_base.forcing, params=batch_no_base.params)
        torch.manual_seed(0)
        loss_no = model.compute_cfm_loss(batch_no)
        torch.manual_seed(0)
        batch_yes = FlowMatchingBatch(
            batch_no_base.states, batch_no_base.obs, batch_no_base.obs_mask,
            batch_no_base.forcing, params=batch_no_base.params,
            true_params=torch.tensor([[10.0, 28.0, 2.66]]))
        loss_yes = model.compute_cfm_loss(batch_yes)
        assert torch.allclose(loss_no, loss_yes)

    def test_unet_output_dim(self):
        from oceanml3d.models.unet import UNet1D
        model = UNet1D(state_dim=3, hidden_channels=[4, 8], use_obs=True,
                       output_dim=6, obs_dim=6)
        x = torch.randn(1, 3, 30)
        obs = torch.randn(1, 6, 30)
        out = model(x, obs)
        assert out.shape == (1, 6, 30)

    def test_nan_obs_zeroed(self):
        model = JointCFM(state_dim=3, param_dim=3, hidden_channels=[4, 8])
        batch = _MockBatch(B=1, T=30, D=3)
        batch.obs[0, 0] = float('nan')
        with torch.no_grad():
            loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss), "NaN in obs should be zeroed in conditioning"


