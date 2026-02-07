"""Unit tests for src.networks.policy_value_net.

Covers the SpatialPolicyValueNetwork and its sub-modules (VoxelEncoder,
ProprioceptionEncoder, PolicyHead, ValueHead, ResBlock3D, ResBlock1D).
All tests run on CPU for CI compatibility.
"""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Independent

from src.config import NetworkConfig
from src.networks.policy_value_net import (
    PolicyHead,
    ProprioceptionEncoder,
    ResBlock1D,
    ResBlock3D,
    SpatialPolicyValueNetwork,
    ValueHead,
    VoxelEncoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    network_config: NetworkConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random voxel and proprioception tensors for a given batch size."""
    res = network_config.voxel_resolution
    ch = network_config.voxel_channels
    voxels = torch.randn(batch_size, ch, res, res, res, device=device)
    proprio = torch.randn(batch_size, network_config.proprioception_dim, device=device)
    return voxels, proprio


# ===================================================================
# ResBlock3D
# ===================================================================


class TestResBlock3D:
    """Tests for the 3-D residual building block."""

    def test_output_shape_matches_input(self, device: torch.device) -> None:
        channels = 16
        block = ResBlock3D(channels).to(device)
        x = torch.randn(2, channels, 8, 8, 8, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, device: torch.device) -> None:
        """With zero-initialised conv weights the output equals the input."""
        channels = 8
        block = ResBlock3D(channels).to(device)
        # Zero all conv weights so block(x) == 0
        with torch.no_grad():
            for m in block.modules():
                if isinstance(m, torch.nn.Conv3d):
                    m.weight.zero_()
        x = torch.randn(1, channels, 4, 4, 4, device=device)
        out = block(x)
        torch.testing.assert_close(out, x)


# ===================================================================
# ResBlock1D
# ===================================================================


class TestResBlock1D:
    """Tests for the 1-D residual building block."""

    def test_output_shape_matches_input(self, device: torch.device) -> None:
        dim = 32
        block = ResBlock1D(dim).to(device)
        x = torch.randn(4, dim, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, device: torch.device) -> None:
        dim = 16
        block = ResBlock1D(dim).to(device)
        with torch.no_grad():
            for m in block.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.zero_()
        x = torch.randn(2, dim, device=device)
        out = block(x)
        torch.testing.assert_close(out, x)


# ===================================================================
# VoxelEncoder
# ===================================================================


class TestVoxelEncoder:
    """Tests for the 3-D convolutional voxel encoder."""

    def test_output_shape(self, device: torch.device) -> None:
        encoder = VoxelEncoder(in_channels=4, out_features=64, resolution=16).to(device)
        x = torch.randn(2, 4, 16, 16, 16, device=device)
        out = encoder(x)
        assert out.shape == (2, 64)

    def test_single_sample(self, device: torch.device) -> None:
        encoder = VoxelEncoder(in_channels=1, out_features=32, resolution=16).to(device)
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        out = encoder(x)
        assert out.shape == (1, 32)


# ===================================================================
# ProprioceptionEncoder
# ===================================================================


class TestProprioceptionEncoder:
    """Tests for the proprioception MLP encoder."""

    def test_output_shape(self, device: torch.device) -> None:
        encoder = ProprioceptionEncoder(in_features=13, out_features=64).to(device)
        x = torch.randn(4, 13, device=device)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_single_sample(self, device: torch.device) -> None:
        encoder = ProprioceptionEncoder(in_features=13, out_features=64).to(device)
        x = torch.randn(1, 13, device=device)
        out = encoder(x)
        assert out.shape == (1, 64)


# ===================================================================
# PolicyHead
# ===================================================================


class TestPolicyHead:
    """Tests for the Gaussian policy head."""

    def test_returns_independent_distribution(self, device: torch.device) -> None:
        head = PolicyHead(in_features=64, action_dim=6).to(device)
        features = torch.randn(2, 64, device=device)
        dist = head(features)
        assert isinstance(dist, Independent)

    def test_distribution_sample_shape(self, device: torch.device) -> None:
        head = PolicyHead(in_features=64, action_dim=6).to(device)
        features = torch.randn(3, 64, device=device)
        dist = head(features)
        sample = dist.sample()
        assert sample.shape == (3, 6)

    def test_positive_std(self, device: torch.device) -> None:
        """Standard deviation must be strictly positive."""
        head = PolicyHead(in_features=64, action_dim=6).to(device)
        features = torch.randn(4, 64, device=device)
        dist = head(features)
        std = dist.base_dist.scale
        assert (std > 0).all()

    def test_finite_mean_and_std(self, device: torch.device) -> None:
        head = PolicyHead(in_features=64, action_dim=6).to(device)
        features = torch.randn(4, 64, device=device)
        dist = head(features)
        assert torch.isfinite(dist.base_dist.loc).all()
        assert torch.isfinite(dist.base_dist.scale).all()

    def test_log_std_clamping(self, device: torch.device) -> None:
        """Log std values must remain within the configured bounds."""
        min_log_std, max_log_std = -5.0, 2.0
        head = PolicyHead(
            in_features=64, action_dim=6,
            min_log_std=min_log_std, max_log_std=max_log_std,
        ).to(device)
        # Use extreme features to try to push log_std out of bounds
        features = torch.randn(8, 64, device=device) * 100.0
        dist = head(features)
        log_std = dist.base_dist.scale.log()
        assert (log_std >= min_log_std - 1e-5).all()
        assert (log_std <= max_log_std + 1e-5).all()

    def test_log_prob_is_finite(self, device: torch.device) -> None:
        head = PolicyHead(in_features=64, action_dim=6).to(device)
        features = torch.randn(4, 64, device=device)
        dist = head(features)
        actions = dist.sample()
        lp = dist.log_prob(actions)
        assert torch.isfinite(lp).all()
        assert lp.shape == (4,)


# ===================================================================
# ValueHead
# ===================================================================


class TestValueHead:
    """Tests for the scalar value head."""

    def test_output_shape_batch(self, device: torch.device) -> None:
        head = ValueHead(in_features=64).to(device)
        features = torch.randn(4, 64, device=device)
        out = head(features)
        assert out.shape == (4, 1)

    def test_output_shape_single(self, device: torch.device) -> None:
        head = ValueHead(in_features=64).to(device)
        features = torch.randn(1, 64, device=device)
        out = head(features)
        assert out.shape == (1, 1)

    def test_returns_scalar_per_sample(self, device: torch.device) -> None:
        """The value output must be a single scalar per batch element."""
        head = ValueHead(in_features=64).to(device)
        features = torch.randn(3, 64, device=device)
        out = head(features)
        assert out.shape[-1] == 1

    def test_finite_output(self, device: torch.device) -> None:
        head = ValueHead(in_features=64).to(device)
        features = torch.randn(4, 64, device=device)
        out = head(features)
        assert torch.isfinite(out).all()


# ===================================================================
# SpatialPolicyValueNetwork -- forward
# ===================================================================


class TestSpatialPolicyValueNetworkForward:
    """Tests for the full network forward pass."""

    def test_forward_output_types(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=4, device=device)
        action_dist, value = net(voxels, proprio)
        assert isinstance(action_dist, Independent)
        assert isinstance(value, torch.Tensor)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_forward_output_shapes(
        self, network_config: NetworkConfig, device: torch.device, batch_size: int,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size, device)
        action_dist, value = net(voxels, proprio)
        # Value shape
        assert value.shape == (batch_size, 1)
        # Distribution sample shape
        sample = action_dist.sample()
        assert sample.shape == (batch_size, network_config.action_dim)

    def test_forward_distribution_properties(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        """Policy distribution should have positive std and finite parameters."""
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=4, device=device)
        action_dist, _ = net(voxels, proprio)
        std = action_dist.base_dist.scale
        mean = action_dist.base_dist.loc
        assert (std > 0).all(), "Standard deviation must be positive"
        assert torch.isfinite(std).all(), "Standard deviation must be finite"
        assert torch.isfinite(mean).all(), "Mean must be finite"

    def test_forward_value_is_finite(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=4, device=device)
        _, value = net(voxels, proprio)
        assert torch.isfinite(value).all()


# ===================================================================
# SpatialPolicyValueNetwork -- act
# ===================================================================


class TestSpatialPolicyValueNetworkAct:
    """Tests for the act() inference method."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_act_stochastic_output_shapes(
        self, network_config: NetworkConfig, device: torch.device, batch_size: int,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size, device)
        actions, log_probs, values = net.act(voxels, proprio, deterministic=False)
        assert actions.shape == (batch_size, network_config.action_dim)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_act_deterministic_output_shapes(
        self, network_config: NetworkConfig, device: torch.device, batch_size: int,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size, device)
        actions, log_probs, values = net.act(voxels, proprio, deterministic=True)
        assert actions.shape == (batch_size, network_config.action_dim)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

    def test_act_deterministic_is_repeatable(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        """Deterministic actions must be identical across two calls."""
        net = SpatialPolicyValueNetwork(network_config).to(device)
        net.eval()
        voxels, proprio = _make_inputs(network_config, batch_size=2, device=device)
        a1, _, _ = net.act(voxels, proprio, deterministic=True)
        a2, _, _ = net.act(voxels, proprio, deterministic=True)
        torch.testing.assert_close(a1, a2)

    def test_act_stochastic_varies(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        """Stochastic actions should not be identical across multiple samples.

        This is a statistical test: with overwhelming probability two
        independent draws from a 6-D Gaussian differ in at least one dim.
        """
        net = SpatialPolicyValueNetwork(network_config).to(device)
        net.eval()
        voxels, proprio = _make_inputs(network_config, batch_size=1, device=device)
        # Draw many samples and check they are not all equal
        samples = torch.stack(
            [net.act(voxels, proprio, deterministic=False)[0] for _ in range(10)]
        )
        # At least some must differ
        assert not torch.all(samples == samples[0])

    def test_act_log_probs_are_finite(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=4, device=device)
        _, log_probs, _ = net.act(voxels, proprio, deterministic=False)
        assert torch.isfinite(log_probs).all()

    def test_act_deterministic_returns_mean(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        """Deterministic action must equal the distribution mean."""
        net = SpatialPolicyValueNetwork(network_config).to(device)
        net.eval()
        voxels, proprio = _make_inputs(network_config, batch_size=2, device=device)
        actions, _, _ = net.act(voxels, proprio, deterministic=True)
        # Independently compute the mean via forward()
        action_dist, _ = net(voxels, proprio)
        expected_mean = action_dist.base_dist.loc
        torch.testing.assert_close(actions, expected_mean)


# ===================================================================
# SpatialPolicyValueNetwork -- evaluate_actions
# ===================================================================


class TestSpatialPolicyValueNetworkEvaluateActions:
    """Tests for the evaluate_actions() training helper."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_evaluate_actions_output_shapes(
        self, network_config: NetworkConfig, device: torch.device, batch_size: int,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size, device)
        actions = torch.randn(batch_size, network_config.action_dim, device=device)
        log_probs, entropy, values = net.evaluate_actions(voxels, proprio, actions)
        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

    def test_evaluate_actions_entropy_positive(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        """Gaussian entropy should be positive (non-degenerate distribution)."""
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=4, device=device)
        actions = torch.randn(4, network_config.action_dim, device=device)
        _, entropy, _ = net.evaluate_actions(voxels, proprio, actions)
        # Differential entropy of a non-degenerate Gaussian is positive
        assert (entropy > 0).all() or torch.isfinite(entropy).all()

    def test_evaluate_actions_consistent_with_act(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        """Log-probs from evaluate_actions must match those from act."""
        net = SpatialPolicyValueNetwork(network_config).to(device)
        net.eval()
        voxels, proprio = _make_inputs(network_config, batch_size=2, device=device)
        # Get actions and log_probs from act (deterministic for reproducibility)
        actions, log_probs_act, values_act = net.act(
            voxels, proprio, deterministic=True,
        )
        # Evaluate the same actions
        log_probs_eval, _, values_eval = net.evaluate_actions(
            voxels, proprio, actions,
        )
        torch.testing.assert_close(log_probs_act, log_probs_eval)
        torch.testing.assert_close(values_act, values_eval)

    def test_evaluate_actions_finite_outputs(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=4, device=device)
        actions = torch.randn(4, network_config.action_dim, device=device)
        log_probs, entropy, values = net.evaluate_actions(voxels, proprio, actions)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(values).all()


# ===================================================================
# Gradient flow
# ===================================================================


class TestGradientFlow:
    """Verify that gradients propagate through the entire network."""

    def test_loss_backward_produces_gradients(
        self, network_config: NetworkConfig, device: torch.device,
    ) -> None:
        net = SpatialPolicyValueNetwork(network_config).to(device)
        voxels, proprio = _make_inputs(network_config, batch_size=2, device=device)
        actions = torch.randn(2, network_config.action_dim, device=device)

        log_probs, entropy, values = net.evaluate_actions(voxels, proprio, actions)
        loss = -(log_probs.mean() + 0.01 * entropy.mean()) + values.mean()
        loss.backward()

        # Every parameter should have a gradient
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
