"""Dual-headed policy/value network for 3D spatial control.

Architecture::

    Voxel input  → VoxelEncoder (3D CNN)
    Proprio input → ProprioceptionEncoder (MLP)
         ↓                ↓
      concat → SharedTrunk (ResNet blocks)
                    ↓             ↓
            PolicyHead       ValueHead
        (Gaussian 6-DOF)     (scalar V)

The policy head outputs parameters of a diagonal Gaussian distribution
over the 6-DOF action space (3 forces + 3 torques).  The value head
outputs a scalar state-value estimate.
"""

from __future__ import annotations

import math

import structlog
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.config import NetworkConfig

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Named architecture constants
# ---------------------------------------------------------------------------

_VOXEL_CONV_KERNEL: int = 3
_VOXEL_CONV_STRIDE: int = 2
_VOXEL_BASE_CHANNELS: int = 64
_VOXEL_MAX_CHANNELS: int = 256
_VALUE_HEAD_REDUCTION: int = 2


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock3D(nn.Module):  # type: ignore[misc]
    """Pre-activation 3D residual block with optional channel projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResBlock1D(nn.Module):  # type: ignore[misc]
    """1-D residual block for the shared trunk after flattening."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class VoxelEncoder(nn.Module):  # type: ignore[misc]
    """3D convolutional encoder for voxelised spatial observations.

    Processes a ``(B, C, D, H, W)`` voxel grid through a series of strided
    convolutions and outputs a flat feature vector.
    """

    def __init__(self, in_channels: int, out_features: int, resolution: int) -> None:
        super().__init__()
        # Determine number of downsampling stages: halve until ≤4
        n_stages = max(1, int(math.log2(resolution)) - 2)
        layers: list[nn.Module] = []
        ch = in_channels
        for i in range(n_stages):
            out_ch = min(_VOXEL_BASE_CHANNELS * (2**i), _VOXEL_MAX_CHANNELS)
            layers += [
                nn.Conv3d(
                    ch,
                    out_ch,
                    kernel_size=_VOXEL_CONV_KERNEL,
                    stride=_VOXEL_CONV_STRIDE,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)
        # Determine output size by a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, resolution, resolution, resolution)
            flat_size = self.encoder(dummy).shape[1]
        self.projection = nn.Linear(flat_size, out_features)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Encode voxel grid to feature vector.

        Args:
            voxels: ``(B, C, D, H, W)`` voxel tensor.

        Returns:
            ``(B, out_features)`` feature vector.
        """
        return self.projection(self.encoder(voxels))


class ProprioceptionEncoder(nn.Module):  # type: ignore[misc]
    """MLP encoder for spacecraft proprioception (pose + velocity)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


# ---------------------------------------------------------------------------
# Policy and value heads
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):  # type: ignore[misc]
    """Outputs a diagonal Gaussian distribution over continuous actions.

    The log standard deviation is learned but clamped to
    ``[min_log_std, max_log_std]``.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_head = nn.Linear(in_features, action_dim)
        self.log_std_head = nn.Linear(in_features, action_dim)
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std

    def forward(self, features: torch.Tensor) -> Independent:
        """Return an ``Independent(Normal(...), 1)`` action distribution."""
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self._min_log_std, self._max_log_std)
        return Independent(Normal(mean, log_std.exp()), reinterpreted_batch_ndims=1)


class ValueHead(nn.Module):  # type: ignore[misc]
    """Scalar state-value estimation."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        mid = in_features // _VALUE_HEAD_REDUCTION
        self.net = nn.Sequential(
            nn.Linear(in_features, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------


class SpatialPolicyValueNetwork(nn.Module):  # type: ignore[misc]
    """Dual-headed policy/value network for 3D spatial control.

    Args:
        config: Network architecture hyper-parameters.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        # Encoders
        self.voxel_encoder = VoxelEncoder(
            in_channels=config.voxel_channels,
            out_features=config.hidden_dim,
            resolution=config.voxel_resolution,
        )
        self.proprio_encoder = ProprioceptionEncoder(
            in_features=config.proprioception_dim,
            out_features=config.hidden_dim,
        )

        # Fusion projection (concat of two hidden_dim vectors → hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            *[ResBlock1D(config.hidden_dim) for _ in range(config.num_res_blocks)]
        )

        # Heads
        self.policy_head = PolicyHead(
            in_features=config.hidden_dim,
            action_dim=config.action_dim,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
        )
        self.value_head = ValueHead(in_features=config.hidden_dim)

        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        logger.info("network_created", total_params=total_params, config=config.model_dump())

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for linear layers, zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv3d,)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        voxels: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[Independent, torch.Tensor]:
        """Forward pass returning action distribution and state value.

        Args:
            voxels: ``(B, C, D, H, W)`` voxel grid.
            proprio: ``(B, proprio_dim)`` proprioception vector.

        Returns:
            A tuple ``(action_distribution, value)`` where *action_distribution*
            is a ``torch.distributions.Independent`` wrapping a diagonal Normal
            and *value* is ``(B, 1)``.
        """
        voxel_features = self.voxel_encoder(voxels)
        proprio_features = self.proprio_encoder(proprio)
        fused = self.fusion(torch.cat([voxel_features, proprio_features], dim=-1))
        trunk_out = self.trunk(fused)
        action_dist = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)
        return action_dist, value

    def act(
        self,
        voxels: torch.Tensor,
        proprio: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or select action for inference.

        Args:
            voxels: ``(B, C, D, H, W)`` voxel grid.
            proprio: ``(B, proprio_dim)`` proprioception vector.
            deterministic: If ``True``, return the distribution mean.

        Returns:
            ``(actions, log_probs, values)`` where shapes are
            ``(B, action_dim)``, ``(B,)``, ``(B, 1)`` respectively.
        """
        action_dist, value = self.forward(voxels, proprio)
        actions = action_dist.base_dist.loc if deterministic else action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        return actions, log_probs, value

    def evaluate_actions(
        self,
        voxels: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of given actions.

        Used during policy updates to compute the loss.

        Args:
            voxels: ``(B, C, D, H, W)`` voxel grid.
            proprio: ``(B, proprio_dim)`` proprioception vector.
            actions: ``(B, action_dim)`` actions to evaluate.

        Returns:
            ``(log_probs, entropy, values)`` with shapes
            ``(B,)``, ``(B,)``, ``(B, 1)``.
        """
        action_dist, value = self.forward(voxels, proprio)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_probs, entropy, value
