"""Pydantic v2 configuration models for the ZeroG-RL system.

All hyperparameters, paths, and external service settings are defined as
typed, validated Pydantic models.  No hardcoded values in application code —
everything flows through these configs which can be loaded from YAML files
and overridden via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class NetworkConfig(BaseModel):
    """Policy / Value network architecture hyperparameters."""

    voxel_resolution: int = Field(
        default=64, ge=16, le=128, description="Voxel grid size (NxNxN)"
    )
    voxel_channels: int = Field(
        default=4, ge=1, description="Channels per voxel (occupancy, velocity, etc.)"
    )
    proprioception_dim: int = Field(
        default=13, ge=1, description="Proprioception vector size (pose 7 + velocity 6)"
    )
    hidden_dim: int = Field(
        default=256, ge=64, description="Shared trunk hidden dimension"
    )
    num_res_blocks: int = Field(
        default=6, ge=1, le=50, description="Number of ResNet blocks in trunk"
    )
    action_dim: int = Field(
        default=6, ge=1, description="Action dimensionality (6-DOF)"
    )
    min_log_std: float = Field(
        default=-5.0, description="Minimum log-std for policy Gaussian"
    )
    max_log_std: float = Field(
        default=2.0, description="Maximum log-std for policy Gaussian"
    )

    @field_validator("voxel_resolution")
    @classmethod
    def _voxel_resolution_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"voxel_resolution must be a power of 2, got {v}")
        return v


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search hyperparameters."""

    num_simulations: int = Field(
        default=800, ge=1, description="MCTS rollouts per action selection"
    )
    c_puct: float = Field(
        default=1.0, gt=0.0, description="UCT exploration constant"
    )
    max_children: int = Field(
        default=32, ge=1, description="Progressive widening limit per node"
    )
    temperature: float = Field(
        default=1.0, ge=0.0, description="Action selection temperature (0 = greedy)"
    )
    dirichlet_alpha: float = Field(
        default=0.25, gt=0.0, description="Dirichlet noise alpha at root"
    )
    dirichlet_epsilon: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Weight of Dirichlet noise at root"
    )
    discount: float = Field(
        default=0.99, ge=0.0, le=1.0, description="Reward discount factor"
    )


class TrainingConfig(BaseModel):
    """Training loop hyperparameters."""

    num_episodes: int = Field(
        default=1000, ge=1, description="Total self-play episodes"
    )
    batch_size: int = Field(
        default=256, ge=1, description="Mini-batch size for policy updates"
    )
    learning_rate: float = Field(
        default=1e-4, gt=0.0, description="Adam optimizer learning rate"
    )
    weight_decay: float = Field(
        default=1e-4, ge=0.0, description="L2 regularisation"
    )
    gradient_clip_norm: float = Field(
        default=1.0, gt=0.0, description="Max gradient L2 norm"
    )
    checkpoint_interval: int = Field(
        default=100, ge=1, description="Save checkpoint every N episodes"
    )
    eval_interval: int = Field(
        default=50, ge=1, description="Evaluate policy every N episodes"
    )
    replay_buffer_size: int = Field(
        default=10_000, ge=1, description="Max transitions in replay buffer"
    )
    value_loss_weight: float = Field(
        default=1.0, gt=0.0, description="Weight for value loss in total loss"
    )
    entropy_weight: float = Field(
        default=0.01, ge=0.0, description="Entropy bonus weight"
    )
    epochs_per_update: int = Field(
        default=4, ge=1, description="Gradient epochs per batch of experience"
    )


class EnvironmentConfig(BaseModel):
    """Simulation environment settings."""

    simulator: str = Field(
        default="mock",
        pattern=r"^(mock|unity|isaac_sim)$",
        description="Physics simulator backend",
    )
    parallel_envs: int = Field(
        default=1, ge=1, description="Number of parallel simulation instances"
    )
    max_episode_steps: int = Field(
        default=500, ge=1, description="Max timesteps per episode"
    )
    workspace_size: float = Field(
        default=10.0, gt=0.0, description="Workspace extent in metres (cube half-size)"
    )
    time_step: float = Field(
        default=0.05, gt=0.0, description="Physics timestep in seconds"
    )
    max_thrust: float = Field(
        default=10.0, gt=0.0, description="Maximum thrust force (N)"
    )
    max_torque: float = Field(
        default=2.0, gt=0.0, description="Maximum torque (N·m)"
    )
    position_tolerance: float = Field(
        default=0.1, gt=0.0, description="Docking success position threshold (m)"
    )
    orientation_tolerance_deg: float = Field(
        default=5.0, gt=0.0, description="Docking success orientation threshold (°)"
    )
    spacecraft_mass: float = Field(
        default=100.0, gt=0.0, description="Spacecraft mass (kg)"
    )
    spacecraft_inertia: tuple[float, float, float] = Field(
        default=(10.0, 10.0, 10.0),
        description="Principal moments of inertia (Ixx, Iyy, Izz) in kg·m²",
    )


class RewardConfig(BaseModel):
    """Reward shaping parameters."""

    position_weight: float = Field(
        default=1.0, ge=0.0, description="Weight for position error penalty"
    )
    orientation_weight: float = Field(
        default=0.5, ge=0.0, description="Weight for orientation error penalty"
    )
    velocity_weight: float = Field(
        default=0.1, ge=0.0, description="Weight for velocity penalty"
    )
    fuel_weight: float = Field(
        default=0.01, ge=0.0, description="Weight for fuel usage penalty"
    )
    success_bonus: float = Field(
        default=100.0, ge=0.0, description="Bonus for successful docking"
    )
    collision_penalty: float = Field(
        default=-50.0, le=0.0, description="Penalty for collision"
    )
    time_penalty: float = Field(
        default=-0.1, le=0.0, description="Per-step time penalty"
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class SystemConfig(BaseModel):
    """Root configuration container for the entire system."""

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    # Paths
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"), description="Directory for model checkpoints"
    )
    log_dir: Path = Field(
        default=Path("logs"), description="Directory for training logs"
    )

    # External services
    wandb_project: str = Field(default="zerog-rl", description="W&B project name")
    wandb_entity: str | None = Field(default=None, description="W&B team/user entity")

    # Reproducibility
    seed: int = Field(default=42, ge=0, description="Global random seed")

    # Runtime flags
    use_gpu: bool = Field(default=True, description="Prefer GPU if available")
    debug: bool = Field(default=False, description="Enable debug logging and checks")
    no_logging: bool = Field(
        default=False, description="Disable external metrics logging"
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_ENV_PREFIX = "ZEROG_"

_ENV_MAP: dict[str, str] = {
    f"{_ENV_PREFIX}VOXEL_RESOLUTION": "network.voxel_resolution",
    f"{_ENV_PREFIX}HIDDEN_DIM": "network.hidden_dim",
    f"{_ENV_PREFIX}NUM_RES_BLOCKS": "network.num_res_blocks",
    f"{_ENV_PREFIX}MCTS_SIMULATIONS": "mcts.num_simulations",
    f"{_ENV_PREFIX}MCTS_TEMPERATURE": "mcts.temperature",
    f"{_ENV_PREFIX}LEARNING_RATE": "training.learning_rate",
    f"{_ENV_PREFIX}BATCH_SIZE": "training.batch_size",
    f"{_ENV_PREFIX}NUM_EPISODES": "training.num_episodes",
    f"{_ENV_PREFIX}SIMULATOR": "environment.simulator",
    f"{_ENV_PREFIX}PARALLEL_ENVS": "environment.parallel_envs",
    f"{_ENV_PREFIX}SEED": "seed",
    f"{_ENV_PREFIX}WANDB_PROJECT": "wandb_project",
    f"{_ENV_PREFIX}WANDB_ENTITY": "wandb_entity",
}


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated keys."""
    parts = dotted_key.split(".")
    obj = data
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value


def _auto_cast(value: str) -> int | float | bool | str:
    """Attempt to cast a string to a more specific Python scalar."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SystemConfig:
    """Load system configuration from an optional YAML file.

    Resolution order (later wins):
        1. Pydantic defaults
        2. YAML file values
        3. Environment variable overrides (``ZEROG_`` prefix)
        4. Explicit *overrides* dict

    Args:
        config_path: Path to a YAML configuration file.  When ``None``, only
            defaults and overrides are used.
        overrides: Explicit key-value overrides using dotted notation
            (e.g. ``{"training.learning_rate": 0.001}``).

    Returns:
        A validated ``SystemConfig`` instance.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        pydantic.ValidationError: If the merged configuration is invalid.
    """
    config_dict: dict[str, Any] = {}

    # 1. YAML file
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            config_dict = loaded

    # 2. Environment variable overrides
    for env_key, dotted_key in _ENV_MAP.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            _set_nested(config_dict, dotted_key, _auto_cast(env_val))

    # 3. Explicit overrides
    if overrides:
        for dotted_key, value in overrides.items():
            _set_nested(config_dict, dotted_key, value)

    return SystemConfig(**config_dict)
