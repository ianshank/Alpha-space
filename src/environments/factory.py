"""Environment factory — creates the correct simulator backend from config."""

from __future__ import annotations

import structlog

from src.config import SystemConfig
from src.environments.base import ZeroGEnv
from src.environments.mock_env import MockZeroGEnv

logger = structlog.get_logger(__name__)

_REGISTRY: dict[str, type[ZeroGEnv]] = {
    "mock": MockZeroGEnv,
}


def register_env(name: str, cls: type[ZeroGEnv]) -> None:
    """Register an environment class at runtime.

    Args:
        name: Key used in ``EnvironmentConfig.simulator``.
        cls: A concrete ``ZeroGEnv`` subclass.
    """
    _REGISTRY[name] = cls
    logger.info("env_registered", name=name, cls=cls.__qualname__)


def make_env(config: SystemConfig) -> ZeroGEnv:
    """Create a single environment instance from the system config.

    Args:
        config: Full system configuration.

    Returns:
        A ``ZeroGEnv`` subclass instance.

    Raises:
        ValueError: If the configured simulator is not registered.
    """
    sim_name = config.environment.simulator
    cls = _REGISTRY.get(sim_name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown simulator '{sim_name}'. Available: {available}")

    env = cls(
        env_config=config.environment,
        reward_config=config.reward,
        voxel_resolution=config.network.voxel_resolution,
        voxel_channels=config.network.voxel_channels,
    )
    logger.info("env_created", simulator=sim_name)
    return env


def make_envs(config: SystemConfig, n: int | None = None) -> list[ZeroGEnv]:
    """Create multiple parallel environment instances.

    Args:
        config: Full system configuration.
        n: Number of instances. Defaults to ``config.environment.parallel_envs``.

    Returns:
        List of ``ZeroGEnv`` instances.
    """
    count = n if n is not None else config.environment.parallel_envs
    return [make_env(config) for _ in range(count)]
