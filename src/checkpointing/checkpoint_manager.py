"""Versioned checkpoint manager with backward-compatible loading.

Checkpoints include model weights, optimizer state, RNG states, episode
counter, and a snapshot of the configuration for full reproducibility.
A version field enables migrations when the checkpoint schema evolves.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

from src.config import SystemConfig

logger = structlog.get_logger(__name__)

CURRENT_VERSION = "1.0.0"


def _compare_versions(a: str, b: str) -> int:
    """Compare two semver strings.  Returns -1, 0, or 1."""
    ta = tuple(int(x) for x in a.split("."))
    tb = tuple(int(x) for x in b.split("."))
    return (ta > tb) - (ta < tb)


class CheckpointManager:
    """Save / load / migrate model checkpoints.

    Args:
        checkpoint_dir: Base directory for checkpoint files.
        max_to_keep: Maximum number of checkpoint files to retain.
            ``0`` means unlimited.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_to_keep: int = 0,
    ) -> None:
        self._dir = checkpoint_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_to_keep = max_to_keep

    def save(
        self,
        policy_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        episode: int,
        config: SystemConfig,
        metrics: dict[str, float] | None = None,
        tag: str | None = None,
    ) -> Path:
        """Write a checkpoint atomically.

        Args:
            policy_net: The policy / value network.
            optimizer: Current optimizer.
            episode: Episode index.
            config: Configuration snapshot.
            metrics: Optional training metrics to embed.
            tag: Optional filename tag (e.g. ``"best"``).  Defaults to
                ``episode_{N}``.

        Returns:
            Path to the saved checkpoint file.
        """
        filename = tag or f"episode_{episode}"
        path = self._dir / f"{filename}.pt"

        checkpoint: dict[str, Any] = {
            "version": CURRENT_VERSION,
            "episode": episode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_net_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.model_dump(mode="json"),
            "metrics": metrics or {},
            "rng_states": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        # Atomic write: save to temp then replace (replace works cross-platform)
        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)

        logger.info("checkpoint_saved", path=str(path), episode=episode)
        self._prune_old_checkpoints()
        return path

    def load(
        self,
        path: Path,
        policy_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> tuple[int, SystemConfig, dict[str, float]]:
        """Load a checkpoint, migrating if needed.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            policy_net: Network to load weights into.
            optimizer: If provided, restore optimizer state.
            device: Target device for tensors.
            restore_rng: Whether to restore RNG states.

        Returns:
            ``(episode, config, metrics)`` tuple.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        version = checkpoint.get("version", "0.0.0")
        if version != CURRENT_VERSION:
            logger.warning(
                "checkpoint_version_mismatch",
                file_version=version,
                current_version=CURRENT_VERSION,
            )
            checkpoint = self._migrate(checkpoint, from_version=version)

        # Model weights (with strict=False for backward compat)
        missing, unexpected = policy_net.load_state_dict(
            checkpoint["policy_net_state_dict"], strict=False
        )
        if missing:
            logger.warning("checkpoint_missing_keys", keys=missing)
        if unexpected:
            logger.warning("checkpoint_unexpected_keys", keys=unexpected)

        # Optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError) as exc:
                logger.warning("optimizer_state_load_failed", error=str(exc))

        # RNG states
        if restore_rng and "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])

        config = SystemConfig(**checkpoint.get("config", {}))
        episode = checkpoint.get("episode", 0)
        metrics = checkpoint.get("metrics", {})

        logger.info(
            "checkpoint_loaded",
            path=str(path),
            episode=episode,
            version=version,
        )
        return episode, config, metrics

    def latest(self) -> Path | None:
        """Find the most recently modified checkpoint in the directory."""
        candidates = sorted(self._dir.glob("episode_*.pt"), key=lambda p: p.stat().st_mtime)
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------ #
    # Migrations
    # ------------------------------------------------------------------ #

    def _migrate(self, ckpt: dict[str, Any], from_version: str) -> dict[str, Any]:
        """Apply sequential migrations to reach CURRENT_VERSION."""
        if _compare_versions(from_version, "1.0.0") < 0:
            # Pre-1.0.0: config might lack new fields
            cfg = ckpt.get("config", {})
            cfg.setdefault("reward", {})
            cfg.get("network", {}).setdefault("voxel_channels", 4)
            cfg.get("network", {}).setdefault("proprioception_dim", 13)
            ckpt["config"] = cfg
            logger.info("checkpoint_migrated", from_version=from_version, to_version="1.0.0")

        ckpt["version"] = CURRENT_VERSION
        return ckpt

    def _prune_old_checkpoints(self) -> None:
        """Delete oldest checkpoints to stay within max_to_keep budget."""
        if self._max_to_keep <= 0:
            return

        candidates = sorted(self._dir.glob("episode_*.pt"), key=lambda p: p.stat().st_mtime)
        while len(candidates) > self._max_to_keep:
            oldest = candidates.pop(0)
            oldest.unlink()
            logger.info("checkpoint_pruned", path=str(oldest))
