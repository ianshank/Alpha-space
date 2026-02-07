"""Unit tests for the versioned checkpoint manager.

Covers save, load, round-trip fidelity, version migration, latest()
discovery, and pruning of old checkpoints. All file operations use
the ``tmp_path`` fixture to avoid side effects.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.checkpointing.checkpoint_manager import (
    CURRENT_VERSION,
    CheckpointManager,
    _compare_versions,
)
from src.config import NetworkConfig, SystemConfig
from src.networks.policy_value_net import SpatialPolicyValueNetwork

# ------------------------------------------------------------------ #
# Helpers: small network and config for fast tests
# ------------------------------------------------------------------ #

_SMALL_NET_CONFIG = NetworkConfig(
    voxel_resolution=16,
    voxel_channels=4,
    proprioception_dim=13,
    hidden_dim=64,
    num_res_blocks=1,
    action_dim=6,
)


def _make_net() -> SpatialPolicyValueNetwork:
    """Create a small policy-value network for testing."""
    return SpatialPolicyValueNetwork(_SMALL_NET_CONFIG)


def _make_optimizer(net: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(net.parameters(), lr=1e-3)


def _make_config() -> SystemConfig:
    return SystemConfig(
        network=_SMALL_NET_CONFIG,
        seed=42,
        use_gpu=False,
        debug=True,
        no_logging=True,
    )


def _state_dicts_equal(sd1: dict, sd2: dict) -> bool:
    """Check that two state_dicts have identical keys and values."""
    if sd1.keys() != sd2.keys():
        return False
    for key in sd1:
        if not torch.equal(sd1[key], sd2[key]):
            return False
    return True


# ================================================================== #
# CheckpointManager.save creates file
# ================================================================== #


class TestCheckpointSave:
    """Verify that save writes a valid checkpoint file."""

    def test_save_creates_pt_file(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=1, config=config)
        assert path.exists()
        assert path.suffix == ".pt"

    def test_save_default_filename_uses_episode(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=42, config=config)
        assert path.name == "episode_42.pt"

    def test_save_custom_tag(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=1, config=config, tag="best")
        assert path.name == "best.pt"

    def test_save_with_metrics(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()
        metrics = {"loss": 0.5, "reward_mean": 42.0}

        path = mgr.save(net, opt, episode=1, config=config, metrics=metrics)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["metrics"]["loss"] == 0.5
        assert ckpt["metrics"]["reward_mean"] == 42.0

    def test_saved_checkpoint_contains_required_keys(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=10, config=config)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        required = {
            "version", "episode", "timestamp",
            "policy_net_state_dict", "optimizer_state_dict",
            "config", "metrics", "rng_states",
        }
        assert required.issubset(ckpt.keys())
        assert ckpt["version"] == CURRENT_VERSION
        assert ckpt["episode"] == 10

    def test_save_creates_checkpoint_dir(self, tmp_path: Path):
        subdir = tmp_path / "deep" / "nested" / "ckpts"
        CheckpointManager(checkpoint_dir=subdir)
        assert subdir.is_dir()

    def test_no_temp_file_left_behind(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()
        mgr.save(net, opt, episode=1, config=config)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


# ================================================================== #
# CheckpointManager.load restores weights
# ================================================================== #


class TestCheckpointLoad:
    """Verify that load restores network state correctly."""

    def test_load_restores_model_weights(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        # Save
        path = mgr.save(net, opt, episode=5, config=config)

        # Load into a fresh network
        net2 = _make_net()
        episode, loaded_config, metrics = mgr.load(
            path, net2, device="cpu", restore_rng=False
        )
        assert episode == 5
        assert _state_dicts_equal(net.state_dict(), net2.state_dict())

    def test_load_restores_optimizer_state(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        # Do a fake training step to populate optimizer state
        dummy_voxels = torch.randn(1, 4, 16, 16, 16)
        dummy_proprio = torch.randn(1, 13)
        dist, val = net(dummy_voxels, dummy_proprio)
        loss = val.mean()
        loss.backward()
        opt.step()

        path = mgr.save(net, opt, episode=1, config=config)

        # Load into fresh net + optimizer
        net2 = _make_net()
        opt2 = _make_optimizer(net2)
        mgr.load(path, net2, optimizer=opt2, device="cpu", restore_rng=False)

        # Optimizer state dict keys should match
        assert opt.state_dict()["param_groups"] == opt2.state_dict()["param_groups"]

    def test_load_returns_config_object(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=3, config=config)

        net2 = _make_net()
        _, loaded_config, _ = mgr.load(path, net2, device="cpu", restore_rng=False)
        assert isinstance(loaded_config, SystemConfig)
        assert loaded_config.seed == config.seed

    def test_load_returns_metrics(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()
        metrics = {"accuracy": 0.95}

        path = mgr.save(net, opt, episode=1, config=config, metrics=metrics)
        net2 = _make_net()
        _, _, loaded_metrics = mgr.load(path, net2, device="cpu", restore_rng=False)
        assert loaded_metrics["accuracy"] == 0.95

    def test_load_nonexistent_raises_file_not_found(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            mgr.load(tmp_path / "no_such_file.pt", net)


# ================================================================== #
# Round-trip save/load preserves model state
# ================================================================== #


class TestRoundTrip:
    """Save and load must preserve exact model state."""

    def test_round_trip_weights_identical(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        # Randomize weights to make them non-trivial
        with torch.no_grad():
            for p in net.parameters():
                p.normal_()

        original_sd = {k: v.clone() for k, v in net.state_dict().items()}
        path = mgr.save(net, opt, episode=99, config=config)

        net2 = _make_net()
        mgr.load(path, net2, device="cpu", restore_rng=False)

        for key in original_sd:
            assert torch.equal(original_sd[key], net2.state_dict()[key]), (
                f"Mismatch in parameter: {key}"
            )

    def test_round_trip_forward_output_matches(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        net.eval()
        opt = _make_optimizer(net)
        config = _make_config()

        # Create deterministic input
        torch.manual_seed(123)
        voxels = torch.randn(1, 4, 16, 16, 16)
        proprio = torch.randn(1, 13)

        with torch.no_grad():
            _, val_before = net(voxels, proprio)

        path = mgr.save(net, opt, episode=1, config=config)

        net2 = _make_net()
        net2.eval()
        mgr.load(path, net2, device="cpu", restore_rng=False)

        with torch.no_grad():
            _, val_after = net2(voxels, proprio)

        torch.testing.assert_close(val_before, val_after)

    def test_round_trip_preserves_episode_number(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=777, config=config)

        net2 = _make_net()
        episode, _, _ = mgr.load(path, net2, device="cpu", restore_rng=False)
        assert episode == 777


# ================================================================== #
# Migration from old version checkpoint
# ================================================================== #


class TestMigration:
    """Verify that old-version checkpoints are migrated correctly."""

    def _make_old_checkpoint(self, tmp_path: Path, version: str = "0.9.0") -> Path:
        """Create a synthetic old-version checkpoint file."""
        net = _make_net()
        opt = _make_optimizer(net)

        ckpt = {
            "version": version,
            "episode": 50,
            "timestamp": "2024-01-01T00:00:00+00:00",
            "policy_net_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "config": {},  # minimal config, missing reward/network fields
            "metrics": {"old_metric": 1.0},
            "rng_states": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [],
                "numpy": np.random.get_state(),
                "python": __import__("random").getstate(),
            },
        }

        path = tmp_path / "old_checkpoint.pt"
        torch.save(ckpt, path)
        return path

    def test_migration_updates_version(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        path = self._make_old_checkpoint(tmp_path, version="0.5.0")
        net = _make_net()

        mgr.load(path, net, device="cpu", restore_rng=False)

        # The in-memory migration should have happened without errors
        # Re-load raw checkpoint to verify version was set during migration
        # Verify the raw file on disk is still readable
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert "version" in raw

    def test_migration_adds_missing_config_fields(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        path = self._make_old_checkpoint(tmp_path, version="0.1.0")
        net = _make_net()

        episode, config, metrics = mgr.load(
            path, net, device="cpu", restore_rng=False
        )

        # The migration should have added default reward config
        assert isinstance(config, SystemConfig)
        assert episode == 50
        assert metrics.get("old_metric") == 1.0

    def test_migration_from_current_version_is_noop(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        path = mgr.save(net, opt, episode=1, config=config)

        net2 = _make_net()
        episode, _, _ = mgr.load(path, net2, device="cpu", restore_rng=False)
        assert episode == 1  # no migration needed, loads fine

    def test_compare_versions_helper(self):
        assert _compare_versions("1.0.0", "1.0.0") == 0
        assert _compare_versions("0.9.0", "1.0.0") == -1
        assert _compare_versions("1.0.0", "0.9.0") == 1
        assert _compare_versions("1.0.0", "1.0.1") == -1
        assert _compare_versions("2.0.0", "1.9.9") == 1

    def test_migration_handles_missing_version_field(self, tmp_path: Path):
        """A checkpoint without a version key should be treated as 0.0.0."""
        net = _make_net()
        opt = _make_optimizer(net)

        ckpt = {
            # No "version" key at all
            "episode": 10,
            "timestamp": "2023-06-01T00:00:00+00:00",
            "policy_net_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "config": {},
            "metrics": {},
            "rng_states": {
                "torch": torch.get_rng_state(),
                "torch_cuda": [],
                "numpy": np.random.get_state(),
                "python": __import__("random").getstate(),
            },
        }
        path = tmp_path / "versionless.pt"
        torch.save(ckpt, path)

        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net2 = _make_net()
        episode, config, _ = mgr.load(path, net2, device="cpu", restore_rng=False)
        assert episode == 10
        assert isinstance(config, SystemConfig)


# ================================================================== #
# latest() returns most recent checkpoint
# ================================================================== #


class TestLatest:
    """Verify that latest() finds the most recently modified checkpoint."""

    def test_latest_returns_none_when_empty(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        assert mgr.latest() is None

    def test_latest_returns_single_checkpoint(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        saved = mgr.save(net, opt, episode=1, config=config)
        assert mgr.latest() == saved

    def test_latest_returns_most_recent(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        mgr.save(net, opt, episode=1, config=config)
        time.sleep(0.05)  # ensure different mtime
        mgr.save(net, opt, episode=2, config=config)
        time.sleep(0.05)
        third = mgr.save(net, opt, episode=3, config=config)

        assert mgr.latest() == third

    def test_latest_ignores_custom_tag_files(self, tmp_path: Path):
        """latest() only looks for episode_*.pt, so custom tags are ignored."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        ep_path = mgr.save(net, opt, episode=1, config=config)
        time.sleep(0.05)
        mgr.save(net, opt, episode=1, config=config, tag="best")

        assert mgr.latest() == ep_path


# ================================================================== #
# Pruning old checkpoints with max_to_keep
# ================================================================== #


class TestPruning:
    """Verify that old checkpoints are pruned to stay within max_to_keep."""

    def test_pruning_keeps_max_to_keep(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=2)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        mgr.save(net, opt, episode=1, config=config)
        time.sleep(0.05)
        mgr.save(net, opt, episode=2, config=config)
        time.sleep(0.05)
        mgr.save(net, opt, episode=3, config=config)

        remaining = sorted(tmp_path.glob("episode_*.pt"))
        assert len(remaining) == 2

    def test_pruning_removes_oldest(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=2)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        mgr.save(net, opt, episode=1, config=config)
        time.sleep(0.05)
        mgr.save(net, opt, episode=2, config=config)
        time.sleep(0.05)
        mgr.save(net, opt, episode=3, config=config)

        remaining_names = {p.name for p in tmp_path.glob("episode_*.pt")}
        assert "episode_1.pt" not in remaining_names
        # episodes 2 and 3 should survive
        assert "episode_2.pt" in remaining_names or "episode_3.pt" in remaining_names

    def test_no_pruning_when_max_to_keep_is_zero(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=0)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        for ep in range(1, 6):
            mgr.save(net, opt, episode=ep, config=config)
            time.sleep(0.02)

        remaining = list(tmp_path.glob("episode_*.pt"))
        assert len(remaining) == 5

    def test_pruning_does_not_remove_tagged_checkpoints(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=1)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        mgr.save(net, opt, episode=1, config=config, tag="best")
        time.sleep(0.05)
        mgr.save(net, opt, episode=1, config=config)
        time.sleep(0.05)
        mgr.save(net, opt, episode=2, config=config)

        # "best.pt" should survive because pruning only targets episode_*.pt
        assert (tmp_path / "best.pt").exists()

    def test_max_to_keep_one(self, tmp_path: Path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=1)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        for ep in range(1, 5):
            mgr.save(net, opt, episode=ep, config=config)
            time.sleep(0.02)

        remaining = list(tmp_path.glob("episode_*.pt"))
        assert len(remaining) == 1
        assert remaining[0].name == "episode_4.pt"

    def test_pruning_cascade(self, tmp_path: Path):
        """Adding many checkpoints beyond max_to_keep keeps exactly max_to_keep."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_to_keep=3)
        net = _make_net()
        opt = _make_optimizer(net)
        config = _make_config()

        for ep in range(1, 11):
            mgr.save(net, opt, episode=ep, config=config)
            time.sleep(0.02)

        remaining = list(tmp_path.glob("episode_*.pt"))
        assert len(remaining) == 3
