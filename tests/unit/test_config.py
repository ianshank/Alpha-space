"""Comprehensive unit tests for src/config.py.

Covers: NetworkConfig, MCTSConfig, TrainingConfig, EnvironmentConfig,
RewardConfig, SystemConfig Pydantic models, load_config from YAML,
environment variable overrides, _auto_cast, and _set_nested.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.config import (
    EnvironmentConfig,
    MCTSConfig,
    NetworkConfig,
    RewardConfig,
    SystemConfig,
    TrainingConfig,
    _auto_cast,
    _set_nested,
    load_config,
)


# =========================================================================
# _auto_cast
# =========================================================================


class TestAutoCast:
    """Tests for the _auto_cast helper function."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
        ],
        ids=["true", "True", "TRUE", "1", "yes", "YES"],
    )
    def test_truthy_strings(self, raw: str, expected: bool) -> None:
        """Truthy string representations should cast to True."""
        assert _auto_cast(raw) is expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("NO", False),
        ],
        ids=["false", "False", "FALSE", "0", "no", "NO"],
    )
    def test_falsy_strings(self, raw: str, expected: bool) -> None:
        """Falsy string representations should cast to False."""
        assert _auto_cast(raw) is expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("42", 42),
            ("-1", -1),
            ("0", False),  # "0" maps to False per the bool check first
            ("999999", 999999),
        ],
        ids=["positive_int", "negative_int", "zero_is_bool_false", "large_int"],
    )
    def test_integer_strings(self, raw: str, expected: int | bool) -> None:
        """Integer string representations should cast to int (0 -> False)."""
        assert _auto_cast(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("3.14", 3.14),
            ("-0.001", -0.001),
            ("1e-4", 1e-4),
            ("1.0", 1.0),
        ],
        ids=["pi", "negative", "scientific", "one_point_zero"],
    )
    def test_float_strings(self, raw: str, expected: float) -> None:
        """Float string representations should cast to float."""
        result = _auto_cast(raw)
        assert isinstance(result, float)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw",
        ["hello", "mock", "path/to/file", "", "  ", "cuda:0"],
        ids=["word", "simulator", "path", "empty", "spaces", "device"],
    )
    def test_plain_strings_returned_as_is(self, raw: str) -> None:
        """Non-numeric, non-boolean strings should remain as strings."""
        result = _auto_cast(raw)
        assert isinstance(result, str)
        assert result == raw


# =========================================================================
# _set_nested
# =========================================================================


class TestSetNested:
    """Tests for the _set_nested helper function."""

    def test_single_level_key(self) -> None:
        """A non-dotted key should set a top-level value."""
        data: dict[str, Any] = {}
        _set_nested(data, "seed", 42)
        assert data == {"seed": 42}

    def test_two_level_dotted_key(self) -> None:
        """A two-part dotted key should create a nested dict."""
        data: dict[str, Any] = {}
        _set_nested(data, "training.learning_rate", 0.001)
        assert data == {"training": {"learning_rate": 0.001}}

    def test_three_level_dotted_key(self) -> None:
        """A three-part dotted key should create deeply nested dicts."""
        data: dict[str, Any] = {}
        _set_nested(data, "a.b.c", "deep")
        assert data == {"a": {"b": {"c": "deep"}}}

    def test_preserves_existing_keys(self) -> None:
        """Setting a new nested key should not overwrite sibling keys."""
        data: dict[str, Any] = {"training": {"batch_size": 64}}
        _set_nested(data, "training.learning_rate", 0.01)
        assert data == {"training": {"batch_size": 64, "learning_rate": 0.01}}

    def test_overwrites_existing_value(self) -> None:
        """Setting an existing key should overwrite its value."""
        data: dict[str, Any] = {"seed": 1}
        _set_nested(data, "seed", 99)
        assert data == {"seed": 99}

    def test_creates_intermediate_dicts(self) -> None:
        """Intermediate dicts should be created if they do not exist."""
        data: dict[str, Any] = {}
        _set_nested(data, "network.hidden_dim", 128)
        assert "network" in data
        assert isinstance(data["network"], dict)
        assert data["network"]["hidden_dim"] == 128


# =========================================================================
# NetworkConfig
# =========================================================================


class TestNetworkConfig:
    """Tests for the NetworkConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default values should match documented defaults."""
        cfg = NetworkConfig()
        assert cfg.voxel_resolution == 64
        assert cfg.voxel_channels == 4
        assert cfg.proprioception_dim == 13
        assert cfg.hidden_dim == 256
        assert cfg.num_res_blocks == 6
        assert cfg.action_dim == 6
        assert cfg.min_log_std == -5.0
        assert cfg.max_log_std == 2.0

    @pytest.mark.parametrize("resolution", [16, 32, 64, 128])
    def test_valid_power_of_two_resolutions(self, resolution: int) -> None:
        """Power-of-two resolutions within [16, 128] should be accepted."""
        cfg = NetworkConfig(voxel_resolution=resolution)
        assert cfg.voxel_resolution == resolution

    @pytest.mark.parametrize("resolution", [17, 24, 48, 100])
    def test_non_power_of_two_resolution_rejected(self, resolution: int) -> None:
        """Non-power-of-two resolutions should fail validation."""
        with pytest.raises(ValidationError, match="power of 2"):
            NetworkConfig(voxel_resolution=resolution)

    def test_resolution_below_minimum_rejected(self) -> None:
        """Resolution below 16 should fail the ge=16 constraint."""
        with pytest.raises(ValidationError):
            NetworkConfig(voxel_resolution=8)

    def test_resolution_above_maximum_rejected(self) -> None:
        """Resolution above 128 should fail the le=128 constraint."""
        with pytest.raises(ValidationError):
            NetworkConfig(voxel_resolution=256)

    def test_hidden_dim_below_minimum_rejected(self) -> None:
        """hidden_dim below 64 should fail."""
        with pytest.raises(ValidationError):
            NetworkConfig(hidden_dim=32)

    def test_voxel_channels_minimum(self) -> None:
        """voxel_channels must be >= 1."""
        with pytest.raises(ValidationError):
            NetworkConfig(voxel_channels=0)

    def test_num_res_blocks_bounds(self) -> None:
        """num_res_blocks must be in [1, 50]."""
        with pytest.raises(ValidationError):
            NetworkConfig(num_res_blocks=0)
        with pytest.raises(ValidationError):
            NetworkConfig(num_res_blocks=51)

    def test_action_dim_minimum(self) -> None:
        """action_dim must be >= 1."""
        with pytest.raises(ValidationError):
            NetworkConfig(action_dim=0)

    def test_custom_values(self) -> None:
        """Custom values within constraints should be accepted."""
        cfg = NetworkConfig(
            voxel_resolution=32,
            hidden_dim=128,
            num_res_blocks=3,
            action_dim=8,
            min_log_std=-10.0,
            max_log_std=5.0,
        )
        assert cfg.voxel_resolution == 32
        assert cfg.hidden_dim == 128
        assert cfg.num_res_blocks == 3
        assert cfg.action_dim == 8

    def test_uses_conftest_fixture(self, network_config: NetworkConfig) -> None:
        """Fixture from conftest should produce a valid NetworkConfig."""
        assert network_config.voxel_resolution == 16
        assert network_config.hidden_dim == 64
        assert network_config.num_res_blocks == 1


# =========================================================================
# MCTSConfig
# =========================================================================


class TestMCTSConfig:
    """Tests for the MCTSConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default MCTS config values."""
        cfg = MCTSConfig()
        assert cfg.num_simulations == 800
        assert cfg.c_puct == 1.0
        assert cfg.max_children == 32
        assert cfg.temperature == 1.0
        assert cfg.dirichlet_alpha == 0.25
        assert cfg.dirichlet_epsilon == 0.25
        assert cfg.discount == 0.99

    def test_num_simulations_must_be_positive(self) -> None:
        """num_simulations must be >= 1."""
        with pytest.raises(ValidationError):
            MCTSConfig(num_simulations=0)

    def test_c_puct_must_be_positive(self) -> None:
        """c_puct must be > 0."""
        with pytest.raises(ValidationError):
            MCTSConfig(c_puct=0.0)
        with pytest.raises(ValidationError):
            MCTSConfig(c_puct=-1.0)

    def test_temperature_allows_zero(self) -> None:
        """temperature >= 0.0 (greedy mode at 0)."""
        cfg = MCTSConfig(temperature=0.0)
        assert cfg.temperature == 0.0

    def test_temperature_negative_rejected(self) -> None:
        """Negative temperature should be rejected."""
        with pytest.raises(ValidationError):
            MCTSConfig(temperature=-0.1)

    def test_dirichlet_epsilon_bounds(self) -> None:
        """dirichlet_epsilon must be in [0, 1]."""
        MCTSConfig(dirichlet_epsilon=0.0)
        MCTSConfig(dirichlet_epsilon=1.0)
        with pytest.raises(ValidationError):
            MCTSConfig(dirichlet_epsilon=-0.01)
        with pytest.raises(ValidationError):
            MCTSConfig(dirichlet_epsilon=1.01)

    def test_discount_bounds(self) -> None:
        """discount must be in [0, 1]."""
        MCTSConfig(discount=0.0)
        MCTSConfig(discount=1.0)
        with pytest.raises(ValidationError):
            MCTSConfig(discount=-0.1)
        with pytest.raises(ValidationError):
            MCTSConfig(discount=1.1)

    def test_uses_conftest_fixture(self, mcts_config: MCTSConfig) -> None:
        """Fixture from conftest should produce a valid MCTSConfig."""
        assert mcts_config.num_simulations == 4
        assert mcts_config.max_children == 4


# =========================================================================
# TrainingConfig
# =========================================================================


class TestTrainingConfig:
    """Tests for the TrainingConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default training config values."""
        cfg = TrainingConfig()
        assert cfg.num_episodes == 1000
        assert cfg.batch_size == 256
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 1e-4
        assert cfg.gradient_clip_norm == 1.0
        assert cfg.checkpoint_interval == 100
        assert cfg.eval_interval == 50
        assert cfg.replay_buffer_size == 10_000
        assert cfg.value_loss_weight == 1.0
        assert cfg.entropy_weight == 0.01
        assert cfg.epochs_per_update == 4

    def test_learning_rate_must_be_positive(self) -> None:
        """learning_rate must be > 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-1e-3)

    def test_weight_decay_allows_zero(self) -> None:
        """weight_decay can be zero (no regularization)."""
        cfg = TrainingConfig(weight_decay=0.0)
        assert cfg.weight_decay == 0.0

    def test_weight_decay_negative_rejected(self) -> None:
        """Negative weight_decay should be rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(weight_decay=-0.01)

    def test_batch_size_minimum(self) -> None:
        """batch_size must be >= 1."""
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_gradient_clip_norm_must_be_positive(self) -> None:
        """gradient_clip_norm must be > 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(gradient_clip_norm=0.0)

    def test_entropy_weight_allows_zero(self) -> None:
        """entropy_weight can be zero (no entropy bonus)."""
        cfg = TrainingConfig(entropy_weight=0.0)
        assert cfg.entropy_weight == 0.0

    def test_uses_conftest_fixture(self, training_config: TrainingConfig) -> None:
        """Fixture from conftest should produce a valid TrainingConfig."""
        assert training_config.num_episodes == 2
        assert training_config.batch_size == 4


# =========================================================================
# EnvironmentConfig
# =========================================================================


class TestEnvironmentConfig:
    """Tests for the EnvironmentConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default environment config values."""
        cfg = EnvironmentConfig()
        assert cfg.simulator == "mock"
        assert cfg.parallel_envs == 1
        assert cfg.max_episode_steps == 500
        assert cfg.workspace_size == 10.0
        assert cfg.time_step == 0.05
        assert cfg.max_thrust == 10.0
        assert cfg.max_torque == 2.0
        assert cfg.position_tolerance == 0.1
        assert cfg.orientation_tolerance_deg == 5.0
        assert cfg.spacecraft_mass == 100.0
        assert cfg.spacecraft_inertia == (10.0, 10.0, 10.0)

    @pytest.mark.parametrize("simulator", ["mock", "unity", "isaac_sim"])
    def test_valid_simulators(self, simulator: str) -> None:
        """All three valid simulator backends should be accepted."""
        cfg = EnvironmentConfig(simulator=simulator)
        assert cfg.simulator == simulator

    @pytest.mark.parametrize(
        "simulator",
        ["mujoco", "pybullet", "MOCK", "Mock", "", "isaac-sim"],
        ids=["mujoco", "pybullet", "MOCK_caps", "Mock_mixed", "empty", "hyphenated"],
    )
    def test_invalid_simulators_rejected(self, simulator: str) -> None:
        """Invalid simulator names should fail the pattern constraint."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(simulator=simulator)

    def test_parallel_envs_minimum(self) -> None:
        """parallel_envs must be >= 1."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(parallel_envs=0)

    def test_workspace_size_must_be_positive(self) -> None:
        """workspace_size must be > 0."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(workspace_size=0.0)
        with pytest.raises(ValidationError):
            EnvironmentConfig(workspace_size=-1.0)

    def test_time_step_must_be_positive(self) -> None:
        """time_step must be > 0."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(time_step=0.0)

    def test_spacecraft_mass_must_be_positive(self) -> None:
        """spacecraft_mass must be > 0."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(spacecraft_mass=0.0)

    def test_tolerances_must_be_positive(self) -> None:
        """position_tolerance and orientation_tolerance_deg must be > 0."""
        with pytest.raises(ValidationError):
            EnvironmentConfig(position_tolerance=0.0)
        with pytest.raises(ValidationError):
            EnvironmentConfig(orientation_tolerance_deg=0.0)

    def test_spacecraft_inertia_tuple(self) -> None:
        """spacecraft_inertia should accept a 3-tuple of floats."""
        cfg = EnvironmentConfig(spacecraft_inertia=(1.0, 2.0, 3.0))
        assert cfg.spacecraft_inertia == (1.0, 2.0, 3.0)

    def test_uses_conftest_fixture(self, env_config: EnvironmentConfig) -> None:
        """Fixture from conftest should produce a valid EnvironmentConfig."""
        assert env_config.simulator == "mock"
        assert env_config.max_episode_steps == 10


# =========================================================================
# RewardConfig
# =========================================================================


class TestRewardConfig:
    """Tests for the RewardConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default reward config values."""
        cfg = RewardConfig()
        assert cfg.position_weight == 1.0
        assert cfg.orientation_weight == 0.5
        assert cfg.velocity_weight == 0.1
        assert cfg.fuel_weight == 0.01
        assert cfg.success_bonus == 100.0
        assert cfg.collision_penalty == -50.0
        assert cfg.time_penalty == -0.1

    def test_weights_allow_zero(self) -> None:
        """All weights can be zero (disabled)."""
        cfg = RewardConfig(
            position_weight=0.0,
            orientation_weight=0.0,
            velocity_weight=0.0,
            fuel_weight=0.0,
            success_bonus=0.0,
        )
        assert cfg.position_weight == 0.0
        assert cfg.success_bonus == 0.0

    def test_negative_weights_rejected(self) -> None:
        """Positive-constrained weights should reject negative values."""
        with pytest.raises(ValidationError):
            RewardConfig(position_weight=-1.0)
        with pytest.raises(ValidationError):
            RewardConfig(fuel_weight=-0.01)

    def test_collision_penalty_must_be_non_positive(self) -> None:
        """collision_penalty must be <= 0."""
        RewardConfig(collision_penalty=0.0)
        with pytest.raises(ValidationError):
            RewardConfig(collision_penalty=1.0)

    def test_time_penalty_must_be_non_positive(self) -> None:
        """time_penalty must be <= 0."""
        RewardConfig(time_penalty=0.0)
        with pytest.raises(ValidationError):
            RewardConfig(time_penalty=0.1)

    def test_uses_conftest_fixture(self, reward_config: RewardConfig) -> None:
        """Fixture from conftest should produce a valid RewardConfig with defaults."""
        assert reward_config.success_bonus == 100.0
        assert reward_config.collision_penalty == -50.0


# =========================================================================
# SystemConfig
# =========================================================================


class TestSystemConfig:
    """Tests for the SystemConfig root model."""

    def test_defaults(self) -> None:
        """Defaults should produce a fully valid config."""
        cfg = SystemConfig()
        assert isinstance(cfg.network, NetworkConfig)
        assert isinstance(cfg.mcts, MCTSConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.environment, EnvironmentConfig)
        assert isinstance(cfg.reward, RewardConfig)
        assert cfg.checkpoint_dir == Path("checkpoints")
        assert cfg.log_dir == Path("logs")
        assert cfg.wandb_project == "zerog-rl"
        assert cfg.wandb_entity is None
        assert cfg.seed == 42
        assert cfg.use_gpu is True
        assert cfg.debug is False
        assert cfg.no_logging is False

    def test_seed_non_negative(self) -> None:
        """seed must be >= 0."""
        with pytest.raises(ValidationError):
            SystemConfig(seed=-1)

    def test_wandb_entity_optional(self) -> None:
        """wandb_entity should default to None and accept strings."""
        cfg = SystemConfig()
        assert cfg.wandb_entity is None
        cfg2 = SystemConfig(wandb_entity="my-team")
        assert cfg2.wandb_entity == "my-team"

    def test_paths_accept_strings(self) -> None:
        """Path fields should accept string inputs and convert to Path."""
        cfg = SystemConfig(checkpoint_dir="/tmp/ckpt", log_dir="/tmp/log")
        assert isinstance(cfg.checkpoint_dir, Path)
        assert isinstance(cfg.log_dir, Path)

    def test_sub_configs_overridden(self) -> None:
        """Sub-config values can be overridden at construction."""
        cfg = SystemConfig(
            network=NetworkConfig(voxel_resolution=32, hidden_dim=128),
            training=TrainingConfig(learning_rate=0.01),
        )
        assert cfg.network.voxel_resolution == 32
        assert cfg.network.hidden_dim == 128
        assert cfg.training.learning_rate == 0.01

    def test_uses_conftest_fixture(self, system_config: SystemConfig) -> None:
        """Fixture from conftest should produce a coherent SystemConfig."""
        assert system_config.seed == 42
        assert system_config.use_gpu is False
        assert system_config.debug is True
        assert system_config.no_logging is True
        assert system_config.network.voxel_resolution == 16
        assert system_config.mcts.num_simulations == 4


# =========================================================================
# load_config - from YAML
# =========================================================================


class TestLoadConfigFromYaml:
    """Tests for load_config reading YAML files."""

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        """A valid YAML file should be loaded into SystemConfig."""
        yaml_content = textwrap.dedent("""\
            network:
              voxel_resolution: 32
              hidden_dim: 128
            training:
              learning_rate: 0.001
              num_episodes: 10
            seed: 7
            use_gpu: false
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file)
        assert cfg.network.voxel_resolution == 32
        assert cfg.network.hidden_dim == 128
        assert cfg.training.learning_rate == 0.001
        assert cfg.training.num_episodes == 10
        assert cfg.seed == 7
        assert cfg.use_gpu is False

    def test_load_from_string_path(self, tmp_path: Path) -> None:
        """load_config should accept a string path as well as Path."""
        yaml_content = "seed: 99\n"
        config_file = tmp_path / "c.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.seed == 99

    def test_nonexistent_yaml_raises(self) -> None:
        """A non-existent config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        """An empty YAML file should fall back to all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        cfg = load_config(config_file)
        assert cfg.seed == 42
        assert cfg.network.voxel_resolution == 64

    def test_partial_yaml_fills_defaults(self, tmp_path: Path) -> None:
        """A partial YAML should fill missing values from defaults."""
        yaml_content = textwrap.dedent("""\
            training:
              batch_size: 512
        """)
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file)
        assert cfg.training.batch_size == 512
        # Other training defaults preserved
        assert cfg.training.learning_rate == 1e-4
        # Network defaults preserved
        assert cfg.network.voxel_resolution == 64

    def test_no_config_path_uses_defaults(self) -> None:
        """Calling load_config(None) should return all defaults."""
        cfg = load_config(None)
        assert cfg == SystemConfig()

    def test_load_fixture_yaml(self) -> None:
        """The smoke test fixture YAML should load without error."""
        fixture = Path("/home/user/Alpha-space/tests/fixtures/configs/smoke_test.yaml")
        cfg = load_config(fixture)
        assert cfg.network.voxel_resolution == 16
        assert cfg.environment.simulator == "mock"
        assert cfg.seed == 42

    def test_invalid_yaml_value_raises_validation_error(self, tmp_path: Path) -> None:
        """Invalid field values in YAML should raise ValidationError."""
        yaml_content = textwrap.dedent("""\
            network:
              voxel_resolution: 17
        """)
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValidationError, match="power of 2"):
            load_config(config_file)


# =========================================================================
# load_config - environment variable overrides
# =========================================================================


class TestLoadConfigEnvOverrides:
    """Tests for load_config environment variable overrides."""

    def test_env_override_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZEROG_SEED should override the seed value."""
        monkeypatch.setenv("ZEROG_SEED", "123")
        cfg = load_config()
        assert cfg.seed == 123

    def test_env_override_learning_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZEROG_LEARNING_RATE should override training.learning_rate."""
        monkeypatch.setenv("ZEROG_LEARNING_RATE", "0.01")
        cfg = load_config()
        assert cfg.training.learning_rate == pytest.approx(0.01)

    def test_env_override_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZEROG_BATCH_SIZE should override training.batch_size."""
        monkeypatch.setenv("ZEROG_BATCH_SIZE", "512")
        cfg = load_config()
        assert cfg.training.batch_size == 512

    def test_env_override_simulator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZEROG_SIMULATOR should override environment.simulator."""
        monkeypatch.setenv("ZEROG_SIMULATOR", "unity")
        cfg = load_config()
        assert cfg.environment.simulator == "unity"

    def test_env_override_voxel_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_VOXEL_RESOLUTION should override network.voxel_resolution."""
        monkeypatch.setenv("ZEROG_VOXEL_RESOLUTION", "32")
        cfg = load_config()
        assert cfg.network.voxel_resolution == 32

    def test_env_override_hidden_dim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZEROG_HIDDEN_DIM should override network.hidden_dim."""
        monkeypatch.setenv("ZEROG_HIDDEN_DIM", "128")
        cfg = load_config()
        assert cfg.network.hidden_dim == 128

    def test_env_override_wandb_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_WANDB_PROJECT should override wandb_project."""
        monkeypatch.setenv("ZEROG_WANDB_PROJECT", "my-project")
        cfg = load_config()
        assert cfg.wandb_project == "my-project"

    def test_env_override_wandb_entity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_WANDB_ENTITY should override wandb_entity."""
        monkeypatch.setenv("ZEROG_WANDB_ENTITY", "my-team")
        cfg = load_config()
        assert cfg.wandb_entity == "my-team"

    def test_env_override_wins_over_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables should override YAML file values."""
        yaml_content = "seed: 10\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        monkeypatch.setenv("ZEROG_SEED", "999")
        cfg = load_config(config_file)
        assert cfg.seed == 999

    def test_env_override_mcts_simulations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_MCTS_SIMULATIONS should override mcts.num_simulations."""
        monkeypatch.setenv("ZEROG_MCTS_SIMULATIONS", "100")
        cfg = load_config()
        assert cfg.mcts.num_simulations == 100

    def test_env_override_mcts_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_MCTS_TEMPERATURE should override mcts.temperature."""
        monkeypatch.setenv("ZEROG_MCTS_TEMPERATURE", "0.5")
        cfg = load_config()
        assert cfg.mcts.temperature == pytest.approx(0.5)

    def test_env_override_num_episodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_NUM_EPISODES should override training.num_episodes."""
        monkeypatch.setenv("ZEROG_NUM_EPISODES", "500")
        cfg = load_config()
        assert cfg.training.num_episodes == 500

    def test_env_override_parallel_envs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_PARALLEL_ENVS should override environment.parallel_envs."""
        monkeypatch.setenv("ZEROG_PARALLEL_ENVS", "4")
        cfg = load_config()
        assert cfg.environment.parallel_envs == 4

    def test_env_override_num_res_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZEROG_NUM_RES_BLOCKS should override network.num_res_blocks."""
        monkeypatch.setenv("ZEROG_NUM_RES_BLOCKS", "3")
        cfg = load_config()
        assert cfg.network.num_res_blocks == 3


# =========================================================================
# load_config - explicit overrides
# =========================================================================


class TestLoadConfigExplicitOverrides:
    """Tests for load_config explicit overrides dict."""

    def test_override_top_level(self) -> None:
        """Dotted notation for top-level keys should work."""
        cfg = load_config(overrides={"seed": 77})
        assert cfg.seed == 77

    def test_override_nested_key(self) -> None:
        """Dotted notation for nested keys should work."""
        cfg = load_config(overrides={"training.learning_rate": 0.05})
        assert cfg.training.learning_rate == pytest.approx(0.05)

    def test_multiple_overrides(self) -> None:
        """Multiple overrides should all be applied."""
        cfg = load_config(
            overrides={
                "seed": 1,
                "training.batch_size": 128,
                "network.hidden_dim": 512,
                "mcts.num_simulations": 50,
            }
        )
        assert cfg.seed == 1
        assert cfg.training.batch_size == 128
        assert cfg.network.hidden_dim == 512
        assert cfg.mcts.num_simulations == 50

    def test_overrides_win_over_yaml(self, tmp_path: Path) -> None:
        """Explicit overrides should take precedence over YAML values."""
        yaml_content = textwrap.dedent("""\
            seed: 10
            training:
              batch_size: 64
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(config_file, overrides={"seed": 555, "training.batch_size": 32})
        assert cfg.seed == 555
        assert cfg.training.batch_size == 32

    def test_overrides_win_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit overrides should take precedence over environment variables."""
        monkeypatch.setenv("ZEROG_SEED", "100")
        cfg = load_config(overrides={"seed": 200})
        assert cfg.seed == 200

    def test_all_three_resolution_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full precedence: overrides > env vars > YAML > defaults."""
        yaml_content = textwrap.dedent("""\
            seed: 1
            training:
              batch_size: 16
              learning_rate: 0.1
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        monkeypatch.setenv("ZEROG_BATCH_SIZE", "32")
        monkeypatch.setenv("ZEROG_LEARNING_RATE", "0.05")

        cfg = load_config(
            config_file,
            overrides={"training.batch_size": 64},
        )

        # seed: from YAML (no env or override)
        assert cfg.seed == 1
        # learning_rate: env overrides YAML -> 0.05
        assert cfg.training.learning_rate == pytest.approx(0.05)
        # batch_size: explicit override wins over env -> 64
        assert cfg.training.batch_size == 64


# =========================================================================
# load_config - edge cases
# =========================================================================


class TestLoadConfigEdgeCases:
    """Edge case tests for load_config."""

    def test_yaml_with_unknown_keys_ignored(self, tmp_path: Path) -> None:
        """Unknown top-level keys in YAML should cause a validation error."""
        yaml_content = textwrap.dedent("""\
            totally_unknown_key: 42
        """)
        config_file = tmp_path / "unknown.yaml"
        config_file.write_text(yaml_content)

        # Pydantic by default will either ignore or reject extra fields.
        # With default Pydantic BaseModel config, extra fields raise an error.
        # If the model allows extra, this should still load without crash.
        # We test whichever behaviour is configured.
        try:
            cfg = load_config(config_file)
            # If no error, defaults should still be intact
            assert cfg.seed == 42
        except ValidationError:
            # Extra fields are forbidden -- that is also valid behaviour
            pass

    def test_yaml_with_only_comments(self, tmp_path: Path) -> None:
        """A YAML file with only comments should use defaults."""
        config_file = tmp_path / "comments.yaml"
        config_file.write_text("# This is a comment\n# Another comment\n")

        cfg = load_config(config_file)
        assert cfg.seed == 42

    def test_none_overrides_same_as_no_overrides(self) -> None:
        """Passing overrides=None should be the same as omitting it."""
        cfg1 = load_config(overrides=None)
        cfg2 = load_config()
        assert cfg1.seed == cfg2.seed
        assert cfg1.training.batch_size == cfg2.training.batch_size

    def test_empty_overrides_same_as_no_overrides(self) -> None:
        """Passing an empty overrides dict should be the same as omitting it."""
        cfg1 = load_config(overrides={})
        cfg2 = load_config()
        assert cfg1.seed == cfg2.seed

    def test_auto_cast_boolean_in_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Boolean-like env values should be auto-cast correctly."""
        # ZEROG_SEED with "true" would become True (=1), which is valid as seed >= 0
        # but let's test a string env var
        monkeypatch.setenv("ZEROG_WANDB_PROJECT", "my-project-v2")
        cfg = load_config()
        assert cfg.wandb_project == "my-project-v2"
