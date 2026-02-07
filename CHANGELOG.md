# Changelog

All notable changes to the Zero-G RL Agent System are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

---

## [1.1.0] - 2026-02-07

### Added

- **Agent module** (`src/agents/`) — `AgentProtocol` (structural typing) and `ZeroGAgent` concrete implementation that owns the policy gradient update step, composes MCTS + network + optimizer, and supports skill blending
- **Skills module** (`src/skills/`) — 5 spacecraft control primitives with registry/factory pattern:
  - `TranslateToGoalSkill` — proportional thrust toward goal position
  - `AlignToGoalSkill` — quaternion-error proportional torque toward goal orientation
  - `BrakeSkill` — opposing force/torque to damp linear and angular velocity
  - `StationKeepSkill` — brake + positional correction for station-keeping
  - `ApproachSkill` — composite translate + align + distance-dependent brake blend
- **Tools module** (`src/tools/`) — 6 sensor/planning tools with registry/factory pattern:
  - `DistanceToGoalTool`, `OrientationErrorTool`, `VelocityMagnitudeTool`, `DockingProgressTool` (sensors)
  - `TrajectoryPlannerTool`, `FuelEstimatorTool` (planning)
- **MetricsStore** (`src/visualization/metrics_store.py`) — JSON file-based metrics persistence with atomic writes, integrated into Trainer and Web UI
- **MCTS policy targets** — `search()` now returns `action_probs` in info dict; Trainer stores `mcts_policy` in `Transition` for policy distillation
- **TrainingConfig fields** — `eval_episodes`, `eval_seed_offset`, `metrics_window`, `max_checkpoints` with `ZEROG_*` env var overrides
- **Module exports** — all `__init__.py` files now re-export public APIs with `__all__`
- **Named constants** — physics, MCTS, and network magic numbers extracted to module-level `_UPPERCASE` constants
- **75 new tests** across 4 new test files and 4 extended existing files:
  - `test_agent.py` — agent construction, action selection, update, mode switching, state dict
  - `test_skills.py` — all 5 skills: shape/range, direction correctness, registry
  - `test_tools.py` — all 6 tools: distance, orientation error, velocity, progress, trajectory, fuel
  - `test_metrics_store.py` — append/read, empty store, persistence, corruption handling
  - Extended `test_mcts.py` with Dirichlet noise tests
  - Extended `test_physics.py` with gyroscopic coupling tests
  - Extended `test_checkpointing.py` with corrupted checkpoint tests
  - Extended `test_instrumentation.py` with structlog capture tests
- **CHANGELOG.md** — this file

### Fixed

- **Hardcoded Linux paths** in `test_smoke.py` and `test_config.py` replaced with dynamic `Path(__file__).resolve().parents[N]`
- **Bare `except Exception`** in `src/main.py` replaced with specific types (`FileNotFoundError`, `ValueError`, `RuntimeError`, `KeyboardInterrupt`)
- **Windows `Path.rename()` failure** in `checkpoint_manager.py` — changed to `Path.replace()` for cross-platform atomic overwrites
- **Symlink test failure on Windows** — added `@pytest.mark.skipif(os.name == "nt")` guard
- **Web UI placeholder data** — now reads real metrics from `MetricsStore` with fallback to dummy data
- **`pyproject.toml` build backend** — changed from `setuptools.backends._legacy:_Backend` to `setuptools.build_meta`

### Changed

- Trainer now uses `TrainingConfig` fields instead of hardcoded values for eval episodes, seed offsets, metrics window, and max checkpoints
- MCTS `search()` return type widened to `dict[str, object]` to accommodate `action_probs` ndarray
- All source and test files formatted with `ruff format`
- `.gitignore` updated with `.claude/` directory

---

## [1.0.0] - 2026-02-07

### Added

- Initial implementation of Zero-G Spatial Awareness RL Agent System
- AlphaZero-style self-play training loop with MCTS
- 3D CNN + MLP dual-headed policy/value network
- Symplectic Euler zero-gravity physics with momentum conservation validation
- Pydantic v2 configuration with YAML loading and env var overrides
- Gymnasium-compatible environment abstraction with mock simulator
- FIFO replay buffer with Parquet persistence
- Versioned checkpoint manager with schema migrations
- Gradio web dashboard with Plotly 3D trajectories
- structlog structured logging with debug instrumentation
- Comprehensive test suite (462 tests)
