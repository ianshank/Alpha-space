# Zero-G Spatial Awareness RL Agent System

An AlphaZero-style self-play reinforcement learning system for training agents to navigate and dock in 3D zero-gravity environments with 6-DOF continuous control.

| Metric | Value |
|--------|-------|
| Tests | 462 passing |
| Coverage | 87% |
| Python | 3.11+ |
| PyTorch | 2.0+ |
| Type Checking | mypy --strict |
| Linting | ruff clean |

---

## Overview

This system extends AlphaZero from discrete 2D board games to **continuous 3D zero-gravity environments**. An agent learns to control a spacecraft with 6 degrees of freedom (3D thrust + 3D torque) to dock at a target pose, using:

- **Monte Carlo Tree Search (MCTS)** with progressive widening for continuous action spaces
- **Self-play** episode generation with MCTS-improved policy targets
- **Dual-headed neural network** (3D CNN + MLP) for policy and value estimation
- **Symplectic Euler** physics integration with momentum conservation validation
- **Quaternion-based** orientation (Hamilton convention, `[w,x,y,z]`) to avoid gimbal lock

### Key Design Principles

- **No hardcoded values** — all hyperparameters flow through Pydantic v2 config models
- **Backward-compatible checkpoints** — versioned schema with automatic migrations
- **Pluggable simulators** — abstract Gymnasium interface with `register_env()` factory
- **Production logging** — structured JSON via structlog, debug instrumentation decorators
- **Comprehensive testing** — unit, integration, property-based (Hypothesis), and smoke tests

---

## Quick Start

### Installation

```bash
# Clone and set up
git clone <repo-url> && cd Alpha-space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env  # edit with your API keys
```

### Training

```bash
# Smoke test — 1 episode, CPU, smallest config (~5s)
python -m src.main train --config configs/smoke_test.yaml

# Quick test — 10 episodes, CPU (~30s)
python -m src.main train --config configs/quick_test.yaml

# Full training — 1000 episodes, GPU
python -m src.main train --config configs/docking_task.yaml

# Resume from checkpoint
python -m src.main train --config configs/docking_task.yaml \
    --resume checkpoints/episode_500.pt

# Override episode count
python -m src.main train --config configs/smoke_test.yaml --episodes 5
```

### Evaluation

```bash
# Evaluate checkpoint with deterministic (greedy) actions
python -m src.main evaluate --checkpoint checkpoints/final.pt --episodes 100

# Evaluate with MCTS (for comparison)
python -m src.main evaluate --checkpoint checkpoints/final.pt \
    --episodes 100 --use-mcts
```

### Web UI

```bash
python -m src.visualization.web_ui --port 7860
```

---

## Testing

```bash
# Full suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Unit tests only (fast, ~8s)
pytest tests/unit/ -v

# Integration tests (~15s)
pytest tests/integration/ -v

# Property tests — Hypothesis (~5s)
pytest tests/properties/ -v

# With coverage threshold enforcement
pytest tests/ --cov=src --cov-fail-under=85
```

### Test Structure

```
tests/
├── conftest.py                          # Shared fixtures (configs, sample data)
├── unit/                                # Fast, isolated tests
│   ├── test_config.py                   # Pydantic config validation
│   ├── test_utils.py                    # Quaternion math, seeding, timer
│   ├── test_networks.py                 # Policy/value network shapes
│   ├── test_mcts.py                     # MCTS tree operations
│   ├── test_physics.py                  # Zero-G dynamics, momentum
│   ├── test_replay_buffer.py            # FIFO buffer, batch sampling
│   ├── test_checkpointing.py            # Versioned save/load/migrate
│   └── test_instrumentation.py          # Debug decorators
├── integration/                         # Multi-component tests
│   ├── test_training_loop.py            # End-to-end training
│   ├── test_evaluator.py               # Checkpoint evaluation
│   └── test_smoke.py                    # CLI subprocess smoke test
├── properties/                          # Hypothesis property tests
│   ├── test_physics_invariants.py       # Momentum conservation, quaternion norms
│   └── test_mcts_properties.py          # Tree structure invariants
└── fixtures/
    └── configs/smoke_test.yaml          # Test-specific config
```

---

## Code Quality

```bash
# Lint
ruff check src/ tests/

# Type check (strict mode)
mypy src/ --strict --allow-untyped-calls --allow-untyped-decorators --ignore-missing-imports

# Format (optional)
black src/ tests/ --line-length 100
isort src/ tests/ --profile black
```

---

## Configuration

All configuration is driven by **Pydantic v2 models** in `src/config.py`. No hardcoded values exist in application code.

### Config Hierarchy

```
YAML file → Environment variables (ZEROG_ prefix) → Explicit overrides dict
```

### Example: Override via Environment Variables

```bash
ZEROG_LEARNING_RATE=0.001 ZEROG_BATCH_SIZE=128 \
    python -m src.main train --config configs/docking_task.yaml
```

### Config Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `NetworkConfig` | Neural network architecture | `voxel_resolution`, `hidden_dim`, `num_res_blocks`, `action_dim` |
| `MCTSConfig` | Tree search hyperparams | `num_simulations`, `c_puct`, `max_children`, `temperature` |
| `TrainingConfig` | Training loop settings | `num_episodes`, `batch_size`, `learning_rate`, `gradient_clip_norm` |
| `EnvironmentConfig` | Simulation settings | `simulator`, `max_episode_steps`, `max_thrust`, `max_torque` |
| `RewardConfig` | Reward shaping weights | `position_weight`, `orientation_weight`, `success_bonus` |
| `SystemConfig` | Top-level aggregator | All above + `checkpoint_dir`, `seed`, `use_gpu` |

### Available Configs

| File | Purpose |
|------|---------|
| `configs/smoke_test.yaml` | 1 episode, 16^3 voxels, 4 MCTS sims — CI/quick sanity |
| `configs/quick_test.yaml` | 10 episodes, 16^3 voxels, 8 MCTS sims — integration testing |
| `configs/docking_task.yaml` | 1000 episodes, 64^3 voxels, 800 MCTS sims — production training |

---

## Project Structure

```
src/
├── main.py                          # CLI entry point (train / evaluate)
├── config.py                        # Pydantic v2 config models + YAML loader
├── logging_config.py                # Structlog setup (console / JSON)
│
├── networks/
│   └── policy_value_net.py          # 3D CNN + MLP dual-headed network
│
├── mcts/
│   └── engine.py                    # Continuous-action MCTS with progressive widening
│
├── physics/
│   └── zero_g_dynamics.py           # Symplectic Euler integrator, momentum validation
│
├── environments/
│   ├── base.py                      # Abstract Gymnasium environment
│   ├── mock_env.py                  # Built-in physics-backed mock simulator
│   └── factory.py                   # make_env() / register_env()
│
├── training/
│   └── trainer.py                   # Self-play + MCTS + policy gradient loop
│
├── evaluation/
│   └── evaluator.py                 # Standalone checkpoint evaluation
│
├── replay_buffer/
│   └── buffer.py                    # FIFO buffer + Parquet persistence
│
├── checkpointing/
│   └── checkpoint_manager.py        # Versioned save/load with migrations
│
├── debugging/
│   └── instrumentation.py           # Timing, memory, GPU, tensor stats decorators
│
├── visualization/
│   └── web_ui.py                    # Gradio dashboard with Plotly 3D trajectories
│
└── utils/
    └── common.py                    # Seeding, quaternions, Timer, path validation
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full C4 diagrams (Context, Container, Component, Code levels).

### Training Loop (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Play Episode                        │
│                                                             │
│   Environment ──obs──▶ Policy Net ──actions──▶ MCTS         │
│       │                                         │           │
│       │◀────────────── improved action ─────────┘           │
│       │                                                     │
│       ▼                                                     │
│   (obs, action, reward, mcts_policy) ──▶ Replay Buffer      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Policy Update                            │
│                                                             │
│   Replay Buffer ──batch──▶ Policy Net ──loss──▶ Optimizer   │
│                                                             │
│   Loss = -log_prob * (return - V) + MSE(V, target) - H     │
└─────────────────────────────────────────────────────────────┘
```

### Neural Network Architecture

```
Voxel Grid (B, 4, D, D, D)     Proprioception (B, 13)
        │                              │
   VoxelEncoder                 ProprioEncoder
   (3D CNN → AdaptivePool)     (MLP 13→H→H)
        │                              │
        └──────── concat ──────────────┘
                    │
              Fusion (2H → H)
                    │
             SharedTrunk (N × ResBlock1D)
                    │
           ┌────────┴────────┐
      PolicyHead          ValueHead
   (H → mean, log_std)    (H → H/2 → 1)
           │                    │
   Normal(mean, std)        scalar V(s)
```

---

## Extending the System

### Register a New Simulator Backend

```python
from src.environments.base import ZeroGEnv
from src.environments.factory import register_env

class MyCustomEnv(ZeroGEnv):
    def _sim_reset(self, seed=None, options=None):
        # Initialize your simulator
        return observation_dict

    def _sim_step(self, action):
        # Step your simulator
        return observation_dict, info_dict

register_env("my_simulator", MyCustomEnv)
```

Then set `simulator: my_simulator` in your YAML config.

### Checkpoint Backward Compatibility

When evolving the config schema, add migrations in `CheckpointManager._migrate()`:

```python
def _migrate(self, ckpt, from_version):
    if _compare_versions(from_version, "1.1.0") < 0:
        ckpt["config"]["new_field"] = default_value
    ckpt["version"] = CURRENT_VERSION
    return ckpt
```

---

## Known Limitations

- **Web UI** uses placeholder data — connect to real training metrics for production
- **Mock environment** has an empty workspace (no obstacles) — real simulators add complexity
- **Single-GPU only** — distributed training requires additional orchestration
- **MCTS** can be slow on CPU (>1s per action at 800 simulations) — use GPU or reduce sims for dev

---

## License

MIT
