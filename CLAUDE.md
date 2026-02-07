# CLAUDE.md - Zero-G RL Agent System

## Build Commands

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with your API keys
```

### Training
```bash
# Smoke test (1 episode, CPU, smallest config)
python -m src.main train --config configs/smoke_test.yaml

# Quick test (10 episodes, CPU)
python -m src.main train --config configs/quick_test.yaml

# Full training (1000 episodes, GPU)
python -m src.main train --config configs/docking_task.yaml

# Resume from checkpoint
python -m src.main train --config configs/docking_task.yaml --resume checkpoints/episode_500.pt

# Override episode count
python -m src.main train --config configs/smoke_test.yaml --episodes 5
```

### Evaluation
```bash
python -m src.main evaluate --checkpoint checkpoints/final.pt --episodes 100
```

### Testing
```bash
# Full suite
pytest tests/ -v --cov=src --cov-report=term-missing

# Unit tests only (fast, ~8s)
pytest tests/unit/ -v

# Integration tests (includes end-to-end training, ~15s)
pytest tests/integration/ -v

# Property tests (Hypothesis, ~5s)
pytest tests/properties/ -v

# With coverage threshold
pytest tests/ --cov=src --cov-fail-under=85
```

## Architecture Decisions

### 2026-02-07: Mock environment as default simulator
Uses internal ZeroGDynamics physics engine with symplectic Euler integration.
Unity ML-Agents / Isaac Sim abstracted behind Gymnasium interface via `ZeroGEnv`
base class — register new backends with `register_env()`.

### 2026-02-07: MCTS uses progressive widening with uniform priors
Continuous action space bounded by `max_children` per node. Actions sampled from
the policy network's Gaussian distribution. PUCT selection with Dirichlet noise
at root for exploration.

### 2026-02-07: Pydantic v2 for all configuration
All hyperparameters flow through validated Pydantic models. No hardcoded values
in application code. YAML files + env var overrides (`ZEROG_` prefix) + explicit
overrides dict. Config schema is versioned for backward-compatible checkpoints.

### 2026-02-07: Quaternions [w,x,y,z] Hamilton convention
Re-normalised after every physics step. Used throughout for orientation to avoid
gimbal lock. `src/utils/common.py` has multiply, normalize, and to-rotation-matrix.

### 2026-02-07: Symplectic Euler integrator for physics
Better energy conservation than explicit Euler. Momentum conservation validated
after every step when `validate=True` (default). PhysicsViolationError raised on
conservation failures.

## Known Issues

- `log_tensor_stats` on scalar tensors: handled by checking `numel() > 1`
- Gradient norm logging uses `torch.tensor([grad_norm.item()])` to avoid copy warning
- Web UI (`visualization/web_ui.py`) uses placeholder data — connect to real
  training metrics in production

## Key File Layout
```
src/
├── main.py                      # CLI entry point
├── config.py                    # Pydantic config models + YAML loader
├── logging_config.py            # Structlog setup
├── networks/policy_value_net.py # 3D CNN + MLP dual-headed network
├── mcts/engine.py               # Continuous-action MCTS with progressive widening
├── physics/zero_g_dynamics.py   # Newton-Euler integrator, momentum validation
├── environments/
│   ├── base.py                  # Abstract Gymnasium env
│   ├── mock_env.py              # Built-in physics-backed mock env
│   └── factory.py               # make_env() / register_env()
├── replay_buffer/buffer.py      # FIFO buffer + Parquet persistence
├── checkpointing/               # Versioned save/load with migration
├── training/trainer.py          # Self-play + MCTS + policy update loop
├── evaluation/evaluator.py      # Standalone checkpoint evaluation
├── debugging/instrumentation.py # Timing/memory/GPU decorators
├── visualization/web_ui.py      # Gradio dashboard (Plotly 3D)
└── utils/common.py              # Seeding, quaternions, Timer, path validation
```
