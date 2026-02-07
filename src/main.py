"""CLI entry point for the ZeroG-RL training system.

Subcommands:
    train     — Run self-play training.
    evaluate  — Evaluate a checkpoint.

Examples::

    python -m src.main train --config configs/smoke_test.yaml
    python -m src.main evaluate --config configs/smoke_test.yaml --checkpoint checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from src.config import load_config
from src.logging_config import setup_logging

logger = structlog.get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zerog-rl",
        description="Zero-G Spatial Awareness RL Agent System",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Minimum log level.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_parser = sub.add_parser("train", help="Run self-play training.")
    train_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from."
    )
    train_parser.add_argument(
        "--episodes", type=int, default=None, help="Override number of episodes."
    )
    train_parser.add_argument("--gpu", type=int, default=None, help="GPU index (or omit for auto).")
    train_parser.add_argument(
        "--no-logging", action="store_true", help="Disable W&B / external logging."
    )
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    train_parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler.")
    train_parser.add_argument(
        "--pdb-on-error", action="store_true", help="Drop into pdb on exception."
    )

    # ---- evaluate ----
    eval_parser = sub.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    eval_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    eval_parser.add_argument(
        "--episodes", type=int, default=100, help="Number of evaluation episodes."
    )
    eval_parser.add_argument("--use-mcts", action="store_true", help="Use MCTS during evaluation.")

    return parser


def _cmd_train(args: argparse.Namespace) -> None:
    """Execute the ``train`` subcommand."""
    overrides: dict[str, object] = {}
    if args.episodes is not None:
        overrides["training.num_episodes"] = args.episodes
    if args.no_logging:
        overrides["no_logging"] = True
    if args.debug:
        overrides["debug"] = True

    config = load_config(config_path=args.config, overrides=overrides)

    from src.training.trainer import Trainer

    resume_path = Path(args.resume) if args.resume else None

    profiler = None
    if args.profile:
        import torch

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
            record_shapes=True,
            profile_memory=True,
        )
        profiler.__enter__()

    try:
        trainer = Trainer(config=config, resume_from=resume_path)
        summary = trainer.train()
        logger.info("training_summary", **summary)
    except KeyboardInterrupt:
        logger.info("training_interrupted_by_user")
        raise
    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("training_failed")
        if args.pdb_on_error:
            import pdb

            pdb.post_mortem()
        raise
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute the ``evaluate`` subcommand."""
    config = load_config(config_path=args.config)

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(config=config, checkpoint_path=Path(args.checkpoint))
    result = evaluator.run(num_episodes=args.episodes, deterministic=not args.use_mcts)
    logger.info("evaluation_summary", **{k: v for k, v in result.items() if k != "episode_rewards"})


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if getattr(args, "debug", False) else args.log_level
    setup_logging(log_level=log_level)

    commands = {
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
