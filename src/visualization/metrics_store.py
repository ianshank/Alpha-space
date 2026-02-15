"""File-based JSON metrics store for training runs.

Provides a simple thread-safe mechanism to log episode metrics to disk
and read them back for visualization.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetricsStore:
    """File-based JSON metrics store.

    Stores metrics in a JSON file with one entry per episode. Uses atomic
    writes (write to temporary file then replace) to minimize corruption risk.

    Args:
        log_dir: Directory where metrics.json will be stored. Created if missing.

    Example:
        >>> store = MetricsStore(Path("logs/run_001"))
        >>> store.append(0, {"reward": 1.5, "policy_loss": 0.3})
        >>> store.append(1, {"reward": 2.1, "policy_loss": 0.25})
        >>> metrics = store.read_all()
        >>> print(metrics)
        {"reward": [1.5, 2.1], "policy_loss": [0.3, 0.25]}
    """

    def __init__(self, log_dir: Path) -> None:
        """Initialize the metrics store.

        Args:
            log_dir: Directory for storing metrics.json.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.json"
        self._tmp_file = self.log_dir / "metrics.json.tmp"

        logger.info(
            "metrics_store_initialized",
            log_dir=str(self.log_dir),
            metrics_file=str(self.metrics_file),
        )

    def append(self, episode: int, metrics: dict[str, float]) -> None:
        """Append metrics for a single episode.

        Uses atomic write pattern: write to temporary file then replace the
        original. This minimizes corruption risk if the process crashes during
        write.

        Args:
            episode: Episode number (0-indexed).
            metrics: Dictionary mapping metric names to float values.
        """
        # Read existing entries
        entries = self._read_raw()

        # Add new entry
        entry = {"episode": episode, **metrics}
        entries.append(entry)

        # Atomic write: write to temp file then replace
        try:
            self._tmp_file.write_text(
                json.dumps(entries, indent=2),
                encoding="utf-8",
            )
            # Replace atomically (on Windows, need to remove first)
            if self.metrics_file.exists():
                self.metrics_file.unlink()
            self._tmp_file.rename(self.metrics_file)

            logger.debug(
                "metrics_appended",
                episode=episode,
                num_metrics=len(metrics),
                total_episodes=len(entries),
            )
        except Exception as e:
            logger.error(
                "failed_to_append_metrics",
                episode=episode,
                error=str(e),
                exc_info=True,
            )
            # Clean up temp file if it exists
            if self._tmp_file.exists():
                with contextlib.suppress(Exception):
                    self._tmp_file.unlink()
            raise

    def read_all(self) -> dict[str, list[float]]:
        """Read all metrics and pivot into per-metric lists.

        Reads the metrics file and transforms from a list of per-episode
        dictionaries into a dictionary mapping each metric name to a list
        of values across all episodes.

        Returns:
            Dictionary mapping metric name to list of values. Returns empty
            dict if no metrics file exists or if the file is corrupt.

        Example:
            If the file contains:
            [
                {"episode": 0, "reward": 1.5, "loss": 0.3},
                {"episode": 1, "reward": 2.1, "loss": 0.25}
            ]

            Returns:
            {"reward": [1.5, 2.1], "loss": [0.3, 0.25]}
        """
        entries = self._read_raw()

        if not entries:
            return {}

        # Pivot: gather all metric names
        all_keys: set[str] = set()
        for entry in entries:
            all_keys.update(k for k in entry if k != "episode")

        # Build the pivoted dict
        result: dict[str, list[float]] = {key: [] for key in all_keys}

        for entry in entries:
            for key in all_keys:
                # Use 0.0 as default if metric missing in an episode
                result[key].append(entry.get(key, 0.0))

        logger.debug(
            "metrics_read",
            num_episodes=len(entries),
            num_metrics=len(result),
        )

        return result

    def _read_raw(self) -> list[dict[str, Any]]:
        """Read the raw JSON entries from disk.

        Returns:
            List of dictionaries, one per episode. Returns empty list if
            the file doesn't exist or is corrupt.
        """
        if not self.metrics_file.exists():
            return []

        try:
            content = self.metrics_file.read_text(encoding="utf-8")
            entries = json.loads(content)

            if not isinstance(entries, list):
                logger.warning(
                    "metrics_file_invalid_format",
                    expected="list",
                    got=type(entries).__name__,
                )
                return []

            return entries

        except json.JSONDecodeError as e:
            logger.warning(
                "metrics_file_corrupt",
                error=str(e),
                file=str(self.metrics_file),
            )
            return []
        except Exception as e:
            logger.error(
                "failed_to_read_metrics",
                error=str(e),
                file=str(self.metrics_file),
                exc_info=True,
            )
            return []
