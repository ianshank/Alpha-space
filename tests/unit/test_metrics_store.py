"""Comprehensive unit tests for src/visualization/metrics_store.py.

Tests the MetricsStore class including append, read, persistence,
error handling, and multi-metric support.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.visualization.metrics_store import MetricsStore

# =========================================================================
# MetricsStore
# =========================================================================


class TestMetricsStore:
    """Tests for the MetricsStore class."""

    def test_append_and_read(self, tmp_path: Path) -> None:
        """Append entries and verify read_all returns correct data."""
        store = MetricsStore(tmp_path)

        # Append some metrics
        store.append(0, {"reward": 1.5, "policy_loss": 0.3})
        store.append(1, {"reward": 2.1, "policy_loss": 0.25})
        store.append(2, {"reward": 1.8, "policy_loss": 0.28})

        # Read back
        metrics = store.read_all()

        assert "reward" in metrics
        assert "policy_loss" in metrics
        assert metrics["reward"] == [1.5, 2.1, 1.8]
        assert metrics["policy_loss"] == [0.3, 0.25, 0.28]

    def test_empty_store_returns_empty_dict(self, tmp_path: Path) -> None:
        """Reading from empty store should return empty dict."""
        store = MetricsStore(tmp_path)
        metrics = store.read_all()
        assert metrics == {}

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Write with one instance, read with another (same directory)."""
        # Write with first instance
        store1 = MetricsStore(tmp_path)
        store1.append(0, {"reward": 10.0})
        store1.append(1, {"reward": 20.0})

        # Read with second instance
        store2 = MetricsStore(tmp_path)
        metrics = store2.read_all()

        assert "reward" in metrics
        assert metrics["reward"] == [10.0, 20.0]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """When JSON file doesn't exist, read_all should return empty dict."""
        store = MetricsStore(tmp_path)
        # Don't write anything, just read
        metrics = store.read_all()
        assert metrics == {}

    def test_corrupt_file_handled_gracefully(self, tmp_path: Path) -> None:
        """Write garbage to file, read_all should return empty dict."""
        store = MetricsStore(tmp_path)

        # Write corrupt data directly to file
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text("{ this is not valid json !", encoding="utf-8")

        # Should handle gracefully and return empty
        metrics = store.read_all()
        assert metrics == {}

    def test_multiple_metrics(self, tmp_path: Path) -> None:
        """Append entries with different metric keys."""
        store = MetricsStore(tmp_path)

        # Episode 0: reward and loss
        store.append(0, {"reward": 1.0, "loss": 0.5})

        # Episode 1: reward, loss, and value_loss
        store.append(1, {"reward": 2.0, "loss": 0.4, "value_loss": 0.3})

        # Episode 2: reward and value_loss only
        store.append(2, {"reward": 1.5, "value_loss": 0.25})

        metrics = store.read_all()

        # All metrics should be present
        assert "reward" in metrics
        assert "loss" in metrics
        assert "value_loss" in metrics

        # Missing metrics should be filled with 0.0
        assert metrics["reward"] == [1.0, 2.0, 1.5]
        assert metrics["loss"] == [0.5, 0.4, 0.0]  # Episode 2 missing loss
        assert metrics["value_loss"] == [0.0, 0.3, 0.25]  # Episode 0 missing value_loss

    def test_episode_ordering_preserved(self, tmp_path: Path) -> None:
        """Entries should be returned in the order they were appended."""
        store = MetricsStore(tmp_path)

        # Append episodes in specific order
        store.append(0, {"reward": 1.0})
        store.append(1, {"reward": 2.0})
        store.append(2, {"reward": 3.0})
        store.append(3, {"reward": 4.0})

        metrics = store.read_all()

        # Order should be preserved
        assert metrics["reward"] == [1.0, 2.0, 3.0, 4.0]

    def test_append_updates_existing_file(self, tmp_path: Path) -> None:
        """Appending should preserve existing entries."""
        store = MetricsStore(tmp_path)

        # First append
        store.append(0, {"reward": 1.0})

        # Verify file exists
        metrics_file = tmp_path / "metrics.json"
        assert metrics_file.exists()

        # Second append
        store.append(1, {"reward": 2.0})

        # Both entries should be present
        metrics = store.read_all()
        assert metrics["reward"] == [1.0, 2.0]

    def test_metrics_file_created_in_log_dir(self, tmp_path: Path) -> None:
        """MetricsStore should create metrics.json in the specified directory."""
        store = MetricsStore(tmp_path)
        store.append(0, {"reward": 1.0})

        metrics_file = tmp_path / "metrics.json"
        assert metrics_file.exists()

    def test_log_dir_created_if_missing(self, tmp_path: Path) -> None:
        """Log directory should be created if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "directory"
        assert not nested_dir.exists()

        MetricsStore(nested_dir)
        assert nested_dir.exists()

    def test_multiple_appends_atomic(self, tmp_path: Path) -> None:
        """Multiple appends should all be persisted correctly."""
        store = MetricsStore(tmp_path)

        # Rapidly append multiple entries
        for i in range(10):
            store.append(i, {"reward": float(i)})

        metrics = store.read_all()
        assert metrics["reward"] == list(float(i) for i in range(10))

    def test_read_after_corrupt_returns_empty(self, tmp_path: Path) -> None:
        """After encountering corrupt file, store should still be usable."""
        store = MetricsStore(tmp_path)

        # Write corrupt data
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text("not json", encoding="utf-8")

        # First read returns empty
        metrics1 = store.read_all()
        assert metrics1 == {}

        # Can still append new data (which overwrites corrupt file)
        store.append(0, {"reward": 5.0})
        metrics2 = store.read_all()
        assert metrics2["reward"] == [5.0]

    def test_non_list_json_handled(self, tmp_path: Path) -> None:
        """If JSON file contains non-list, should return empty dict."""
        store = MetricsStore(tmp_path)

        # Write valid JSON but wrong type (dict instead of list)
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        metrics = store.read_all()
        assert metrics == {}

    def test_float_conversion(self, tmp_path: Path) -> None:
        """Metric values should be stored and retrieved as floats."""
        store = MetricsStore(tmp_path)

        # Append integer values
        store.append(0, {"reward": 10, "loss": 5})

        metrics = store.read_all()

        # Should still work (Python's JSON handles int/float conversion)
        assert metrics["reward"] == [10]
        assert metrics["loss"] == [5]
