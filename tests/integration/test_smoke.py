"""Smoke test: verify the CLI entry point can run 1 episode end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


class TestSmoke:
    def test_smoke_train_one_episode(self) -> None:
        """Run the CLI training command for 1 episode via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.main",
                "train",
                "--config",
                "configs/smoke_test.yaml",
                "--episodes",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            f"Smoke test failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
