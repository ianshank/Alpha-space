"""Gradio-based web UI for training visualisation and control.

Provides a dashboard with:
  - 3D trajectory viewer (via Plotly)
  - Live training metrics charts
  - Episode replay controls
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def create_trajectory_plot(
    positions: np.ndarray,
    goal: np.ndarray | None = None,
    title: str = "3D Trajectory",
) -> Any:
    """Create a Plotly 3D scatter of a spacecraft trajectory.

    Args:
        positions: ``(T, 3)`` array of positions over time.
        goal: Optional ``(3,)`` goal position.
        title: Plot title.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines+markers",
            marker=dict(size=2, color=np.arange(len(positions)), colorscale="Viridis"),
            line=dict(width=2),
            name="Trajectory",
        )
    )

    # Start marker
    fig.add_trace(
        go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode="markers",
            marker=dict(size=8, color="green"),
            name="Start",
        )
    )

    # End marker
    fig.add_trace(
        go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="End",
        )
    )

    if goal is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[goal[0]],
                y=[goal[1]],
                z=[goal[2]],
                mode="markers",
                marker=dict(size=10, color="gold", symbol="diamond"),
                name="Goal",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def create_metrics_plot(
    metrics: dict[str, list[float]],
    title: str = "Training Metrics",
) -> Any:
    """Create a Plotly line chart of training metrics over episodes.

    Args:
        metrics: Dict mapping metric name to list of per-episode values.
        title: Plot title.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, values in metrics.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(title=title, xaxis_title="Episode", yaxis_title="Value")
    return fig


def launch_ui(
    checkpoint_path: Path | None = None,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio web interface.

    Args:
        checkpoint_path: Optional checkpoint to load for replay.
        port: HTTP port.
        share: If ``True``, create a public Gradio link.
    """
    try:
        import gradio as gr
    except ImportError:
        logger.error("gradio not installed — run: pip install gradio")
        return

    with gr.Blocks(title="ZeroG-RL Dashboard") as demo:
        gr.Markdown("# ZeroG-RL Training Dashboard")

        with gr.Tab("Training Metrics"):
            metrics_plot = gr.Plot(label="Metrics")
            refresh_btn = gr.Button("Refresh")

            def _refresh_metrics() -> Any:
                # Placeholder — in production would read from metrics DB
                dummy = {"reward": list(np.random.randn(50).cumsum())}
                return create_metrics_plot(dummy)

            refresh_btn.click(fn=_refresh_metrics, outputs=metrics_plot)

        with gr.Tab("3D Trajectory"):
            traj_plot = gr.Plot(label="Trajectory")
            gen_btn = gr.Button("Generate Sample Trajectory")

            def _gen_trajectory() -> Any:
                t = np.linspace(0, 4 * np.pi, 200)
                pos = np.column_stack([np.sin(t), np.cos(t), t / (4 * np.pi) * 5])
                return create_trajectory_plot(pos, goal=np.array([0.0, 1.0, 5.0]))

            gen_btn.click(fn=_gen_trajectory, outputs=traj_plot)

        with gr.Tab("Configuration"):
            gr.JSON(label="Current Config")
            gr.Markdown("*Load a config file to display here.*")

    logger.info("launching_web_ui", port=port, share=share)
    demo.launch(server_port=port, share=share, prevent_thread_lock=True)
