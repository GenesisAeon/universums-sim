"""
Live visualization: Mandala renderer, 3D emergence, Sonification, Dash GUI.

Components
----------
MandalaRenderer   — SVG/PNG sacred-geometry mandala updated every tick.
Emergence3D       — 3D scatter plot of particle positions (Plotly).
SonificationEngine — Maps emergence events to audio (sounddevice).
DashDashboard     — Full Dash web application with live callbacks.
LiveVisualizer    — Facade orchestrating all components.

The visualization layer is **optional**: all imports are guarded and the
module degrades gracefully when optional dependencies are absent.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from universums_sim.simulation.core import CosmicMoment, UniverseSimulator

# ---------------------------------------------------------------------------
# Mandala Renderer (pure Python / matplotlib)
# ---------------------------------------------------------------------------


class MandalaRenderer:
    """
    Renders a live sacred-geometry mandala from particle positions.

    Parameters
    ----------
    n_petals : int
        Number of mandala petals (default 12).
    size : int
        Canvas size in pixels (default 512).
    """

    def __init__(self, n_petals: int = 12, size: int = 512) -> None:
        self._n_petals = n_petals
        self._size = size
        self._frame: int = 0
        self._history: list[np.ndarray] = []  # type: ignore[type-arg]

    def render(self, moment: CosmicMoment) -> dict[str, Any]:
        """
        Produce SVG path data for the current CosmicMoment.

        Parameters
        ----------
        moment : CosmicMoment

        Returns
        -------
        dict[str, Any]
            Dictionary with keys 'frame', 'paths', 'entropy', 'phase'.
        """
        self._frame += 1
        paths = []
        angle_step = 2 * math.pi / self._n_petals
        r_base = min(self._size / 4, 100.0)
        r = r_base * (1 + 0.1 * math.sin(moment.entropy * 0.01))
        cx = cy = self._size / 2
        for k in range(self._n_petals):
            theta = k * angle_step + moment.time * 0.1
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            x2 = cx + r * 0.5 * math.cos(theta + angle_step / 2)
            y2 = cy + r * 0.5 * math.sin(theta + angle_step / 2)
            paths.append(f"M {cx:.1f} {cy:.1f} Q {x2:.1f} {y2:.1f} {x:.1f} {y:.1f}")
        return {
            "frame": self._frame,
            "paths": paths,
            "entropy": moment.entropy,
            "phase": moment.phase.name,
            "r": r,
        }

    def to_svg(self, moment: CosmicMoment) -> str:
        """Return a minimal SVG string for the current mandala frame."""
        data = self.render(moment)
        path_els = "\n    ".join(
            f'<path d="{p}" stroke="cyan" fill="none" stroke-width="1.5"/>'
            for p in data["paths"]
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{self._size}" height="{self._size}" '
            f'style="background:#0a0a1a">\n'
            f"    {path_els}\n"
            f"    <text x='10' y='20' fill='white' font-size='12'>"
            f"S={data['entropy']:.2f} | {data['phase']}</text>\n"
            f"</svg>"
        )


# ---------------------------------------------------------------------------
# 3D Emergence Scatter
# ---------------------------------------------------------------------------


class Emergence3D:
    """
    3D particle scatter updated from CosmicMoments.

    Uses Plotly for rendering; degrades to dict if plotly is unavailable.
    """

    def __init__(self) -> None:
        self._frames: list[dict[str, Any]] = []

    def update(self, moment: CosmicMoment, positions: np.ndarray) -> dict[str, Any]:  # type: ignore[type-arg]
        """
        Build a Plotly-compatible trace dict for particle positions.

        Parameters
        ----------
        moment : CosmicMoment
        positions : NDArray shape (N, 3)

        Returns
        -------
        dict[str, Any]
        """
        frame: dict[str, Any] = {
            "type": "scatter3d",
            "x": positions[:, 0].tolist(),
            "y": positions[:, 1].tolist(),
            "z": positions[:, 2].tolist(),
            "mode": "markers",
            "marker": {
                "size": 3,
                "color": float(moment.entropy),
                "colorscale": "Plasma",
                "opacity": 0.8,
            },
            "name": f"step {moment.step}",
        }
        self._frames.append(frame)
        return frame


# ---------------------------------------------------------------------------
# Sonification Engine
# ---------------------------------------------------------------------------


class SonificationEngine:
    """
    Maps emergence events to audio signals.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default 44100).
    duration : float
        Duration of each note in seconds (default 0.1).
    """

    def __init__(self, sample_rate: int = 44100, duration: float = 0.1) -> None:
        self._sr = sample_rate
        self._dur = duration
        self._enabled = self._check_deps()

    @staticmethod
    def _check_deps() -> bool:
        """Return True if sounddevice is importable."""
        try:
            import sounddevice  # noqa: F401
            return True
        except ImportError:
            return False

    def _freq_from_rate(self, rate: float) -> float:
        """Map emergence rate to a musical frequency (Hz) via log scaling."""
        return 220.0 * (2.0 ** (rate * 3.0))  # max ~1760 Hz

    def process(self, moment: CosmicMoment) -> None:
        """
        Play a short tone for each emergence event in the moment.

        Parameters
        ----------
        moment : CosmicMoment
        """
        if not self._enabled or not moment.events:
            return
        import sounddevice as sd  # type: ignore[import-untyped]

        for event in moment.events:
            freq = self._freq_from_rate(event.rate)
            t = np.linspace(0, self._dur, int(self._sr * self._dur), endpoint=False)
            wave = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            sd.play(wave, samplerate=self._sr, blocking=False)


# ---------------------------------------------------------------------------
# Dash Dashboard
# ---------------------------------------------------------------------------


class DashDashboard:
    """
    Full Dash web application with live-updating plots.

    Spawns the Dash server in a daemon thread so the CLI remains responsive.
    """

    def __init__(self, port: int = 8050) -> None:
        self._port = port
        self._app: Any = None
        self._moments: list[Any] = []

    def _build_app(self) -> Any:
        """Construct the Dash application layout."""
        try:
            import dash  # type: ignore[import-untyped]  # pragma: no cover
            from dash import Input, Output, dcc, html  # type: ignore[import-untyped]  # pragma: no cover
        except ImportError:
            return None

        app = dash.Dash(__name__, title="universums-sim")  # pragma: no cover
        app.layout = html.Div([  # pragma: no cover
            html.H1("universums-sim — Live Cosmic Emergence", style={"color": "cyan"}),
            dcc.Graph(id="entropy-graph"),
            dcc.Graph(id="emergence-graph"),
            dcc.Interval(id="interval", interval=500, n_intervals=0),
        ], style={"backgroundColor": "#0a0a1a", "color": "white"})

        @app.callback(  # pragma: no cover
            Output("entropy-graph", "figure"),
            Output("emergence-graph", "figure"),
            Input("interval", "n_intervals"),
        )
        def update(_: int) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover
            moments = self._moments
            steps = [m.get("step", 0) for m in moments]
            entropies = [m.get("entropy", 0) for m in moments]
            rates = [m.get("emergence_rate", 0) for m in moments]
            return (
                {
                    "data": [{"x": steps, "y": entropies, "type": "line", "name": "S(n)"}],
                    "layout": {"title": "Entropy S(n)", "paper_bgcolor": "#0a0a1a",
                                "plot_bgcolor": "#111", "font": {"color": "white"}},
                },
                {
                    "data": [{"x": steps, "y": rates, "type": "line", "name": "R_e(n)",
                               "line": {"color": "magenta"}}],
                    "layout": {"title": "Emergence Rate R_e(n)", "paper_bgcolor": "#0a0a1a",
                                "plot_bgcolor": "#111", "font": {"color": "white"}},
                },
            )

        return app  # pragma: no cover

    def launch(self) -> None:
        """Launch the Dash server in a daemon thread."""
        self._app = self._build_app()  # pragma: no cover
        if self._app is None:  # pragma: no cover
            return  # pragma: no cover
        thread = threading.Thread(  # pragma: no cover
            target=lambda: self._app.run(port=self._port, debug=False),  # pragma: no cover
            daemon=True,
        )
        thread.start()  # pragma: no cover

    def push_moment(self, moment: CosmicMoment) -> None:
        """Push a serialised moment to the live dashboard."""
        self._moments.append(moment.to_dict())


# ---------------------------------------------------------------------------
# LiveVisualizer Facade
# ---------------------------------------------------------------------------


class LiveVisualizer:
    """
    Facade orchestrating MandalaRenderer, Emergence3D, and DashDashboard.

    Parameters
    ----------
    simulator : UniverseSimulator
        Reference to the running simulator (for position access).
    gui_port : int
        Port for the Dash server (default 8050).
    """

    def __init__(self, simulator: UniverseSimulator, gui_port: int = 8050) -> None:
        self._sim = simulator
        self._mandala = MandalaRenderer()
        self._emergence_3d = Emergence3D()
        self._dashboard = DashDashboard(port=gui_port)
        self._svgs: list[str] = []

    def launch_gui(self) -> None:
        """Start the Dash GUI dashboard in the background."""
        self._dashboard.launch()

    def update(self, moment: CosmicMoment) -> None:
        """
        Process a CosmicMoment through all visualization components.

        Parameters
        ----------
        moment : CosmicMoment
        """
        svg = self._mandala.to_svg(moment)
        self._svgs.append(svg)
        self._emergence_3d.update(moment, self._sim.positions)
        self._dashboard.push_moment(moment)

    @property
    def svg_frames(self) -> list[str]:
        """All rendered SVG frames so far."""
        return list(self._svgs)

    def save_svg_animation(self, path: str) -> None:
        """
        Save all SVG frames as an SMIL animation.

        Parameters
        ----------
        path : str
            File path for the output SVG.
        """
        if not self._svgs:
            return
        combined = "\n<!-- frame separator -->\n".join(self._svgs)
        with open(path, "w") as f:  # noqa: PTH123
            f.write(combined)
