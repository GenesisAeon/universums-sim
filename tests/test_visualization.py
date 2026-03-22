"""
Tests for universums_sim.visualization.live.

Covers:
- MandalaRenderer: render(), to_svg()
- Emergence3D: update()
- SonificationEngine: check_deps(), process() (mocked)
- DashDashboard: push_moment()
- LiveVisualizer: update(), svg_frames, save_svg_animation()
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from universums_sim.simulation.core import (
    CosmicMoment,
    SimulationConfig,
    SimulationPhase,
    UniverseSimulator,
)
from universums_sim.simulation.emergence import EmergenceType
from universums_sim.simulation.lagrangian import CollapseState
from universums_sim.visualization.live import (
    DashDashboard,
    Emergence3D,
    LiveVisualizer,
    MandalaRenderer,
    SonificationEngine,
)


def _make_moment(step: int = 0, entropy: float = 10.0) -> CosmicMoment:
    return CosmicMoment(
        step=step,
        time=float(step) * 0.01,
        entropy=entropy,
        emergence_rate=0.05,
        hamiltonian=-3.0,
        phase=SimulationPhase.GENESIS,
        events=(),
        collapse_state=CollapseState.STABLE,
        observer_hash="test",
        wall_time=time.time(),
    )


# ---------------------------------------------------------------------------
# MandalaRenderer (25 tests)
# ---------------------------------------------------------------------------

class TestMandalaRenderer:
    def test_creation(self):
        r = MandalaRenderer()
        assert r is not None

    def test_default_n_petals(self):
        r = MandalaRenderer()
        assert r._n_petals == 12

    def test_custom_n_petals(self):
        r = MandalaRenderer(n_petals=8)
        assert r._n_petals == 8

    def test_custom_size(self):
        r = MandalaRenderer(size=256)
        assert r._size == 256

    def test_frame_starts_zero(self):
        r = MandalaRenderer()
        assert r._frame == 0

    def test_render_returns_dict(self):
        r = MandalaRenderer()
        m = _make_moment()
        d = r.render(m)
        assert isinstance(d, dict)

    def test_render_increments_frame(self):
        r = MandalaRenderer()
        r.render(_make_moment())
        assert r._frame == 1

    def test_render_has_paths(self):
        r = MandalaRenderer()
        d = r.render(_make_moment())
        assert "paths" in d

    def test_render_paths_count(self):
        r = MandalaRenderer(n_petals=6)
        d = r.render(_make_moment())
        assert len(d["paths"]) == 6

    def test_render_has_entropy(self):
        r = MandalaRenderer()
        d = r.render(_make_moment(entropy=77.0))
        assert d["entropy"] == 77.0

    def test_render_has_phase(self):
        r = MandalaRenderer()
        d = r.render(_make_moment())
        assert "phase" in d

    def test_render_has_frame(self):
        r = MandalaRenderer()
        d = r.render(_make_moment())
        assert "frame" in d

    def test_render_has_r(self):
        r = MandalaRenderer()
        d = r.render(_make_moment())
        assert "r" in d

    def test_to_svg_returns_string(self):
        r = MandalaRenderer()
        svg = r.to_svg(_make_moment())
        assert isinstance(svg, str)

    def test_to_svg_contains_svg_tag(self):
        r = MandalaRenderer()
        svg = r.to_svg(_make_moment())
        assert "<svg" in svg

    def test_to_svg_contains_path(self):
        r = MandalaRenderer()
        svg = r.to_svg(_make_moment())
        assert "<path" in svg

    def test_to_svg_contains_entropy(self):
        r = MandalaRenderer()
        svg = r.to_svg(_make_moment(entropy=42.5))
        assert "42.5" in svg

    def test_to_svg_closes_svg(self):
        r = MandalaRenderer()
        svg = r.to_svg(_make_moment())
        assert "</svg>" in svg

    def test_render_multiple_moments(self):
        r = MandalaRenderer()
        for i in range(10):
            r.render(_make_moment(step=i, entropy=float(i)))
        assert r._frame == 10

    def test_render_path_strings(self):
        r = MandalaRenderer()
        d = r.render(_make_moment())
        for p in d["paths"]:
            assert isinstance(p, str)
            assert "M " in p

    def test_different_entropy_different_r(self):
        r = MandalaRenderer()
        d1 = r.render(_make_moment(entropy=1.0))
        r2 = MandalaRenderer()
        d2 = r2.render(_make_moment(entropy=100.0))
        assert d1["r"] != d2["r"]

    def test_to_svg_width(self):
        r = MandalaRenderer(size=512)
        svg = r.to_svg(_make_moment())
        assert 'width="512"' in svg

    def test_to_svg_height(self):
        r = MandalaRenderer(size=256)
        svg = r.to_svg(_make_moment())
        assert 'height="256"' in svg

    def test_history_empty_initial(self):
        r = MandalaRenderer()
        assert r._history == []

    def test_render_phase_name_in_dict(self):
        r = MandalaRenderer()
        d = r.render(_make_moment())
        assert d["phase"] == "GENESIS"


# ---------------------------------------------------------------------------
# Emergence3D (15 tests)
# ---------------------------------------------------------------------------

class TestEmergence3D:
    def test_creation(self):
        e = Emergence3D()
        assert e is not None

    def test_frames_empty_initial(self):
        e = Emergence3D()
        assert e._frames == []

    def test_update_returns_dict(self):
        e = Emergence3D()
        pos = np.random.default_rng(0).standard_normal((8, 3))
        d = e.update(_make_moment(), pos)
        assert isinstance(d, dict)

    def test_update_has_x(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(), pos)
        assert "x" in d

    def test_update_has_y(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(), pos)
        assert "y" in d

    def test_update_has_z(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(), pos)
        assert "z" in d

    def test_update_x_length(self):
        e = Emergence3D()
        pos = np.random.default_rng(0).standard_normal((6, 3))
        d = e.update(_make_moment(), pos)
        assert len(d["x"]) == 6

    def test_update_appends_frame(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        e.update(_make_moment(), pos)
        assert len(e._frames) == 1

    def test_update_multiple_frames(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        for i in range(5):
            e.update(_make_moment(step=i), pos)
        assert len(e._frames) == 5

    def test_update_type_scatter3d(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(), pos)
        assert d["type"] == "scatter3d"

    def test_update_mode_markers(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(), pos)
        assert d["mode"] == "markers"

    def test_update_marker_color_is_entropy(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(entropy=55.0), pos)
        assert d["marker"]["color"] == 55.0

    def test_update_name_contains_step(self):
        e = Emergence3D()
        pos = np.zeros((4, 3))
        d = e.update(_make_moment(step=7), pos)
        assert "7" in d["name"]

    def test_update_positions_correct_x(self):
        e = Emergence3D()
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d = e.update(_make_moment(), pos)
        assert d["x"] == pytest.approx([1.0, 4.0])

    def test_update_positions_correct_z(self):
        e = Emergence3D()
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d = e.update(_make_moment(), pos)
        assert d["z"] == pytest.approx([3.0, 6.0])


# ---------------------------------------------------------------------------
# SonificationEngine (20 tests)
# ---------------------------------------------------------------------------

class TestSonificationEngine:
    def test_creation(self):
        s = SonificationEngine()
        assert s is not None

    def test_sample_rate_stored(self):
        s = SonificationEngine(sample_rate=22050)
        assert s._sr == 22050

    def test_duration_stored(self):
        s = SonificationEngine(duration=0.2)
        assert s._dur == 0.2

    def test_check_deps_bool(self):
        result = SonificationEngine._check_deps()
        assert isinstance(result, bool)

    def test_freq_from_rate_base(self):
        s = SonificationEngine()
        f = s._freq_from_rate(0.0)
        assert f == pytest.approx(220.0)

    def test_freq_from_rate_positive(self):
        s = SonificationEngine()
        f = s._freq_from_rate(1.0)
        assert f > 220.0

    def test_freq_from_rate_increases(self):
        s = SonificationEngine()
        f1 = s._freq_from_rate(0.5)
        f2 = s._freq_from_rate(1.0)
        assert f2 > f1

    def test_process_no_events_no_error(self):
        s = SonificationEngine()
        s._enabled = False
        s.process(_make_moment())  # should not raise

    def test_process_disabled_no_error(self):
        s = SonificationEngine()
        s._enabled = False
        m = _make_moment()
        s.process(m)

    def test_process_with_sounddevice_mocked(self):
        import universums_sim.simulation.emergence as em
        from universums_sim.simulation.emergence import EmergenceEvent
        evt = EmergenceEvent(
            event_id="e1", step=0, time=0.0, entropy=1.0, rate=0.5,
            kind=EmergenceType.MESO, description="test",
        )
        import time as _time

        m = CosmicMoment(
            step=0, time=0.0, entropy=1.0, emergence_rate=0.5, hamiltonian=-1.0,
            phase=SimulationPhase.GENESIS, events=(evt,),
            collapse_state=CollapseState.STABLE, observer_hash="x",
            wall_time=_time.time(),
        )
        mock_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            s = SonificationEngine()
            s._enabled = True
            s.process(m)
            mock_sd.play.assert_called_once()

    def test_freq_range_valid(self):
        s = SonificationEngine()
        for rate in [0.0, 0.5, 1.0, 2.0, 3.0]:
            f = s._freq_from_rate(rate)
            assert f > 100

    def test_default_sample_rate(self):
        s = SonificationEngine()
        assert s._sr == 44100

    def test_default_duration(self):
        s = SonificationEngine()
        assert s._dur == 0.1

    def test_enabled_is_bool(self):
        s = SonificationEngine()
        assert isinstance(s._enabled, bool)

    def test_freq_monotone(self):
        s = SonificationEngine()
        rates = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
        freqs = [s._freq_from_rate(r) for r in rates]
        for f1, f2 in zip(freqs, freqs[1:]):
            assert f2 >= f1

    def test_process_empty_events_enabled(self):
        s = SonificationEngine()
        s._enabled = True
        m = _make_moment()  # no events
        # Should not raise even when enabled
        mock_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            s.process(m)
            mock_sd.play.assert_not_called()

    def test_sounddevice_not_installed(self):
        with patch.dict("sys.modules", {"sounddevice": None}):
            result = SonificationEngine._check_deps()
            assert isinstance(result, bool)

    def test_freq_large_rate(self):
        s = SonificationEngine()
        f = s._freq_from_rate(10.0)
        assert np.isfinite(f)


# ---------------------------------------------------------------------------
# DashDashboard (10 tests)
# ---------------------------------------------------------------------------

class TestDashDashboard:
    def test_creation(self):
        d = DashDashboard()
        assert d is not None

    def test_default_port(self):
        d = DashDashboard()
        assert d._port == 8050

    def test_custom_port(self):
        d = DashDashboard(port=9090)
        assert d._port == 9090

    def test_moments_empty_initial(self):
        d = DashDashboard()
        assert d._moments == []

    def test_push_moment_appends(self):
        d = DashDashboard()
        d.push_moment(_make_moment())
        assert len(d._moments) == 1

    def test_push_moment_multiple(self):
        d = DashDashboard()
        for i in range(5):
            d.push_moment(_make_moment(step=i))
        assert len(d._moments) == 5

    def test_push_moment_is_dict(self):
        d = DashDashboard()
        d.push_moment(_make_moment())
        assert isinstance(d._moments[0], dict)

    def test_build_app_without_dash(self):
        with patch.dict("sys.modules", {"dash": None}):
            d = DashDashboard()
            result = d._build_app()
            assert result is None

    def test_app_none_initially(self):
        d = DashDashboard()
        assert d._app is None

    def test_push_preserves_step(self):
        d = DashDashboard()
        d.push_moment(_make_moment(step=99))
        assert d._moments[0]["step"] == 99


# ---------------------------------------------------------------------------
# LiveVisualizer (20 tests)
# ---------------------------------------------------------------------------

class TestLiveVisualizer:
    @pytest.fixture
    def sim(self) -> UniverseSimulator:
        return UniverseSimulator(SimulationConfig(n_particles=4, seed=0))

    def test_creation(self, sim):
        v = LiveVisualizer(sim)
        assert v is not None

    def test_svg_frames_empty_initial(self, sim):
        v = LiveVisualizer(sim)
        assert v.svg_frames == []

    def test_update_adds_svg_frame(self, sim):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        assert len(v.svg_frames) == 1

    def test_update_multiple_frames(self, sim):
        v = LiveVisualizer(sim)
        for i in range(5):
            v.update(_make_moment(step=i))
        assert len(v.svg_frames) == 5

    def test_svg_frames_are_strings(self, sim):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        for s in v.svg_frames:
            assert isinstance(s, str)

    def test_svg_frames_contain_svg(self, sim):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        assert "<svg" in v.svg_frames[0]

    def test_save_svg_animation_empty(self, sim, tmp_path):
        v = LiveVisualizer(sim)
        path = str(tmp_path / "anim.svg")
        v.save_svg_animation(path)  # no frames, should not raise

    def test_save_svg_animation_creates_file(self, sim, tmp_path):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        path = str(tmp_path / "anim.svg")
        v.save_svg_animation(path)
        assert Path(path).exists()

    def test_save_svg_animation_content(self, sim, tmp_path):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        path = str(tmp_path / "anim.svg")
        v.save_svg_animation(path)
        content = Path(path).read_text()
        assert "<svg" in content

    def test_dashboard_pushed(self, sim):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        assert len(v._dashboard._moments) == 1

    def test_svg_frames_returns_copy(self, sim):
        v = LiveVisualizer(sim)
        v.update(_make_moment())
        frames1 = v.svg_frames
        frames2 = v.svg_frames
        assert frames1 == frames2

    def test_mandala_renderer_initialized(self, sim):
        v = LiveVisualizer(sim)
        assert v._mandala is not None

    def test_emergence3d_initialized(self, sim):
        v = LiveVisualizer(sim)
        assert v._emergence_3d is not None

    def test_dashboard_initialized(self, sim):
        v = LiveVisualizer(sim)
        assert v._dashboard is not None

    def test_update_entropy_in_svg(self, sim):
        v = LiveVisualizer(sim)
        v.update(_make_moment(entropy=99.5))
        assert "99.5" in v.svg_frames[0]

    def test_launch_gui_no_crash_without_dash(self, sim):
        v = LiveVisualizer(sim)
        # Should not raise even without dash installed
        with patch.object(v._dashboard, "launch"):
            v.launch_gui()

    def test_custom_gui_port(self, sim):
        v = LiveVisualizer(sim, gui_port=9999)
        assert v._dashboard._port == 9999

    def test_10_updates(self, sim):
        v = LiveVisualizer(sim)
        for i in range(10):
            v.update(_make_moment(step=i))
        assert len(v.svg_frames) == 10

    def test_sim_reference_stored(self, sim):
        v = LiveVisualizer(sim)
        assert v._sim is sim

    def test_save_multiple_frames(self, sim, tmp_path):
        v = LiveVisualizer(sim)
        for i in range(3):
            v.update(_make_moment(step=i))
        path = str(tmp_path / "multi.svg")
        v.save_svg_animation(path)
        content = Path(path).read_text()
        assert content.count("<svg") == 3
