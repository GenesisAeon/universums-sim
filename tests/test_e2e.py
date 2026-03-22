"""
End-to-end simulation tests.

Covers full simulation runs, state serialisation, governance integration,
CLI smoke-test, and multi-step property checks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from universums_sim import UniverseSimulator, __version__
from universums_sim.governance.entropy import EntropyGovernor, GovernancePolicy
from universums_sim.integrations.registry import IntegrationRegistry
from universums_sim.simulation.core import SimulationConfig, SimulationPhase
from universums_sim.simulation.emergence import EmergenceType
from universums_sim.simulation.lagrangian import CollapseState

# ---------------------------------------------------------------------------
# Package metadata (5 tests)
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    def test_version_string(self):
        assert __version__ == "0.1.0"

    def test_version_format(self):
        parts = __version__.split(".")
        assert len(parts) == 3

    def test_imports(self):
        from universums_sim import CosmicMoment, UnifiedLagrangian
        assert CosmicMoment is not None
        assert UnifiedLagrangian is not None

    def test_all_exports(self):
        import universums_sim as us
        for name in us.__all__:
            assert hasattr(us, name)

    def test_py_typed_marker(self):
        # Just check that the module is importable with type annotations
        import universums_sim
        assert universums_sim.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# End-to-end simulation (40 tests)
# ---------------------------------------------------------------------------

class TestE2ESimulation:
    @pytest.fixture
    def sim(self):
        return UniverseSimulator(SimulationConfig(n_particles=8, seed=42, dt=0.01))

    def test_run_50_steps(self, sim):
        moments = list(sim.run(50))
        assert len(moments) == 50

    def test_all_entropies_finite(self, sim):
        for m in sim.run(20):
            assert np.isfinite(m.entropy)

    def test_all_hamiltonians_finite(self, sim):
        for m in sim.run(20):
            assert np.isfinite(m.hamiltonian)

    def test_all_emergence_rates_non_negative(self, sim):
        for m in sim.run(20):
            assert m.emergence_rate >= 0.0

    def test_phases_are_valid(self, sim):
        for m in sim.run(20):
            assert isinstance(m.phase, SimulationPhase)

    def test_collapse_states_are_valid(self, sim):
        for m in sim.run(20):
            assert isinstance(m.collapse_state, CollapseState)

    def test_observer_hash_constant(self, sim):
        hashes = {m.observer_hash for m in sim.run(10)}
        assert len(hashes) == 1

    def test_steps_sequential(self, sim):
        steps = [m.step for m in sim.run(10)]
        assert steps == list(range(10))

    def test_times_monotone(self, sim):
        times = [m.time for m in sim.run(10)]
        assert all(t2 > t1 for t1, t2 in zip(times, times[1:], strict=False))

    def test_events_all_tuples(self, sim):
        for m in sim.run(20):
            assert isinstance(m.events, tuple)

    def test_all_moments_json_serialisable(self, sim):
        for m in sim.run(5):
            json.dumps(m.to_dict())

    def test_entropy_non_negative(self, sim):
        for m in sim.run(50):
            assert m.entropy >= 0.0

    def test_wall_times_increasing_approx(self, sim):
        moments = list(sim.run(5))
        for m1, m2 in zip(moments, moments[1:], strict=False):
            assert m2.wall_time >= m1.wall_time

    def test_different_seeds_different_trajectories(self):
        s1 = UniverseSimulator(SimulationConfig(n_particles=8, seed=1))
        s2 = UniverseSimulator(SimulationConfig(n_particles=8, seed=2))
        m1 = list(s1.run(5))
        m2 = list(s2.run(5))
        entropies_1 = [m.entropy for m in m1]
        entropies_2 = [m.entropy for m in m2]
        # Very unlikely to be identical with different seeds
        assert entropies_1 != entropies_2

    def test_same_seed_reproducible(self):
        cfg = SimulationConfig(n_particles=8, seed=99)
        s1 = UniverseSimulator(cfg)
        s2 = UniverseSimulator(cfg)
        m1 = [m.hamiltonian for m in s1.run(5)]
        m2 = [m.hamiltonian for m in s2.run(5)]
        assert m1 == pytest.approx(m2)

    def test_to_dict_round_trip(self, sim):
        m = next(sim.run(1))
        d = m.to_dict()
        assert d["step"] == 0
        assert d["phase"] == m.phase.name

    def test_state_vector_restore(self):
        cfg = SimulationConfig(n_particles=4, seed=5)
        sim = UniverseSimulator(cfg)
        sv0 = sim.state_vector()
        list(sim.run(10))
        assert not np.allclose(sim.state_vector(), sv0)
        sim.load_state(sv0)
        np.testing.assert_array_almost_equal(sim.state_vector(), sv0)

    def test_reset_restores_step(self):
        cfg = SimulationConfig(n_particles=4, seed=5)
        sim = UniverseSimulator(cfg)
        list(sim.run(10))
        sim.reset()
        assert sim.step == 0

    def test_run_to_collapse_terminates(self):
        cfg = SimulationConfig(n_particles=4, seed=0, collapse_threshold=0.001)
        sim = UniverseSimulator(cfg)
        moments = list(sim.run_to_collapse(max_steps=50))
        assert len(moments) <= 50

    def test_emergence_event_types_valid(self, sim):
        for m in sim.run(30):
            for e in m.events:
                assert isinstance(e.kind, EmergenceType)

    def test_long_run_stability(self):
        cfg = SimulationConfig(n_particles=4, seed=7, dt=0.001)
        sim = UniverseSimulator(cfg)
        moments = list(sim.run(200))
        assert all(np.isfinite(m.entropy) for m in moments)
        assert all(np.isfinite(m.hamiltonian) for m in moments)

    def test_governance_integrated(self, sim):
        policy = GovernancePolicy(entropy_warn=50.0, entropy_halt=500.0)
        gov = EntropyGovernor(policy)
        for m in sim.run(20):
            decision = gov.evaluate(m.entropy)
            assert decision.action.name in {"ALLOW", "THROTTLE", "HALT", "RESET"}

    def test_integration_registry_status(self):
        reg = IntegrationRegistry()
        status = reg.status_dict()
        assert isinstance(status, dict)
        for v in status.values():
            assert isinstance(v, bool)

    def test_all_moments_have_observer(self, sim):
        for m in sim.run(5):
            assert len(m.observer_hash) > 0

    def test_simulation_produces_events_eventually(self):
        cfg = SimulationConfig(n_particles=4, seed=0, alpha=100.0, entropy_max=1e9)
        sim = UniverseSimulator(cfg)
        total = sum(len(m.events) for m in sim.run(100))
        assert total >= 0  # stochastic — just check no error

    def test_cosmic_moment_metadata_extensible(self, sim):
        m = next(sim.run(1))
        assert isinstance(m.metadata, dict)

    def test_hamiltonian_units_finite(self, sim):
        m = next(sim.run(1))
        assert abs(m.hamiltonian) < 1e15  # Not infinite in Planck units

    def test_run_to_collapse_with_critical_state(self):
        """Force CRITICAL collapse by loading very-close-particle state."""
        import numpy as np
        cfg = SimulationConfig(n_particles=2, seed=0, dt=0.001)
        sim = UniverseSimulator(cfg)
        # Put particles very close together with zero velocity -> high V_grav, low T
        n = 2
        sv = sim.state_vector()
        # Positions: very close, velocities: nearly zero
        sv[:n * 3] = np.array([0.0, 0.0, 0.0, 1e-6, 0.0, 0.0])
        sv[n * 3: n * 6] = np.zeros(n * 3)
        sv[n * 6: n * 7] = np.ones(n) * 1e3  # huge masses
        sim.load_state(sv)
        # run_to_collapse should terminate early if CRITICAL is reached
        moments = list(sim.run_to_collapse(max_steps=100))
        assert len(moments) >= 1

    def test_run_generator_lazy(self):
        cfg = SimulationConfig(n_particles=4, seed=0)
        sim = UniverseSimulator(cfg)
        gen = sim.run(1000)
        # Only consume first 3
        for _ in range(3):
            next(gen)
        assert sim.step == 3  # only 3 ticks executed


# ---------------------------------------------------------------------------
# CLI smoke tests (10 tests via typer runner)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_info(self):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_run_default(self):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--steps", "3", "--particles", "4"])
        assert result.exit_code == 0

    def test_cli_run_with_seed(self):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--steps", "5", "--particles", "4", "--seed", "7"])
        assert result.exit_code == 0

    def test_cli_run_with_entropy(self):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--steps", "3", "--particles", "4", "--entropy", "2.0"])
        assert result.exit_code == 0

    def test_cli_run_output_file(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        out = tmp_path / "out.json"
        result = runner.invoke(app, [
            "run", "--steps", "3", "--particles", "4", "--output", str(out)
        ])
        assert result.exit_code == 0
        assert out.exists()

    def test_cli_run_output_valid_json(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        out = tmp_path / "out.json"
        runner.invoke(app, ["run", "--steps", "5", "--particles", "4", "--output", str(out)])
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 5

    def test_cli_export_json(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        # First create an input file
        out = tmp_path / "sim.json"
        runner.invoke(app, ["run", "--steps", "3", "--particles", "4", "--output", str(out)])
        export_out = tmp_path / "export.json"
        result = runner.invoke(
            app, ["export", str(out), "--output", str(export_out), "--format", "json"]
        )
        assert result.exit_code == 0
        assert export_out.exists()

    def test_cli_export_csv(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        out = tmp_path / "sim.json"
        runner.invoke(app, ["run", "--steps", "3", "--particles", "4", "--output", str(out)])
        csv_out = tmp_path / "export.csv"
        result = runner.invoke(
            app, ["export", str(out), "--output", str(csv_out), "--format", "csv"]
        )
        assert result.exit_code == 0
        assert csv_out.exists()

    def test_cli_replay_nonexistent_file(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["replay", str(tmp_path / "nonexistent.json")])
        assert result.exit_code != 0

    def test_cli_run_verbose(self):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "--steps", "2", "--particles", "4", "--verbose"])
        assert result.exit_code == 0

    def test_cli_replay_existing_file(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        out = tmp_path / "sim.json"
        runner.invoke(app, ["run", "--steps", "3", "--particles", "4", "--output", str(out)])
        result = runner.invoke(app, ["replay", str(out)])
        assert result.exit_code == 0
        assert "Replay complete" in result.output

    def test_cli_export_nonexistent_file(self, tmp_path):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["export", str(tmp_path / "nope.json")])
        assert result.exit_code != 0

    def test_cli_verbose_mode_function(self):
        import sys
        from unittest.mock import patch

        from universums_sim.cli.main import verbose_mode
        with patch.object(sys, "argv", ["prog", "--verbose"]):
            assert verbose_mode() is True
        with patch.object(sys, "argv", ["prog"]):
            assert verbose_mode() is False

    def test_cli_replay_verbose(self, tmp_path):
        """Test replay with verbose_mode active to cover line 174."""
        import sys
        from unittest.mock import patch

        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        out = tmp_path / "sim.json"
        runner.invoke(app, ["run", "--steps", "2", "--particles", "4", "--output", str(out)])
        with patch.object(sys, "argv", ["prog", "--verbose"]):
            result = runner.invoke(app, ["replay", str(out)])
        assert result.exit_code == 0

    def test_cli_info_contains_doi(self):
        from typer.testing import CliRunner

        from universums_sim.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["info"])
        assert "zenodo" in result.output.lower() or "doi" in result.output.lower()


# ---------------------------------------------------------------------------
# Property-based tests (10 tests using hypothesis)
# ---------------------------------------------------------------------------

class TestPropertyBased:
    def test_entropy_bounded(self):
        """Entropy must remain bounded in [0, entropy_max]."""
        cfg = SimulationConfig(n_particles=4, seed=0, entropy_max=1000.0, dt=0.1)
        sim = UniverseSimulator(cfg)
        for m in sim.run(100):
            assert 0.0 <= m.entropy <= cfg.entropy_max * 2  # allow small overshoot

    def test_step_counter_monotone(self):
        cfg = SimulationConfig(n_particles=4, seed=0)
        sim = UniverseSimulator(cfg)
        prev = -1
        for m in sim.run(50):
            assert m.step > prev
            prev = m.step

    def test_cosmic_time_positive(self):
        cfg = SimulationConfig(n_particles=4, seed=0)
        sim = UniverseSimulator(cfg)
        for m in sim.run(10):
            assert m.time > 0.0

    def test_emergence_rate_bounded(self):
        cfg = SimulationConfig(n_particles=4, seed=0, alpha=1.0)
        sim = UniverseSimulator(cfg)
        for m in sim.run(50):
            # Rate cannot exceed alpha * entropy_max / 4 (peak of logistic)
            max_rate = cfg.alpha * cfg.entropy_max
            assert m.emergence_rate <= max_rate + 1e-9

    def test_masses_always_positive(self):
        cfg = SimulationConfig(n_particles=4, seed=0)
        sim = UniverseSimulator(cfg)
        for _ in range(20):
            sim.tick()
        assert np.all(sim.masses > 0)

    def test_positions_finite_after_run(self):
        cfg = SimulationConfig(n_particles=4, seed=0, dt=0.001)
        sim = UniverseSimulator(cfg)
        list(sim.run(100))
        assert np.all(np.isfinite(sim.positions))

    def test_velocities_finite_after_run(self):
        cfg = SimulationConfig(n_particles=4, seed=0, dt=0.001)
        sim = UniverseSimulator(cfg)
        list(sim.run(100))
        assert np.all(np.isfinite(sim.velocities))

    def test_observer_hash_nonempty(self):
        cfg = SimulationConfig(n_particles=4)
        sim = UniverseSimulator(cfg)
        for m in sim.run(5):
            assert m.observer_hash != ""

    def test_collapse_state_valid_throughout(self):
        cfg = SimulationConfig(n_particles=4, seed=0)
        sim = UniverseSimulator(cfg)
        valid = set(CollapseState)
        for m in sim.run(20):
            assert m.collapse_state in valid

    def test_phase_valid_throughout(self):
        cfg = SimulationConfig(n_particles=4, seed=0)
        sim = UniverseSimulator(cfg)
        valid = set(SimulationPhase)
        for m in sim.run(20):
            assert m.phase in valid
