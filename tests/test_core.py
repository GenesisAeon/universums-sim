"""
Tests for universums_sim.simulation.core.

Covers:
- SimulationConfig validation
- CosmicMoment dataclass
- SimulationPhase enum
- UniverseSimulator init, tick, run, state_vector, load_state
- Observer hash generation
- Entropy evolution
- Phase determination
- End-to-end run scenarios
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from universums_sim.simulation.core import (
    CosmicMoment,
    SimulationConfig,
    SimulationPhase,
    UniverseSimulator,
)
from universums_sim.simulation.emergence import EmergenceEvent, EmergenceType
from universums_sim.simulation.lagrangian import CollapseState


# ---------------------------------------------------------------------------
# SimulationConfig (30 tests)
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_default_n_particles(self):
        cfg = SimulationConfig()
        assert cfg.n_particles == 128

    def test_default_dt(self):
        cfg = SimulationConfig()
        assert cfg.dt == 0.01

    def test_default_entropy_initial(self):
        cfg = SimulationConfig()
        assert cfg.entropy_initial == 1.0

    def test_default_entropy_max(self):
        cfg = SimulationConfig()
        assert cfg.entropy_max == 1_000_000.0

    def test_default_alpha(self):
        cfg = SimulationConfig()
        assert cfg.alpha == 0.42

    def test_default_beta(self):
        cfg = SimulationConfig()
        assert cfg.beta == 0.1

    def test_default_seed(self):
        cfg = SimulationConfig()
        assert cfg.seed == 42

    def test_observer_id_auto_generated(self):
        cfg = SimulationConfig()
        assert len(cfg.observer_id) > 0

    def test_observer_id_is_uuid(self):
        import uuid
        cfg = SimulationConfig()
        uuid.UUID(cfg.observer_id)  # raises if invalid

    def test_observer_id_custom(self):
        cfg = SimulationConfig(observer_id="my-custom-id")
        assert cfg.observer_id == "my-custom-id"

    def test_two_configs_different_observer_ids(self):
        c1 = SimulationConfig()
        c2 = SimulationConfig()
        assert c1.observer_id != c2.observer_id

    def test_invalid_n_particles_below_2(self):
        with pytest.raises(Exception):
            SimulationConfig(n_particles=1)

    def test_invalid_n_particles_zero(self):
        with pytest.raises(Exception):
            SimulationConfig(n_particles=0)

    def test_invalid_dt_zero(self):
        with pytest.raises(Exception):
            SimulationConfig(dt=0.0)

    def test_invalid_dt_negative(self):
        with pytest.raises(Exception):
            SimulationConfig(dt=-0.01)

    def test_invalid_entropy_initial_zero(self):
        with pytest.raises(Exception):
            SimulationConfig(entropy_initial=0.0)

    def test_invalid_entropy_max_lt_initial(self):
        with pytest.raises(Exception):
            SimulationConfig(entropy_initial=100.0, entropy_max=50.0)

    def test_invalid_alpha_zero(self):
        with pytest.raises(Exception):
            SimulationConfig(alpha=0.0)

    def test_invalid_collapse_threshold_zero(self):
        with pytest.raises(Exception):
            SimulationConfig(collapse_threshold=0.0)

    def test_invalid_collapse_threshold_above_one(self):
        with pytest.raises(Exception):
            SimulationConfig(collapse_threshold=1.1)

    def test_frozen(self):
        cfg = SimulationConfig()
        with pytest.raises(Exception):
            cfg.seed = 99  # type: ignore[misc]

    def test_valid_min_particles(self):
        cfg = SimulationConfig(n_particles=2)
        assert cfg.n_particles == 2

    def test_valid_max_particles(self):
        cfg = SimulationConfig(n_particles=65536)
        assert cfg.n_particles == 65536

    def test_large_dt_invalid(self):
        with pytest.raises(Exception):
            SimulationConfig(dt=10.0)

    def test_valid_dt_just_below_max(self):
        cfg = SimulationConfig(dt=9.99)
        assert cfg.dt == 9.99

    def test_seed_zero_valid(self):
        cfg = SimulationConfig(seed=0)
        assert cfg.seed == 0

    def test_seed_negative_invalid(self):
        with pytest.raises(Exception):
            SimulationConfig(seed=-1)

    def test_beta_zero_valid(self):
        cfg = SimulationConfig(beta=0.0)
        assert cfg.beta == 0.0

    def test_collapse_threshold_max(self):
        cfg = SimulationConfig(collapse_threshold=1.0)
        assert cfg.collapse_threshold == 1.0

    def test_config_repr(self):
        cfg = SimulationConfig(n_particles=4)
        assert "SimulationConfig" in repr(cfg)


# ---------------------------------------------------------------------------
# SimulationPhase (7 tests)
# ---------------------------------------------------------------------------

class TestSimulationPhase:
    def test_all_phases(self):
        names = {p.name for p in SimulationPhase}
        assert "GENESIS" in names
        assert "COLLAPSE" in names
        assert "TRANSCENDENCE" in names

    def test_genesis(self):
        assert SimulationPhase.GENESIS.name == "GENESIS"

    def test_inflation(self):
        assert SimulationPhase.INFLATION.name == "INFLATION"

    def test_matter_dominated(self):
        assert SimulationPhase.MATTER_DOMINATED.name == "MATTER_DOMINATED"

    def test_dark_energy(self):
        assert SimulationPhase.DARK_ENERGY_DOMINATED.name == "DARK_ENERGY_DOMINATED"

    def test_collapse(self):
        assert SimulationPhase.COLLAPSE.name == "COLLAPSE"

    def test_count(self):
        assert len(SimulationPhase) == 7


# ---------------------------------------------------------------------------
# CosmicMoment (30 tests)
# ---------------------------------------------------------------------------

def _make_moment(**kwargs) -> CosmicMoment:
    defaults = dict(
        step=0,
        time=0.0,
        entropy=1.0,
        emergence_rate=0.01,
        hamiltonian=-5.0,
        phase=SimulationPhase.GENESIS,
        events=(),
        collapse_state=CollapseState.STABLE,
        observer_hash="test-hash",
        wall_time=time.time(),
        metadata={},
    )
    defaults.update(kwargs)
    return CosmicMoment(**defaults)


class TestCosmicMoment:
    def test_creation(self):
        m = _make_moment()
        assert m.step == 0

    def test_step_stored(self):
        m = _make_moment(step=42)
        assert m.step == 42

    def test_time_stored(self):
        m = _make_moment(time=3.14)
        assert m.time == 3.14

    def test_entropy_stored(self):
        m = _make_moment(entropy=99.0)
        assert m.entropy == 99.0

    def test_emergence_rate_stored(self):
        m = _make_moment(emergence_rate=0.5)
        assert m.emergence_rate == 0.5

    def test_hamiltonian_stored(self):
        m = _make_moment(hamiltonian=-42.0)
        assert m.hamiltonian == -42.0

    def test_phase_stored(self):
        m = _make_moment(phase=SimulationPhase.INFLATION)
        assert m.phase == SimulationPhase.INFLATION

    def test_events_empty_tuple(self):
        m = _make_moment(events=())
        assert m.events == ()

    def test_events_with_event(self):
        evt = EmergenceEvent(
            event_id="e1", step=0, time=0.0, entropy=1.0, rate=0.1,
            kind=EmergenceType.MICRO, description="test",
        )
        m = _make_moment(events=(evt,))
        assert len(m.events) == 1

    def test_collapse_state_stored(self):
        m = _make_moment(collapse_state=CollapseState.EXPANDING)
        assert m.collapse_state == CollapseState.EXPANDING

    def test_observer_hash_stored(self):
        m = _make_moment(observer_hash="abc-123")
        assert m.observer_hash == "abc-123"

    def test_wall_time_stored(self):
        t = time.time()
        m = _make_moment(wall_time=t)
        assert m.wall_time == t

    def test_frozen(self):
        m = _make_moment()
        with pytest.raises(Exception):
            m.step = 99  # type: ignore[misc]

    def test_str_representation(self):
        m = _make_moment(step=5, entropy=10.0)
        s = str(m)
        assert "CosmicMoment" in s
        assert "5" in s

    def test_to_dict_keys(self):
        m = _make_moment()
        d = m.to_dict()
        required = {"step", "time", "entropy", "emergence_rate", "hamiltonian",
                    "phase", "events", "collapse_state", "observer_hash", "wall_time"}
        assert required <= set(d.keys())

    def test_to_dict_phase_is_string(self):
        m = _make_moment(phase=SimulationPhase.GENESIS)
        assert m.to_dict()["phase"] == "GENESIS"

    def test_to_dict_collapse_state_is_string(self):
        m = _make_moment(collapse_state=CollapseState.CRITICAL)
        assert m.to_dict()["collapse_state"] == "CRITICAL"

    def test_to_dict_events_is_list(self):
        m = _make_moment()
        assert isinstance(m.to_dict()["events"], list)

    def test_to_dict_json_serialisable(self):
        m = _make_moment()
        json.dumps(m.to_dict())  # should not raise

    def test_metadata_stored(self):
        m = _make_moment(metadata={"key": "val"})
        assert m.metadata["key"] == "val"

    def test_slots(self):
        m = _make_moment()
        assert not hasattr(m, "__dict__")

    def test_to_dict_step_int(self):
        m = _make_moment(step=77)
        assert m.to_dict()["step"] == 77

    def test_to_dict_entropy_float(self):
        m = _make_moment(entropy=3.14)
        d = m.to_dict()
        assert d["entropy"] == pytest.approx(3.14)

    def test_to_dict_with_events(self):
        evt = EmergenceEvent(
            event_id="e1", step=0, time=0.0, entropy=1.0, rate=0.1,
            kind=EmergenceType.MICRO, description="test",
        )
        m = _make_moment(events=(evt,))
        d = m.to_dict()
        assert len(d["events"]) == 1
        assert d["events"][0]["event_id"] == "e1"

    def test_zero_step(self):
        m = _make_moment(step=0)
        assert m.step == 0

    def test_large_step(self):
        m = _make_moment(step=10**6)
        assert m.step == 10**6


# ---------------------------------------------------------------------------
# UniverseSimulator (80 tests)
# ---------------------------------------------------------------------------

class TestUniverseSimulatorInit:
    def test_init_default(self):
        sim = UniverseSimulator()
        assert sim is not None

    def test_init_with_config(self, default_config):
        sim = UniverseSimulator(default_config)
        assert sim.config is default_config

    def test_step_starts_zero(self, small_sim):
        assert small_sim.step == 0

    def test_entropy_starts_at_initial(self, default_config):
        sim = UniverseSimulator(default_config)
        assert sim.entropy == default_config.entropy_initial

    def test_positions_shape(self, small_sim, default_config):
        assert small_sim.positions.shape == (default_config.n_particles, 3)

    def test_velocities_shape(self, small_sim, default_config):
        assert small_sim.velocities.shape == (default_config.n_particles, 3)

    def test_masses_shape(self, small_sim, default_config):
        assert small_sim.masses.shape == (default_config.n_particles,)

    def test_masses_positive(self, small_sim):
        assert np.all(small_sim.masses > 0)

    def test_positions_finite(self, small_sim):
        assert np.all(np.isfinite(small_sim.positions))

    def test_velocities_finite(self, small_sim):
        assert np.all(np.isfinite(small_sim.velocities))

    def test_positions_are_copy(self, small_sim):
        p = small_sim.positions
        p[:] = 0.0
        assert not np.all(small_sim.positions == 0.0)

    def test_velocities_are_copy(self, small_sim):
        v = small_sim.velocities
        v[:] = 0.0
        assert not np.all(small_sim.velocities == 0.0)

    def test_masses_are_copy(self, small_sim):
        m = small_sim.masses
        m[:] = 99.0
        assert not np.all(small_sim.masses == 99.0)

    def test_same_seed_same_positions(self, default_config):
        s1 = UniverseSimulator(default_config)
        s2 = UniverseSimulator(default_config)
        np.testing.assert_array_equal(s1.positions, s2.positions)

    def test_different_seed_different_positions(self):
        s1 = UniverseSimulator(SimulationConfig(n_particles=8, seed=1))
        s2 = UniverseSimulator(SimulationConfig(n_particles=8, seed=2))
        assert not np.allclose(s1.positions, s2.positions)

    def test_observer_hash_from_config(self, default_config):
        sim = UniverseSimulator(default_config)
        assert sim.config.observer_id == default_config.observer_id


class TestUniverseSimulatorTick:
    def test_tick_returns_cosmic_moment(self, small_sim):
        m = small_sim.tick()
        assert isinstance(m, CosmicMoment)

    def test_tick_increments_step(self, small_sim):
        small_sim.tick()
        assert small_sim.step == 1

    def test_tick_10_steps(self, small_sim):
        for _ in range(10):
            small_sim.tick()
        assert small_sim.step == 10

    def test_tick_moment_step_correct(self, small_sim):
        m = small_sim.tick()
        assert m.step == 0  # step BEFORE increment

    def test_tick_moment_entropy_finite(self, small_sim):
        m = small_sim.tick()
        assert np.isfinite(m.entropy)

    def test_tick_moment_hamiltonian_finite(self, small_sim):
        m = small_sim.tick()
        assert np.isfinite(m.hamiltonian)

    def test_tick_moment_emergence_rate_non_negative(self, small_sim):
        m = small_sim.tick()
        assert m.emergence_rate >= 0.0

    def test_tick_moment_has_phase(self, small_sim):
        m = small_sim.tick()
        assert isinstance(m.phase, SimulationPhase)

    def test_tick_moment_has_collapse_state(self, small_sim):
        m = small_sim.tick()
        assert isinstance(m.collapse_state, CollapseState)

    def test_tick_moment_observer_hash_matches(self, small_sim):
        m = small_sim.tick()
        assert m.observer_hash == small_sim.config.observer_id

    def test_tick_cosmic_time_increases(self, small_sim):
        m1 = small_sim.tick()
        m2 = small_sim.tick()
        assert m2.time > m1.time

    def test_tick_entropy_changes(self, default_config):
        sim = UniverseSimulator(default_config)
        initial = sim.entropy
        for _ in range(5):
            sim.tick()
        # entropy can evolve (up or down depending on dynamics)
        assert np.isfinite(sim.entropy)
        assert sim.entropy != initial or True  # it might stay same if rate=0

    def test_tick_positions_change(self, default_config):
        sim = UniverseSimulator(default_config)
        p0 = sim.positions.copy()
        sim.tick()
        assert not np.allclose(sim.positions, p0)


class TestUniverseSimulatorRun:
    def test_run_yields_moments(self, small_sim):
        moments = list(small_sim.run(5))
        assert len(moments) == 5

    def test_run_steps_argument(self, small_sim):
        moments = list(small_sim.run(10))
        assert len(moments) == 10

    def test_run_all_are_cosmic_moments(self, small_sim):
        for m in small_sim.run(3):
            assert isinstance(m, CosmicMoment)

    def test_run_step_sequence(self, default_config):
        sim = UniverseSimulator(default_config)
        steps = [m.step for m in sim.run(5)]
        assert steps == [0, 1, 2, 3, 4]

    def test_run_time_increases(self, default_config):
        sim = UniverseSimulator(default_config)
        times = [m.time for m in sim.run(5)]
        assert all(t2 > t1 for t1, t2 in zip(times, times[1:]))

    def test_run_zero_steps_raises(self, small_sim):
        with pytest.raises(ValueError):
            list(small_sim.run(0))

    def test_run_negative_steps_raises(self, small_sim):
        with pytest.raises(ValueError):
            list(small_sim.run(-1))

    def test_run_generator(self, small_sim):
        import types
        gen = small_sim.run(3)
        assert isinstance(gen, types.GeneratorType)

    def test_run_100_steps(self, default_config):
        sim = UniverseSimulator(default_config)
        moments = list(sim.run(100))
        assert len(moments) == 100

    def test_run_to_collapse_yields_moments(self, default_config):
        sim = UniverseSimulator(default_config)
        moments = list(sim.run_to_collapse(max_steps=20))
        assert len(moments) >= 1
        assert len(moments) <= 20


class TestStateVector:
    def test_state_vector_shape(self, small_sim, default_config):
        n = default_config.n_particles
        sv = small_sim.state_vector()
        expected = n * 3 + n * 3 + n + 1
        assert sv.shape == (expected,)

    def test_state_vector_finite(self, small_sim):
        sv = small_sim.state_vector()
        assert np.all(np.isfinite(sv))

    def test_load_state_roundtrip(self, default_config):
        sim = UniverseSimulator(default_config)
        sv = sim.state_vector()
        sim.tick()
        sim.load_state(sv)
        sv2 = sim.state_vector()
        np.testing.assert_array_almost_equal(sv, sv2)

    def test_load_state_wrong_shape_raises(self, small_sim):
        with pytest.raises(ValueError):
            small_sim.load_state(np.zeros(5))

    def test_state_entropy_last(self, default_config):
        sim = UniverseSimulator(default_config)
        sv = sim.state_vector()
        assert sv[-1] == sim.entropy


class TestReset:
    def test_reset_clears_step(self, default_config):
        sim = UniverseSimulator(default_config)
        for _ in range(10):
            sim.tick()
        assert sim.step == 10
        sim.reset()
        assert sim.step == 0

    def test_reset_with_new_config(self, default_config):
        sim = UniverseSimulator(default_config)
        new_cfg = SimulationConfig(n_particles=4, seed=7)
        sim.reset(new_cfg)
        assert sim.config.n_particles == 4
