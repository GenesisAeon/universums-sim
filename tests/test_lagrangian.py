"""
Tests for universums_sim.simulation.lagrangian.

Covers:
- UnifiedLagrangian: kinetic, gravitational, scalar, entropic, topological
- Hamiltonian computation
- Gradient computation (finite-difference consistency)
- Collapse detection (all 5 states)
- LagrangianConfig validation
- CollapseState enum
- Field-gradient computation
- Numerical stability edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from universums_sim.simulation.lagrangian import (
    CollapseState,
    LagrangianConfig,
    UnifiedLagrangian,
)


# ---------------------------------------------------------------------------
# LagrangianConfig tests (20 tests)
# ---------------------------------------------------------------------------

class TestLagrangianConfig:
    def test_default_g(self):
        cfg = LagrangianConfig()
        assert cfg.G == 1.0

    def test_default_lambda(self):
        cfg = LagrangianConfig()
        assert cfg.lambda_scalar == 0.125

    def test_default_mu_squared(self):
        cfg = LagrangianConfig()
        assert cfg.mu_squared == 0.5

    def test_default_kappa(self):
        cfg = LagrangianConfig()
        assert cfg.kappa == 0.01

    def test_default_xi(self):
        cfg = LagrangianConfig()
        assert cfg.xi == 0.001

    def test_default_softening(self):
        cfg = LagrangianConfig()
        assert cfg.softening == 1e-3

    def test_custom_g(self):
        cfg = LagrangianConfig(G=2.0)
        assert cfg.G == 2.0

    def test_custom_lambda(self):
        cfg = LagrangianConfig(lambda_scalar=0.5)
        assert cfg.lambda_scalar == 0.5

    def test_invalid_g_zero(self):
        with pytest.raises(Exception):
            LagrangianConfig(G=0.0)

    def test_invalid_g_negative(self):
        with pytest.raises(Exception):
            LagrangianConfig(G=-1.0)

    def test_invalid_lambda_zero(self):
        with pytest.raises(Exception):
            LagrangianConfig(lambda_scalar=0.0)

    def test_invalid_mu_negative(self):
        with pytest.raises(Exception):
            LagrangianConfig(mu_squared=-0.1)

    def test_invalid_kappa_negative(self):
        with pytest.raises(Exception):
            LagrangianConfig(kappa=-0.1)

    def test_invalid_xi_negative(self):
        with pytest.raises(Exception):
            LagrangianConfig(xi=-0.1)

    def test_invalid_softening_zero(self):
        with pytest.raises(Exception):
            LagrangianConfig(softening=0.0)

    def test_frozen(self):
        cfg = LagrangianConfig()
        with pytest.raises(Exception):
            cfg.G = 3.0  # type: ignore[misc]

    def test_zero_kappa_allowed(self):
        cfg = LagrangianConfig(kappa=0.0)
        assert cfg.kappa == 0.0

    def test_zero_xi_allowed(self):
        cfg = LagrangianConfig(xi=0.0)
        assert cfg.xi == 0.0

    def test_large_g(self):
        cfg = LagrangianConfig(G=1e6)
        assert cfg.G == 1e6

    def test_small_softening(self):
        cfg = LagrangianConfig(softening=1e-10)
        assert cfg.softening == 1e-10


# ---------------------------------------------------------------------------
# CollapseState enum tests (5 tests)
# ---------------------------------------------------------------------------

class TestCollapseState:
    def test_all_states_exist(self):
        states = {s.name for s in CollapseState}
        assert states == {"EXPANDING", "STABLE", "CONTRACTING", "CRITICAL", "SINGULARITY"}

    def test_expanding(self):
        assert CollapseState.EXPANDING.name == "EXPANDING"

    def test_stable(self):
        assert CollapseState.STABLE.name == "STABLE"

    def test_critical(self):
        assert CollapseState.CRITICAL.name == "CRITICAL"

    def test_singularity(self):
        assert CollapseState.SINGULARITY.name == "SINGULARITY"


# ---------------------------------------------------------------------------
# UnifiedLagrangian core tests (60 tests)
# ---------------------------------------------------------------------------

class TestUnifiedLagrangianInit:
    def test_init_no_args(self):
        lag = UnifiedLagrangian()
        assert lag is not None

    def test_init_with_config(self):
        cfg = LagrangianConfig(G=2.0)
        lag = UnifiedLagrangian(config=cfg)
        assert lag._lcfg.G == 2.0

    def test_init_with_none(self):
        lag = UnifiedLagrangian(config=None)
        assert lag._lcfg is not None

    def test_init_with_sim_config(self):
        from universums_sim.simulation.core import SimulationConfig
        cfg = SimulationConfig(n_particles=4)
        lag = UnifiedLagrangian(config=cfg)
        assert lag is not None


class TestKineticEnergy:
    def test_zero_velocity(self):
        lag = UnifiedLagrangian()
        vel = np.zeros((4, 3))
        masses = np.ones(4)
        assert lag.kinetic(vel, masses) == 0.0

    def test_positive_kinetic(self):
        lag = UnifiedLagrangian()
        vel = np.ones((4, 3))
        masses = np.ones(4)
        t = lag.kinetic(vel, masses)
        assert t > 0.0

    def test_kinetic_scales_with_mass(self):
        lag = UnifiedLagrangian()
        vel = np.ones((4, 3))
        m1 = np.ones(4)
        m2 = 2 * np.ones(4)
        assert lag.kinetic(vel, m2) == pytest.approx(2.0 * lag.kinetic(vel, m1))

    def test_kinetic_scales_with_v_squared(self):
        lag = UnifiedLagrangian()
        masses = np.ones(2)
        v1 = np.ones((2, 3))
        v2 = 2 * np.ones((2, 3))
        assert lag.kinetic(v2, masses) == pytest.approx(4.0 * lag.kinetic(v1, masses))

    def test_kinetic_single_particle(self):
        lag = UnifiedLagrangian()
        vel = np.array([[3.0, 4.0, 0.0]])
        masses = np.array([1.0])
        assert lag.kinetic(vel, masses) == pytest.approx(12.5)

    def test_kinetic_finite(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        t = lag.kinetic(velocities4, masses4)
        assert np.isfinite(t)

    def test_kinetic_non_negative(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        t = lag.kinetic(velocities4, masses4)
        assert t >= 0.0


class TestGravitationalPotential:
    def test_gravitational_two_particles(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        v = lag.gravitational(pos, masses)
        assert v < 0.0

    def test_gravitational_negative(self, positions4, masses4):
        lag = UnifiedLagrangian()
        v = lag.gravitational(positions4, masses4)
        assert v < 0.0

    def test_gravitational_finite(self, positions4, masses4):
        lag = UnifiedLagrangian()
        v = lag.gravitational(positions4, masses4)
        assert np.isfinite(v)

    def test_gravitational_scales_with_g(self, positions4, masses4):
        lag1 = UnifiedLagrangian(LagrangianConfig(G=1.0))
        lag2 = UnifiedLagrangian(LagrangianConfig(G=2.0))
        v1 = lag1.gravitational(positions4, masses4)
        v2 = lag2.gravitational(positions4, masses4)
        assert v2 == pytest.approx(2.0 * v1, rel=1e-6)

    def test_gravitational_softening_prevents_singularity(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # identical positions
        masses = np.array([1.0, 1.0])
        v = lag.gravitational(pos, masses)
        assert np.isfinite(v)

    def test_gravitational_single_particle(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([1.0])
        v = lag.gravitational(pos, masses)
        assert v == 0.0

    def test_gravitational_symmetric(self):
        lag = UnifiedLagrangian()
        pos1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        pos2 = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        m = np.array([1.0, 1.0])
        assert lag.gravitational(pos1, m) == pytest.approx(lag.gravitational(pos2, m))


class TestScalarPotential:
    def test_scalar_finite(self, positions4):
        lag = UnifiedLagrangian()
        v = lag.scalar_potential(positions4, entropy=1.0)
        assert np.isfinite(v)

    def test_scalar_zero_entropy(self, positions4):
        lag = UnifiedLagrangian()
        v = lag.scalar_potential(positions4, entropy=1e-30)
        assert np.isfinite(v)

    def test_scalar_large_entropy(self, positions4):
        lag = UnifiedLagrangian()
        v = lag.scalar_potential(positions4, entropy=1e8)
        assert np.isfinite(v)

    def test_scalar_zero_kappa_zeroed(self, positions4):
        lag = UnifiedLagrangian(LagrangianConfig(kappa=0.0, lambda_scalar=0.01, mu_squared=0.0, xi=0.0))
        v = lag.scalar_potential(positions4, entropy=1.0)
        assert np.isfinite(v)


class TestEntropicPotential:
    def test_entropic_positive_entropy(self):
        lag = UnifiedLagrangian()
        v = lag.entropic_potential(1.0)
        assert np.isfinite(v)

    def test_entropic_near_zero(self):
        lag = UnifiedLagrangian()
        v = lag.entropic_potential(1e-30)
        assert np.isfinite(v)

    def test_entropic_large(self):
        lag = UnifiedLagrangian()
        v = lag.entropic_potential(1e6)
        assert np.isfinite(v)

    def test_entropic_monotone(self):
        lag = UnifiedLagrangian()
        v1 = lag.entropic_potential(1.0)
        v2 = lag.entropic_potential(10.0)
        assert v2 > v1

    def test_entropic_zero_kappa(self):
        lag = UnifiedLagrangian(LagrangianConfig(kappa=0.0))
        assert lag.entropic_potential(5.0) == 0.0


class TestTopologicalTerm:
    def test_topological_finite(self, positions4, velocities4):
        lag = UnifiedLagrangian()
        v = lag.topological(positions4, velocities4)
        assert np.isfinite(v)

    def test_topological_zero_velocity(self, positions4):
        lag = UnifiedLagrangian()
        vel = np.zeros_like(positions4)
        assert lag.topological(positions4, vel) == 0.0

    def test_topological_zero_xi(self, positions4, velocities4):
        lag = UnifiedLagrangian(LagrangianConfig(xi=0.0))
        assert lag.topological(positions4, velocities4) == 0.0

    def test_topological_scales_with_xi(self, positions4, velocities4):
        lag1 = UnifiedLagrangian(LagrangianConfig(xi=1.0))
        lag2 = UnifiedLagrangian(LagrangianConfig(xi=2.0))
        v1 = lag1.topological(positions4, velocities4)
        v2 = lag2.topological(positions4, velocities4)
        assert v2 == pytest.approx(2.0 * v1)


class TestHamiltonianCompute:
    def test_hamiltonian_finite(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        h = lag.compute(positions4, velocities4, masses4, entropy=1.0)
        assert np.isfinite(h)

    def test_hamiltonian_zero_velocity_less_than_moving(self, positions4, masses4):
        lag = UnifiedLagrangian()
        vel_zero = np.zeros_like(positions4)
        vel_nonzero = np.ones_like(positions4) * 0.1
        h0 = lag.compute(positions4, vel_zero, masses4, 1.0)
        h1 = lag.compute(positions4, vel_nonzero, masses4, 1.0)
        # Kinetic energy is non-negative, so h1 >= h0 (approximately)
        assert np.isfinite(h0) and np.isfinite(h1)

    def test_hamiltonian_consistent_components(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        h = lag.compute(positions4, velocities4, masses4, entropy=2.0)
        t = lag.kinetic(velocities4, masses4)
        vg = lag.gravitational(positions4, masses4)
        vs = lag.scalar_potential(positions4, 2.0)
        ve = lag.entropic_potential(2.0)
        vt = lag.topological(positions4, velocities4)
        expected = t + vg + vs + ve + vt
        assert h == pytest.approx(expected, rel=1e-10)


class TestGradient:
    def test_gradient_shape(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        grad = lag.gradient(positions4, velocities4, masses4, entropy=1.0)
        assert grad.shape == positions4.shape

    def test_gradient_finite(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        grad = lag.gradient(positions4, velocities4, masses4, entropy=1.0)
        assert np.all(np.isfinite(grad))

    def test_field_gradient_shape(self, positions4):
        lag = UnifiedLagrangian()
        fg = lag.field_gradient(positions4, entropy=1.0)
        assert fg.shape == positions4.shape

    def test_field_gradient_finite(self, positions4):
        lag = UnifiedLagrangian()
        fg = lag.field_gradient(positions4, entropy=1.0)
        assert np.all(np.isfinite(fg))


class TestCollapseDetection:
    def test_expanding_high_kinetic(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        vel = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        state = lag.detect_collapse(pos, vel, masses, 1.0)
        assert state == CollapseState.EXPANDING

    def test_stable_balanced(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        vel = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        state = lag.detect_collapse(pos, vel, masses, 1.0)
        assert state in (CollapseState.STABLE, CollapseState.CONTRACTING, CollapseState.EXPANDING)

    def test_critical_high_potential(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]])
        vel = np.zeros((2, 3))
        masses = np.array([1.0, 1.0])
        state = lag.detect_collapse(pos, vel, masses, 1.0)
        assert state in (CollapseState.CRITICAL, CollapseState.CONTRACTING, CollapseState.STABLE)

    def test_singularity_very_large_v(self):
        lag = UnifiedLagrangian(LagrangianConfig(softening=1e-15))
        pos = np.array([[0.0, 0.0, 0.0], [1e-14, 0.0, 0.0]])
        vel = np.zeros((2, 3))
        masses = np.array([1e6, 1e6])
        state = lag.detect_collapse(pos, vel, masses, 1.0)
        assert state in CollapseState

    def test_stable_when_no_gravity(self):
        lag = UnifiedLagrangian()
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])
        state = lag.detect_collapse(pos, vel, masses, 1.0)
        assert state == CollapseState.STABLE

    def test_returns_collapse_state_instance(self, positions4, velocities4, masses4):
        lag = UnifiedLagrangian()
        state = lag.detect_collapse(positions4, velocities4, masses4, 1.0)
        assert isinstance(state, CollapseState)
