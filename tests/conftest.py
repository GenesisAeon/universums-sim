"""Shared pytest fixtures for universums-sim tests."""

from __future__ import annotations

import numpy as np
import pytest

from universums_sim.governance.entropy import EntropyGovernor, GovernancePolicy
from universums_sim.simulation.core import SimulationConfig, UniverseSimulator
from universums_sim.simulation.emergence import EmergenceEngine
from universums_sim.simulation.lagrangian import UnifiedLagrangian

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_config() -> SimulationConfig:
    """Minimal config for fast tests."""
    return SimulationConfig(n_particles=8, dt=0.01, seed=0)


@pytest.fixture(scope="session")
def medium_config() -> SimulationConfig:
    """Medium-size config."""
    return SimulationConfig(n_particles=32, dt=0.01, seed=42)


@pytest.fixture(scope="session")
def large_config() -> SimulationConfig:
    """Larger config for integration tests."""
    return SimulationConfig(n_particles=64, dt=0.005, seed=99)


# ---------------------------------------------------------------------------
# Simulator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_sim(default_config: SimulationConfig) -> UniverseSimulator:
    """Fresh small simulator (8 particles)."""
    return UniverseSimulator(default_config)


@pytest.fixture
def medium_sim(medium_config: SimulationConfig) -> UniverseSimulator:
    """Fresh medium simulator (32 particles)."""
    return UniverseSimulator(medium_config)


# ---------------------------------------------------------------------------
# Lagrangian fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lagrangian() -> UnifiedLagrangian:
    """Shared UnifiedLagrangian with default config."""
    return UnifiedLagrangian()


@pytest.fixture(scope="session")
def positions4() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((4, 3))


@pytest.fixture(scope="session")
def velocities4() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.standard_normal((4, 3)) * 0.1


@pytest.fixture(scope="session")
def masses4() -> np.ndarray:
    return np.array([1.0, 0.5, 2.0, 1.5])


# ---------------------------------------------------------------------------
# EmergenceEngine fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def emergence_engine() -> EmergenceEngine:
    rng = np.random.default_rng(77)
    return EmergenceEngine(alpha=0.5, beta=0.1, entropy_max=1e4, rng=rng)


# ---------------------------------------------------------------------------
# Governance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_policy() -> GovernancePolicy:
    return GovernancePolicy(entropy_warn=100.0, entropy_halt=1000.0, entropy_reset=10000.0)


@pytest.fixture
def governor(default_policy: GovernancePolicy) -> EntropyGovernor:
    return EntropyGovernor(default_policy)
