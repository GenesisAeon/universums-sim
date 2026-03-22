"""
Tests for the universums_sim top-level package.

Covers __init__.py exports, version, author, DOI, and import stability.
"""

from __future__ import annotations

import pytest


class TestPackageInit:
    def test_version_importable(self):
        from universums_sim import __version__
        assert __version__ == "0.1.0"

    def test_author_importable(self):
        from universums_sim import __author__
        assert __author__ == "GenesisAeon"

    def test_license_importable(self):
        from universums_sim import __license__
        assert __license__ == "MIT"

    def test_email_importable(self):
        from universums_sim import __email__
        assert "@" in __email__

    def test_doi_importable(self):
        from universums_sim import __doi__
        assert "zenodo" in __doi__

    def test_universe_simulator_importable(self):
        from universums_sim import UniverseSimulator
        assert UniverseSimulator is not None

    def test_cosmic_moment_importable(self):
        from universums_sim import CosmicMoment
        assert CosmicMoment is not None

    def test_unified_lagrangian_importable(self):
        from universums_sim import UnifiedLagrangian
        assert UnifiedLagrangian is not None

    def test_collapse_state_importable(self):
        from universums_sim import CollapseState
        assert CollapseState is not None

    def test_all_list_complete(self):
        import universums_sim as us
        expected = {"CosmicMoment", "UniverseSimulator", "UnifiedLagrangian",
                    "CollapseState", "__version__"}
        assert expected <= set(us.__all__)

    def test_simulation_subpackage(self):
        from universums_sim import simulation
        assert simulation is not None

    def test_cli_subpackage(self):
        from universums_sim import cli
        assert cli is not None

    def test_visualization_subpackage(self):
        from universums_sim import visualization
        assert visualization is not None

    def test_governance_subpackage(self):
        from universums_sim import governance
        assert governance is not None

    def test_integrations_subpackage(self):
        from universums_sim import integrations
        assert integrations is not None

    def test_simulation_init_exports(self):
        from universums_sim.simulation import (
            CosmicMoment,
            EmergenceEngine,
            EmergenceEvent,
            SimulationConfig,
            UnifiedLagrangian,
            UniverseSimulator,
            CollapseState,
        )
        assert all(x is not None for x in [
            CosmicMoment, EmergenceEngine, EmergenceEvent,
            SimulationConfig, UnifiedLagrangian, UniverseSimulator, CollapseState,
        ])

    def test_governance_init_exports(self):
        from universums_sim.governance import EntropyGovernor, GovernancePolicy
        assert EntropyGovernor is not None
        assert GovernancePolicy is not None

    def test_visualization_init_exports(self):
        from universums_sim.visualization import LiveVisualizer, SonificationEngine
        assert LiveVisualizer is not None
        assert SonificationEngine is not None

    def test_integrations_init_exports(self):
        from universums_sim.integrations import IntegrationRegistry
        assert IntegrationRegistry is not None

    def test_instantiate_simulator_from_top_level_import(self):
        from universums_sim import UniverseSimulator
        sim = UniverseSimulator()
        assert sim is not None

    def test_run_one_step_from_top_level(self):
        from universums_sim import CosmicMoment, UniverseSimulator
        sim = UniverseSimulator()
        m = next(sim.run(1))
        assert isinstance(m, CosmicMoment)
