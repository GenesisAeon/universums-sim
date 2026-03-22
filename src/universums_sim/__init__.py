"""
universums-sim v0.1.0 — Complete cosmic emergence simulation for GenesisAeon.

Provides:
- UniverseSimulator: full N-body + field-theory simulation engine
- UnifiedLagrangian: extended Lagrangian for cosmic scales with collapse detection
- CosmicMoment: timestamped emergence events
- CLI (Typer): universums-sim run / replay / export
- Live visualisation: Mandala renderer, 3D emergence, Sonification, Dash GUI
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "GenesisAeon"
__license__ = "MIT"
__email__ = "genesis@universums-sim.dev"
__doi__ = "10.5281/zenodo.PLACEHOLDER"

from universums_sim.simulation.core import CosmicMoment, UniverseSimulator
from universums_sim.simulation.lagrangian import CollapseState, UnifiedLagrangian

__all__ = [
    "CosmicMoment",
    "UniverseSimulator",
    "UnifiedLagrangian",
    "CollapseState",
    "__version__",
]
