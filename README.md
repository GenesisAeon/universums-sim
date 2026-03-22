# universums-sim

<p align="center">
  <img src="docs/assets/unified-mandala.svg" alt="Unified Mandala Logo" width="200"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/universums-sim/"><img src="https://img.shields.io/pypi/v/universums-sim.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/universums-sim/"><img src="https://img.shields.io/pypi/pyversions/universums-sim.svg" alt="Python"></a>
  <a href="https://doi.org/10.5281/zenodo.PLACEHOLDER"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/GenesisAeon/universums-sim/actions"><img src="https://github.com/GenesisAeon/universums-sim/workflows/CI/badge.svg" alt="CI"></a>
</p>

> *"A system that listens — a pattern that lives."*

**universums-sim** is a complete cosmic emergence simulation package for the GenesisAeon framework. It provides a self-reflective N-body + field-theory simulation engine driven by an extended Unified Lagrangian, live mandala visualisation, sonification of emergence events, and a full Dash GUI — all behind a clean Typer CLI.

---

## Mathematical Foundation

### Unified Lagrangian

The total action of the system is:

$$S = \int \mathcal{L} \, dt$$

where the Lagrangian density combines kinetic, gravitational, scalar, entropic, and topological terms:

$$\mathcal{L} = T_{\text{kin}} - V_{\text{grav}} - V_{\text{scalar}} - V_{\text{entropy}} - V_{\text{topo}}$$

**Kinetic energy:**

$$T_{\text{kin}} = \frac{1}{2} \sum_{i=1}^{N} m_i \left|\mathbf{v}_i\right|^2$$

**Gravitational potential (softened):**

$$V_{\text{grav}} = -\frac{G}{2} \sum_{i \neq j} \frac{m_i m_j}{\left|\mathbf{r}_{ij}\right| + \varepsilon}$$

**Higgs-type scalar potential:**

$$V_{\text{scalar}} = \frac{\lambda}{4}\phi^4 - \frac{\mu^2}{2}\phi^2, \quad \phi = \frac{\langle|\mathbf{r}|\rangle}{\sqrt{S}}$$

**Entropic potential:**

$$V_{\text{entropy}} = \kappa \cdot S \ln\!\left(\frac{S}{S_0}\right)$$

**Chern-Simons topological term:**

$$V_{\text{topo}} = \xi \cdot \Omega, \quad \Omega = \sum_i \left(\mathbf{r}_i \times \mathbf{v}_i\right)_z$$

### Emergence Rate

The instantaneous emergence rate at step $n$ is:

$$R_e(n) = \alpha \cdot S(n) \cdot \left(1 - \frac{S(n)}{S_{\max}}\right) \cdot e^{-\beta\left|\nabla H\right|}$$

### Collapse Detection (Virial Ratio)

$$Q = \frac{2 T_{\text{kin}}}{|V_{\text{grav}}|}$$

| $Q$               | State           |
|-------------------|-----------------|
| $Q > 2$           | `EXPANDING`     |
| $1 \le Q \le 2$   | `STABLE`        |
| $0.5 \le Q < 1$   | `CONTRACTING`   |
| $Q < 0.5$         | `CRITICAL`      |
| $V \to -\infty$   | `SINGULARITY`   |

### CosmicMoment

Each simulation tick produces a `CosmicMoment` snapshot:

$$\mathcal{M}_n = \left\{ n,\; t_n,\; S(n),\; R_e(n),\; H(n),\; \Phi(n),\; \{E_k\},\; Q_n \right\}$$

---

## Installation

```bash
# Minimal (simulation + CLI only)
pip install universums-sim

# Full stack (all GenesisAeon packages)
pip install 'universums-sim[full-stack]'

# GUI dashboard
pip install 'universums-sim[gui]'

# Sonification
pip install 'universums-sim[sonify]'

# Everything
pip install 'universums-sim[full-stack,gui,sonify]'
```

---

## Quick Start

```python
from universums_sim import UniverseSimulator
from universums_sim.simulation.core import SimulationConfig

cfg = SimulationConfig(n_particles=64, seed=42)
sim = UniverseSimulator(cfg)

for moment in sim.run(steps=100):
    print(moment)
```

---

## CLI

```bash
# Basic run
universums-sim run --steps 500 --entropy 1.0

# With live visualization
universums-sim run --steps 500 --visualize

# With sonification
universums-sim run --steps 500 --sonify

# Full GUI dashboard (opens http://localhost:8050)
universums-sim run --steps 1000 --gui

# Save to JSON
universums-sim run --steps 500 --output sim.json

# Export to CSV
universums-sim export sim.json --format csv

# Show version and citation
universums-sim info
```

---

## Package Structure

```
src/universums_sim/
├── simulation/
│   ├── core.py          # UniverseSimulator, CosmicMoment, SimulationConfig
│   ├── lagrangian.py    # UnifiedLagrangian, CollapseState
│   └── emergence.py     # EmergenceEngine, EmergenceEvent, EmergenceType
├── cli/
│   └── main.py          # Typer CLI (run / replay / export / info)
├── visualization/
│   └── live.py          # MandalaRenderer, Emergence3D, SonificationEngine,
│                        #   DashDashboard, LiveVisualizer
├── governance/
│   └── entropy.py       # EntropyGovernor, GovernancePolicy (UTAC)
└── integrations/
    └── registry.py      # IntegrationRegistry (full-stack detection)
```

---

## Optional Dependencies ([full-stack] extra)

| Package | Version | Role |
|---------|---------|------|
| `genesis-os` | >= 0.2.0 | Base OS layer for GenesisAeon |
| `aeon-ai` | >= 0.2.0 | AI inference integration |
| `cosmic-web` | >= 0.1.0 | Cosmic web connectivity |
| `fieldtheory` | >= 0.1.0 | Field-theory primitives |
| `mirror-machine` | >= 0.1.0 | Self-reflective loop engine |
| `advanced-weighting-systems` | >= 0.1.0 | Weighted entropy dynamics |
| `sigillin` | >= 0.1.0 | Symbolic sigil control |
| `entropy-governance` | >= 0.1.0 | Entropy policy enforcement |
| `utac-core` | >= 0.1.0 | Universal Thermodynamic Autonomy |
| `mandala-visualizer` | >= 0.1.0 | Sacred-geometry rendering |
| `sonification` | >= 0.1.0 | Emergence-to-audio mapping |
| `climate-dashboard` | >= 0.1.0 | Climate entropy dashboard |
| `implosive-genesis` | >= 0.1.0 | Implosion dynamics |
| `entropy-table` | >= 0.1.0 | Tabular entropy tracking |

---

## Development

```bash
git clone https://github.com/GenesisAeon/universums-sim
cd universums-sim
pip install -e '.[dev]'

# Tests (>600, >99% coverage)
pytest

# Linting
ruff check src tests

# Type checking
mypy src

# Docs
mkdocs serve
```

---

## Scientific Citation

```bibtex
@software{genesisaeon_universumssim_2024,
  author       = {GenesisAeon},
  title        = {universums-sim: Complete Cosmic Emergence Simulation},
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.PLACEHOLDER},
  url          = {https://doi.org/10.5281/zenodo.PLACEHOLDER}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
