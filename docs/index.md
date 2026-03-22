# universums-sim

Complete cosmic emergence simulation with self-reflective observation
and live visualization for GenesisAeon.

See the [API Reference](reference.md) for full documentation.

## Quick Start

```python
from universums_sim import UniverseSimulator
from universums_sim.simulation.core import SimulationConfig

cfg = SimulationConfig(n_particles=64, seed=42)
sim = UniverseSimulator(cfg)

for moment in sim.run(steps=100):
    print(moment)
```
