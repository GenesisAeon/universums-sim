# API Reference

## simulation.core

### SimulationConfig

Validated Pydantic configuration for a `UniverseSimulator` run.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_particles` | `int` | 128 | Number of gravitational particles |
| `dt` | `float` | 0.01 | Integration time-step (Planck units) |
| `entropy_initial` | `float` | 1.0 | Initial entropy $S_0$ |
| `entropy_max` | `float` | 1e6 | Maximum entropy $S_{\max}$ |
| `alpha` | `float` | 0.42 | Emergence coupling constant $\alpha$ |
| `beta` | `float` | 0.1 | Gradient-suppression exponent $\beta$ |
| `collapse_threshold` | `float` | 0.95 | Virial-ratio collapse threshold |
| `seed` | `int` | 42 | Random seed |
| `observer_id` | `str` | auto | UUID4 for self-reflective observer |

---

### CosmicMoment

Immutable snapshot of a single simulation tick.

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `step` | `int` | Discrete step index $n \ge 0$ |
| `time` | `float` | Cosmic time (Planck units) |
| `entropy` | `float` | Total entropy $S(n)$ |
| `emergence_rate` | `float` | Instantaneous rate $R_e(n)$ |
| `hamiltonian` | `float` | Total Hamiltonian $H(n)$ |
| `phase` | `SimulationPhase` | Qualitative cosmic phase |
| `events` | `tuple[EmergenceEvent, ...]` | Events fired this tick |
| `collapse_state` | `CollapseState` | Virial-ratio collapse state |
| `observer_hash` | `str` | UUID of the observer instance |
| `wall_time` | `float` | UNIX timestamp |

**Methods:**

```python
moment.to_dict()  # -> dict[str, Any]  (JSON-compatible)
```

---

### UniverseSimulator

Full cosmic emergence simulator using a second-order symplectic leapfrog integrator.

```python
sim = UniverseSimulator(config)
sim.tick()                    # -> CosmicMoment
sim.run(steps=100)            # -> Generator[CosmicMoment]
sim.run_to_collapse(max_steps=100_000)  # -> Iterator[CosmicMoment]
sim.state_vector()            # -> NDArray[float64]
sim.load_state(sv)
sim.reset(new_config)
```

---

## simulation.lagrangian

### UnifiedLagrangian

Extended Lagrangian for N-body + scalar-field + entropic dynamics.

**Emergence Rate Formula:**

$$R_e(n) = \alpha \cdot S(n) \cdot \left(1 - \frac{S(n)}{S_{\max}}\right) \cdot e^{-\beta |\nabla H|}$$

**Methods:**

```python
lag.kinetic(velocities, masses)           # -> float
lag.gravitational(positions, masses)      # -> float
lag.scalar_potential(positions, entropy)  # -> float
lag.entropic_potential(entropy)           # -> float
lag.topological(positions, velocities)    # -> float
lag.compute(pos, vel, masses, entropy)    # -> float  (full H)
lag.gradient(pos, vel, masses, entropy)   # -> NDArray  (dH/dr)
lag.field_gradient(positions, entropy)    # -> NDArray  (field force)
lag.detect_collapse(pos, vel, masses, entropy)  # -> CollapseState
```

### CollapseState

```python
class CollapseState(Enum):
    EXPANDING   # Q > 2
    STABLE      # 1 <= Q <= 2
    CONTRACTING # 0.5 <= Q < 1
    CRITICAL    # Q < 0.5
    SINGULARITY # |V_grav| -> inf
```

---

## simulation.emergence

### EmergenceEngine

Computes emergence rates and fires `EmergenceEvent` instances.

$$R_e = \alpha S \left(1 - \frac{S}{S_{\max}}\right) e^{-\beta|\nabla H|}$$

```python
engine.compute_rate(entropy, grad_h)  # -> float
engine.fire_events(step, time, entropy, rate)  # -> list[EmergenceEvent]
engine.event_count  # -> int
```

### EmergenceEvent

Frozen dataclass snapshot of one emergence event.

```python
event.to_dict()  # -> dict[str, Any]
```

### EmergenceType

```python
class EmergenceType(Enum):
    MICRO       # < 0.001
    MESO        # 0.001 .. 0.01
    MACRO       # 0.01 .. 0.1
    COSMIC      # 0.1 .. 1.0
    TRANSCENDENT  # >= 1.0
```

---

## governance.entropy

### GovernancePolicy

UTAC-compatible entropy governance policy (Pydantic model).

```python
GovernancePolicy(
    entropy_warn=1_000.0,
    entropy_halt=10_000.0,
    entropy_reset=100_000.0,
    weight_ethical=1.0,
)
```

### EntropyGovernor

Evaluates entropy against `GovernancePolicy` and returns `GovernanceDecision`.

```python
gov = EntropyGovernor(policy)
decision = gov.evaluate(entropy)       # -> GovernanceDecision
rate     = gov.apply_weight(rate, entropy)  # -> float
```

### PolicyAction

```python
class PolicyAction(Enum):
    ALLOW    # entropy within safe bounds
    THROTTLE # entropy above warn threshold
    HALT     # entropy above halt threshold
    RESET    # entropy above reset threshold
```

---

## visualization.live

### LiveVisualizer

Facade orchestrating `MandalaRenderer`, `Emergence3D`, and `DashDashboard`.

```python
viz = LiveVisualizer(simulator, gui_port=8050)
viz.launch_gui()
viz.update(moment)
viz.svg_frames           # list[str]
viz.save_svg_animation("out.svg")
```

### MandalaRenderer

```python
renderer = MandalaRenderer(n_petals=12, size=512)
renderer.render(moment)  # -> dict
renderer.to_svg(moment)  # -> str
```

### SonificationEngine

```python
sonifier = SonificationEngine(sample_rate=44100, duration=0.1)
sonifier.process(moment)  # plays audio for emergence events
```

---

## integrations.registry

### IntegrationRegistry

Lazy registry of optional GenesisAeon package availability.

```python
reg = IntegrationRegistry()
reg.is_available("genesis_os")    # -> bool
reg.available_packages()          # -> list[str]
reg.unavailable_packages()        # -> list[str]
reg.status_dict()                 # -> dict[str, bool]
reg.get_module("genesis_os")      # -> module | None
```

---

## CLI Reference

```
universums-sim run [OPTIONS]
  --steps     INTEGER   Number of simulation steps [default: 100]
  --entropy   FLOAT     Initial entropy S_0           [default: 1.0]
  --particles INTEGER   Number of particles            [default: 64]
  --dt        FLOAT     Integration time-step          [default: 0.01]
  --alpha     FLOAT     Emergence coupling             [default: 0.42]
  --seed      INTEGER   Random seed                    [default: 42]
  --visualize           Enable live mandala rendering
  --sonify              Enable emergence sonification
  --gui                 Launch Dash GUI dashboard
  --output    PATH      Save moments to JSON
  --verbose             Verbose structured logging

universums-sim replay INPUT_FILE [OPTIONS]
  --speed     FLOAT     Replay speed multiplier        [default: 1.0]
  --visualize           Enable live rendering

universums-sim export INPUT_FILE [OPTIONS]
  --output    PATH      Output file path
  --format    TEXT      json|csv|hdf5                  [default: json]

universums-sim info
```
