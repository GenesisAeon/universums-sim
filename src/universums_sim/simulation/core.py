"""
Core simulation engine for universums-sim.

Implements the UniverseSimulator with CosmicMoment event tracking and
the self-reflective emergence engine driving cosmic-scale dynamics.

Mathematical foundation
-----------------------
The discrete time-evolution follows a symplectic leapfrog integrator
applied to the Hamiltonian derived from the UnifiedLagrangian:

    H = T + V_grav + V_field + V_entropy

where the emergence rate at step n is:

    R_e(n) = alpha * S(n) * (1 - S(n)/S_max) * exp(-beta * |nabla H|)

References
----------
- Penrose, R. (2004). *The Road to Reality*. Jonathan Cape.
- Smolin, L. (2013). *Time Reborn*. Houghton Mifflin Harcourt.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from universums_sim.simulation.emergence import EmergenceEngine, EmergenceEvent
from universums_sim.simulation.lagrangian import CollapseState, UnifiedLagrangian

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SimulationPhase(Enum):
    """Qualitative phase of the cosmic simulation."""

    GENESIS = auto()
    INFLATION = auto()
    RADIATION_DOMINATED = auto()
    MATTER_DOMINATED = auto()
    DARK_ENERGY_DOMINATED = auto()
    COLLAPSE = auto()
    TRANSCENDENCE = auto()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CosmicMoment:
    """
    Immutable snapshot of a single simulation tick.

    Attributes
    ----------
    step : int
        Discrete simulation step index (n >= 0).
    time : float
        Cosmic time in Planck units.
    entropy : float
        Total system entropy S(n) in nats.
    emergence_rate : float
        Instantaneous emergence rate R_e(n).
    hamiltonian : float
        Total Hamiltonian energy H(n).
    phase : SimulationPhase
        Qualitative cosmic phase label.
    events : tuple[EmergenceEvent, ...]
        All emergence events fired this tick.
    collapse_state : CollapseState
        Lagrangian collapse-detection output.
    observer_hash : str
        UUID4 identifying the self-reflective observer instance.
    wall_time : float
        Wall-clock UNIX timestamp of this record.
    metadata : dict[str, Any]
        Arbitrary extensible metadata.
    """

    step: int
    time: float
    entropy: float
    emergence_rate: float
    hamiltonian: float
    phase: SimulationPhase
    events: tuple[EmergenceEvent, ...]
    collapse_state: CollapseState
    observer_hash: str
    wall_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"CosmicMoment(step={self.step}, t={self.time:.4f}, "
            f"S={self.entropy:.4f}, R_e={self.emergence_rate:.6f}, "
            f"phase={self.phase.name})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "step": self.step,
            "time": self.time,
            "entropy": self.entropy,
            "emergence_rate": self.emergence_rate,
            "hamiltonian": self.hamiltonian,
            "phase": self.phase.name,
            "events": [e.to_dict() for e in self.events],
            "collapse_state": self.collapse_state.name,
            "observer_hash": self.observer_hash,
            "wall_time": self.wall_time,
            "metadata": self.metadata,
        }


class SimulationConfig(BaseModel):
    """
    Validated configuration for a UniverseSimulator run.

    Parameters
    ----------
    n_particles : int
        Number of gravitational particles (default 128).
    dt : float
        Integration time-step in Planck units (default 0.01).
    entropy_initial : float
        Initial entropy S_0 (default 1.0).
    entropy_max : float
        Maximum entropy S_max (default 1e6).
    alpha : float
        Emergence coupling constant alpha (default 0.42).
    beta : float
        Gradient-suppression constant beta (default 0.1).
    collapse_threshold : float
        Normalised energy ratio triggering collapse detection (default 0.95).
    seed : int
        Random seed for reproducibility (default 42).
    observer_id : str
        UUID for the self-reflective observer (auto-generated if empty).
    """

    n_particles: int = Field(default=128, ge=2, le=65536)
    dt: float = Field(default=0.01, gt=0.0, lt=10.0)
    entropy_initial: float = Field(default=1.0, gt=0.0)
    entropy_max: float = Field(default=1_000_000.0, gt=1.0)
    alpha: float = Field(default=0.42, gt=0.0)
    beta: float = Field(default=0.1, ge=0.0)
    collapse_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)
    observer_id: str = Field(default="")

    @field_validator("entropy_max")
    @classmethod
    def _entropy_max_gt_initial(cls, v: float, info: Any) -> float:
        if "entropy_initial" in (info.data or {}) and v <= info.data["entropy_initial"]:
            msg = "entropy_max must be greater than entropy_initial"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _ensure_observer_id(self) -> "SimulationConfig":
        if not self.observer_id:
            object.__setattr__(self, "observer_id", str(uuid.uuid4()))
        return self

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Universe Simulator
# ---------------------------------------------------------------------------


class UniverseSimulator:
    """
    Full cosmic emergence simulator.

    Integrates the equations of motion derived from the UnifiedLagrangian
    using a second-order symplectic (leapfrog) scheme and fires CosmicMoment
    events at every tick.

    Parameters
    ----------
    config : SimulationConfig
        Validated configuration object.

    Examples
    --------
    >>> cfg = SimulationConfig(n_particles=32, seed=7)
    >>> sim = UniverseSimulator(cfg)
    >>> moment = next(sim.run(steps=1))
    >>> assert moment.step == 0
    """

    _PHASE_THRESHOLDS: tuple[tuple[float, SimulationPhase], ...] = (
        (0.001, SimulationPhase.GENESIS),
        (0.01, SimulationPhase.INFLATION),
        (0.1, SimulationPhase.RADIATION_DOMINATED),
        (1.0, SimulationPhase.MATTER_DOMINATED),
        (10.0, SimulationPhase.DARK_ENERGY_DOMINATED),
        (float("inf"), SimulationPhase.COLLAPSE),
    )

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._cfg = config or SimulationConfig()
        self._rng = np.random.default_rng(self._cfg.seed)
        self._lagrangian = UnifiedLagrangian(config=self._cfg)
        self._emergence = EmergenceEngine(
            alpha=self._cfg.alpha,
            beta=self._cfg.beta,
            entropy_max=self._cfg.entropy_max,
            rng=self._rng,
        )
        # State arrays — shape (N, 3) in reduced Planck units
        self._positions: NDArray[np.float64] = self._rng.standard_normal(
            (self._cfg.n_particles, 3)
        )
        self._velocities: NDArray[np.float64] = self._rng.standard_normal(
            (self._cfg.n_particles, 3)
        ) * 0.1
        self._masses: NDArray[np.float64] = (
            np.abs(self._rng.standard_normal(self._cfg.n_particles)) + 0.5
        )
        self._entropy: float = self._cfg.entropy_initial
        self._step: int = 0
        self._cosmic_time: float = 0.0
        log.info(
            "UniverseSimulator initialised",
            n_particles=self._cfg.n_particles,
            observer=self._cfg.observer_id[:8],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SimulationConfig:
        """Current simulation configuration (immutable)."""
        return self._cfg

    @property
    def step(self) -> int:
        """Current simulation step index."""
        return self._step

    @property
    def entropy(self) -> float:
        """Current total entropy S(n)."""
        return self._entropy

    @property
    def positions(self) -> NDArray[np.float64]:
        """Particle positions array of shape (N, 3)."""
        return self._positions.copy()

    @property
    def velocities(self) -> NDArray[np.float64]:
        """Particle velocities array of shape (N, 3)."""
        return self._velocities.copy()

    @property
    def masses(self) -> NDArray[np.float64]:
        """Particle masses array of shape (N,)."""
        return self._masses.copy()

    # ------------------------------------------------------------------
    # Core integration
    # ------------------------------------------------------------------

    def _compute_accelerations(self) -> NDArray[np.float64]:
        """Compute pairwise gravitational + field accelerations (O(N^2))."""
        acc = np.zeros_like(self._positions)
        n = self._cfg.n_particles
        softening = 0.01  # Planck lengths
        for i in range(n):
            diff = self._positions - self._positions[i]  # (N,3)
            r2 = np.sum(diff**2, axis=1) + softening**2
            r2[i] = 1.0  # avoid self-force (overwritten by zero mass trick)
            inv_r3 = r2 ** (-1.5)
            inv_r3[i] = 0.0
            acc[i] = np.sum(
                (self._masses[:, None] * inv_r3[:, None]) * diff, axis=0
            )
        # Add field-theory correction from Lagrangian
        field_force = self._lagrangian.field_gradient(self._positions, self._entropy)
        acc += field_force
        return acc

    def _leapfrog_step(self) -> None:
        """Advance positions and velocities by one leapfrog step."""
        dt = self._cfg.dt
        acc = self._compute_accelerations()
        # half-kick
        self._velocities += 0.5 * dt * acc
        # drift
        self._positions += dt * self._velocities
        # refresh accelerations
        acc = self._compute_accelerations()
        # half-kick
        self._velocities += 0.5 * dt * acc
        self._cosmic_time += dt

    def _update_entropy(self, emergence_rate: float) -> None:
        """Evolve entropy according to dS/dt = R_e * (1 - S/S_max)."""
        ds = (
            emergence_rate
            * self._entropy
            * (1.0 - self._entropy / self._cfg.entropy_max)
            * self._cfg.dt
        )
        self._entropy = max(0.0, self._entropy + ds)

    def _determine_phase(self) -> SimulationPhase:
        """Map current entropy to a SimulationPhase label."""
        ratio = self._entropy / self._cfg.entropy_max
        if ratio > self._cfg.collapse_threshold:
            return SimulationPhase.COLLAPSE
        for threshold, phase in self._PHASE_THRESHOLDS:
            if self._cosmic_time < threshold:
                return phase
        return SimulationPhase.TRANSCENDENCE  # pragma: no cover

    def tick(self) -> CosmicMoment:
        """
        Execute a single simulation tick and return the resulting CosmicMoment.

        Returns
        -------
        CosmicMoment
            Fully populated snapshot of the simulation at step n.
        """
        self._leapfrog_step()
        hamiltonian = self._lagrangian.compute(
            self._positions, self._velocities, self._masses, self._entropy
        )
        grad_h = float(np.linalg.norm(self._lagrangian.gradient(
            self._positions, self._velocities, self._masses, self._entropy
        )))
        emergence_rate = self._emergence.compute_rate(self._entropy, grad_h)
        events = self._emergence.fire_events(
            step=self._step,
            time=self._cosmic_time,
            entropy=self._entropy,
            emergence_rate=emergence_rate,
        )
        collapse_state = self._lagrangian.detect_collapse(
            self._positions, self._velocities, self._masses, self._entropy
        )
        self._update_entropy(emergence_rate)
        phase = self._determine_phase()
        moment = CosmicMoment(
            step=self._step,
            time=self._cosmic_time,
            entropy=self._entropy,
            emergence_rate=emergence_rate,
            hamiltonian=float(hamiltonian),
            phase=phase,
            events=tuple(events),
            collapse_state=collapse_state,
            observer_hash=self._cfg.observer_id,
            wall_time=time.time(),
        )
        log.debug("tick", step=self._step, phase=phase.name, entropy=self._entropy)
        self._step += 1
        return moment

    def run(self, steps: int = 100) -> Generator[CosmicMoment, None, None]:
        """
        Yield CosmicMoments for *steps* consecutive ticks.

        Parameters
        ----------
        steps : int
            Number of ticks to execute (must be >= 1).

        Yields
        ------
        CosmicMoment
            One moment per tick.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        for _ in range(steps):
            yield self.tick()

    def run_to_collapse(self, max_steps: int = 100_000) -> Iterator[CosmicMoment]:
        """
        Run until collapse is detected or max_steps is reached.

        Parameters
        ----------
        max_steps : int
            Hard upper bound on simulation steps.

        Yields
        ------
        CosmicMoment
        """
        for moment in self.run(max_steps):
            yield moment
            if moment.collapse_state in (
                CollapseState.CRITICAL,
                CollapseState.SINGULARITY,
            ):
                log.info("Collapse detected", step=moment.step, state=moment.collapse_state.name)
                break

    def reset(self, config: SimulationConfig | None = None) -> None:
        """
        Reset the simulator to initial conditions.

        Parameters
        ----------
        config : SimulationConfig, optional
            If provided, replaces the current configuration.
        """
        if config is not None:
            self._cfg = config
        self.__init__(self._cfg)  # type: ignore[misc]

    def state_vector(self) -> NDArray[np.float64]:
        """
        Flatten current state to a 1-D numpy array.

        Returns
        -------
        NDArray[np.float64]
            Concatenation of [positions, velocities, masses, [entropy]].
        """
        return np.concatenate([
            self._positions.ravel(),
            self._velocities.ravel(),
            self._masses,
            [self._entropy],
        ])

    def load_state(self, state: NDArray[np.float64]) -> None:
        """
        Restore simulator from a flattened state vector.

        Parameters
        ----------
        state : NDArray[np.float64]
            Vector produced by :meth:`state_vector`.
        """
        n = self._cfg.n_particles
        expected = n * 3 + n * 3 + n + 1
        if state.shape != (expected,):
            msg = f"Expected state shape ({expected},), got {state.shape}"
            raise ValueError(msg)
        self._positions = state[: n * 3].reshape(n, 3)
        self._velocities = state[n * 3 : n * 6].reshape(n, 3)
        self._masses = state[n * 6 : n * 7]
        self._entropy = float(state[-1])
