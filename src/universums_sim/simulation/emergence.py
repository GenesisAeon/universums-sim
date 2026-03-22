"""
Emergence engine: event detection and rate computation.

Emergence rate formula:

    R_e(n) = alpha * S(n) * (1 - S(n)/S_max) * exp(-beta * |nablaH|)

An EmergenceEvent is fired when R_e crosses a threshold drawn from an
exponential distribution — analogous to a Poisson process in continuous time.

References
----------
- Kauffman, S. A. (1993). *The Origins of Order*. Oxford University Press.
- Walker, S. I. & Davies, P. C. W. (2013). "The algorithmic origins of life."
  *J. R. Soc. Interface*, 10(79), 20120869.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.random import Generator


class EmergenceType(Enum):
    """Qualitative classification of an emergence event."""

    MICRO = auto()       # sub-Planck fluctuation
    MESO = auto()        # molecular / cellular scale
    MACRO = auto()       # planetary / stellar scale
    COSMIC = auto()      # galactic / universal scale
    TRANSCENDENT = auto()  # beyond known physics


@dataclass(frozen=True, slots=True)
class EmergenceEvent:
    """
    A single emergence event fired by the EmergenceEngine.

    Attributes
    ----------
    event_id : str
        Unique UUID4 for this event.
    step : int
        Simulation step at which the event fired.
    time : float
        Cosmic time at firing.
    entropy : float
        Entropy level S(n) at firing.
    rate : float
        Instantaneous emergence rate R_e at firing.
    kind : EmergenceType
        Qualitative scale of the emergence.
    description : str
        Human-readable description.
    metadata : dict[str, Any]
        Optional extensible metadata.
    """

    event_id: str
    step: int
    time: float
    entropy: float
    rate: float
    kind: EmergenceType
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "step": self.step,
            "time": self.time,
            "entropy": self.entropy,
            "rate": self.rate,
            "kind": self.kind.name,
            "description": self.description,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Emergence Engine
# ---------------------------------------------------------------------------

_KIND_THRESHOLDS: tuple[tuple[float, EmergenceType], ...] = (
    (0.001, EmergenceType.MICRO),
    (0.01, EmergenceType.MESO),
    (0.1, EmergenceType.MACRO),
    (1.0, EmergenceType.COSMIC),
    (float("inf"), EmergenceType.TRANSCENDENT),
)

_KIND_DESCRIPTIONS: dict[EmergenceType, str] = {
    EmergenceType.MICRO: "Quantum fluctuation emergence below Planck scale",
    EmergenceType.MESO: "Meso-scale pattern crystallisation",
    EmergenceType.MACRO: "Macro-structure formation (stellar/planetary)",
    EmergenceType.COSMIC: "Galactic-web emergence event",
    EmergenceType.TRANSCENDENT: "Transcendent emergence beyond standard physics",
}


class EmergenceEngine:
    """
    Computes emergence rates and fires discrete EmergenceEvents.

    Parameters
    ----------
    alpha : float
        Emergence coupling constant.
    beta : float
        Gradient-suppression exponent.
    entropy_max : float
        Maximum system entropy S_max.
    rng : numpy.random.Generator
        Shared random number generator.
    base_threshold : float
        Base Poisson threshold for event firing (default 0.5).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> engine = EmergenceEngine(alpha=0.42, beta=0.1, entropy_max=1e6, rng=rng)
    >>> rate = engine.compute_rate(entropy=100.0, grad_h=0.5)
    >>> assert rate >= 0.0
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        entropy_max: float,
        rng: Generator,
        base_threshold: float = 0.5,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._entropy_max = entropy_max
        self._rng = rng
        self._base_threshold = base_threshold
        self._event_counter: int = 0

    # ------------------------------------------------------------------
    # Rate computation
    # ------------------------------------------------------------------

    def compute_rate(self, entropy: float, grad_h: float) -> float:
        """
        Compute the instantaneous emergence rate.

        .. math::

            R_e = \\alpha \\cdot S \\cdot
            \\left(1 - \\frac{S}{S_{\\max}}\\right) \\cdot
            e^{-\\beta |\\nabla H|}

        Parameters
        ----------
        entropy : float
            Current entropy S(n).
        grad_h : float
            Euclidean norm of the Hamiltonian gradient |∇H|.

        Returns
        -------
        float
            Non-negative emergence rate R_e(n).
        """
        if entropy <= 0:
            return 0.0
        saturation = max(0.0, 1.0 - entropy / self._entropy_max)
        damping = float(np.exp(-self._beta * abs(grad_h)))
        return float(self._alpha * entropy * saturation * damping)

    # ------------------------------------------------------------------
    # Event firing
    # ------------------------------------------------------------------

    def _classify(self, rate: float) -> EmergenceType:
        """Map a rate value to a qualitative EmergenceType."""
        for threshold, kind in _KIND_THRESHOLDS:
            if rate < threshold:
                return kind
        return EmergenceType.TRANSCENDENT  # pragma: no cover

    def fire_events(
        self,
        step: int,
        time: float,
        entropy: float,
        emergence_rate: float,
    ) -> list[EmergenceEvent]:
        """
        Stochastically fire emergence events for this tick.

        The probability of at least one event per tick is modelled as a
        Poisson process with mean ``emergence_rate * dt`` approximated
        by the threshold test:

            fire if U[0,1) < min(emergence_rate / base_threshold, 1)

        Parameters
        ----------
        step : int
        time : float
        entropy : float
        emergence_rate : float

        Returns
        -------
        list[EmergenceEvent]
            Zero or more events for this tick.
        """
        events: list[EmergenceEvent] = []
        prob = min(emergence_rate / max(self._base_threshold, 1e-30), 1.0)
        if self._rng.random() < prob:
            kind = self._classify(emergence_rate)
            event = EmergenceEvent(
                event_id=f"evt-{self._event_counter:08d}",
                step=step,
                time=time,
                entropy=entropy,
                rate=emergence_rate,
                kind=kind,
                description=_KIND_DESCRIPTIONS[kind],
                metadata={"wall_time": time_module_time()},
            )
            events.append(event)
            self._event_counter += 1
        return events

    @property
    def event_count(self) -> int:
        """Total number of emergence events fired so far."""
        return self._event_counter


def time_module_time() -> float:
    """Return current wall-clock time (thin wrapper for testability)."""
    return time.time()
