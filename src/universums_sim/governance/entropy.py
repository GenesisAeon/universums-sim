"""
Entropy governance: policy-based entropy bounds and ethical constraints.

Implements UTAC (Universal Thermodynamic Autonomy Constraints) compatible
governance on top of the simulation entropy budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, Field


class PolicyAction(Enum):
    """Action taken by the governor for a given entropy level."""

    ALLOW = auto()
    THROTTLE = auto()
    HALT = auto()
    RESET = auto()


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """
    Decision returned by EntropyGovernor.evaluate().

    Attributes
    ----------
    action : PolicyAction
    entropy : float
        Entropy at decision time.
    reason : str
        Human-readable rationale.
    metadata : dict[str, Any]
    """

    action: PolicyAction
    entropy: float
    reason: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "action": self.action.name,
            "entropy": self.entropy,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class GovernancePolicy(BaseModel):
    """
    UTAC-compatible governance policy for entropy control.

    Parameters
    ----------
    entropy_warn : float
        Entropy level triggering a THROTTLE action.
    entropy_halt : float
        Entropy level triggering a HALT action.
    entropy_reset : float
        Entropy level triggering a RESET action (beyond halt).
    weight_ethical : float
        Ethical weight factor applied to emergence probability.
    """

    entropy_warn: float = Field(default=1_000.0, gt=0.0)
    entropy_halt: float = Field(default=10_000.0, gt=0.0)
    entropy_reset: float = Field(default=100_000.0, gt=0.0)
    weight_ethical: float = Field(default=1.0, gt=0.0, le=10.0)

    model_config = {"frozen": True}


class EntropyGovernor:
    """
    Evaluates entropy against a GovernancePolicy and returns decisions.

    Parameters
    ----------
    policy : GovernancePolicy
        Active governance policy.

    Examples
    --------
    >>> policy = GovernancePolicy(entropy_warn=500.0, entropy_halt=5000.0)
    >>> gov = EntropyGovernor(policy)
    >>> decision = gov.evaluate(entropy=100.0)
    >>> assert decision.action == PolicyAction.ALLOW
    """

    def __init__(self, policy: GovernancePolicy | None = None) -> None:
        self._policy = policy or GovernancePolicy()

    @property
    def policy(self) -> GovernancePolicy:
        """Active policy (immutable)."""
        return self._policy

    def evaluate(self, entropy: float) -> GovernanceDecision:
        """
        Evaluate the current entropy against the active policy.

        Parameters
        ----------
        entropy : float
            Current system entropy.

        Returns
        -------
        GovernanceDecision
        """
        p = self._policy
        if entropy >= p.entropy_reset:
            return GovernanceDecision(
                action=PolicyAction.RESET,
                entropy=entropy,
                reason=f"Entropy {entropy:.2f} exceeds reset threshold {p.entropy_reset}",
                metadata={},
            )
        if entropy >= p.entropy_halt:
            return GovernanceDecision(
                action=PolicyAction.HALT,
                entropy=entropy,
                reason=f"Entropy {entropy:.2f} exceeds halt threshold {p.entropy_halt}",
                metadata={},
            )
        if entropy >= p.entropy_warn:
            return GovernanceDecision(
                action=PolicyAction.THROTTLE,
                entropy=entropy,
                reason=f"Entropy {entropy:.2f} exceeds warning threshold {p.entropy_warn}",
                metadata={"throttle_factor": entropy / p.entropy_halt},
            )
        return GovernanceDecision(
            action=PolicyAction.ALLOW,
            entropy=entropy,
            reason="Entropy within safe bounds",
            metadata={"headroom": p.entropy_warn - entropy},
        )

    def apply_weight(self, emergence_rate: float, entropy: float) -> float:
        """
        Apply ethical weighting to an emergence rate.

        Parameters
        ----------
        emergence_rate : float
        entropy : float

        Returns
        -------
        float
            Weighted emergence rate.
        """
        decision = self.evaluate(entropy)
        if decision.action == PolicyAction.HALT:
            return 0.0
        if decision.action == PolicyAction.RESET:
            return 0.0
        if decision.action == PolicyAction.THROTTLE:
            factor = decision.metadata.get("throttle_factor", 1.0)
            return emergence_rate * max(0.0, 1.0 - float(factor))
        return emergence_rate * self._policy.weight_ethical
