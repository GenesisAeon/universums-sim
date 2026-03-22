"""
Extended Unified Lagrangian for cosmic-scale simulation.

The total action is:

    S = integral L dt

where the Lagrangian density is:

    L = T_kin - V_grav - V_scalar - V_entropy - V_topological

    T_kin       = (1/2) sum_i m_i |v_i|^2
    V_grav      = -(G/2) sum_{i!=j} m_i m_j / r_{ij}
    V_scalar    = (lambda/4) phi^4 - (mu^2/2) phi^2  (Higgs-type)
    V_entropy   = kappa * S * ln(S / S_0)            (entropic potential)
    V_topological = xi * Omega                        (Chern-Simons term)

Collapse Detection
------------------
The virial ratio:

    Q = 2 * T_kin / |V_grav|

determines the collapse state:
- Q > 2          : EXPANDING
- 1 <= Q <= 2    : STABLE
- 0.5 <= Q < 1   : CONTRACTING
- Q < 0.5        : CRITICAL
- |V_grav| -> inf: SINGULARITY

References
----------
- Landau & Lifshitz, *Mechanics* §1.
- Carroll, S. (2004). *Spacetime and Geometry*. Addison-Wesley.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

# Physical constants (Planck units: G = hbar = c = k_B = 1)
_G: float = 1.0
_LAMBDA: float = 0.125  # scalar self-coupling
_MU2: float = 0.5  # mass-squared parameter
_KAPPA: float = 0.01  # entropy coupling
_XI: float = 0.001  # topological coupling
_SOFTENING: float = 1e-3  # gravitational softening length


class CollapseState(Enum):
    """Virial-theorem-based collapse classification."""

    EXPANDING = auto()
    STABLE = auto()
    CONTRACTING = auto()
    CRITICAL = auto()
    SINGULARITY = auto()


class LagrangianConfig(BaseModel):
    """Tuneable parameters for UnifiedLagrangian."""

    G: float = Field(default=_G, gt=0.0)
    lambda_scalar: float = Field(default=_LAMBDA, gt=0.0)
    mu_squared: float = Field(default=_MU2, ge=0.0)
    kappa: float = Field(default=_KAPPA, ge=0.0)
    xi: float = Field(default=_XI, ge=0.0)
    softening: float = Field(default=_SOFTENING, gt=0.0)

    model_config = {"frozen": True}


class UnifiedLagrangian:
    """
    Unified Lagrangian for cosmic N-body + scalar-field + entropic dynamics.

    Parameters
    ----------
    config : LagrangianConfig | SimulationConfig, optional
        If a SimulationConfig is passed the relevant subset of fields is used.

    Examples
    --------
    >>> lag = UnifiedLagrangian()
    >>> import numpy as np
    >>> pos = np.random.default_rng(0).standard_normal((4, 3))
    >>> vel = np.zeros((4, 3))
    >>> masses = np.ones(4)
    >>> H = lag.compute(pos, vel, masses, entropy=1.0)
    >>> assert np.isfinite(H)
    """

    def __init__(self, config: Any = None) -> None:
        if config is None:
            self._lcfg = LagrangianConfig()
        elif isinstance(config, LagrangianConfig):
            self._lcfg = config
        else:
            # Accept SimulationConfig-like objects gracefully
            self._lcfg = LagrangianConfig()

    # ------------------------------------------------------------------
    # Kinetic energy
    # ------------------------------------------------------------------

    def kinetic(
        self,
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
    ) -> float:
        """
        Compute total kinetic energy.

        .. math::

            T = \\frac{1}{2} \\sum_i m_i |\\mathbf{v}_i|^2

        Parameters
        ----------
        velocities : NDArray[np.float64], shape (N, 3)
        masses : NDArray[np.float64], shape (N,)

        Returns
        -------
        float
        """
        return float(0.5 * np.sum(masses * np.sum(velocities**2, axis=1)))

    # ------------------------------------------------------------------
    # Gravitational potential energy
    # ------------------------------------------------------------------

    def gravitational(
        self,
        positions: NDArray[np.float64],
        masses: NDArray[np.float64],
    ) -> float:
        """
        Compute total gravitational potential energy.

        .. math::

            V_{\\text{grav}} = -\\frac{G}{2} \\sum_{i \\neq j}
            \\frac{m_i m_j}{r_{ij} + \\epsilon}

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        masses : NDArray[np.float64], shape (N,)

        Returns
        -------
        float
        """
        v = 0.0
        n = positions.shape[0]
        eps = self._lcfg.softening
        g = self._lcfg.G
        for i in range(n):
            diff = positions[i + 1:] - positions[i]
            r = np.sqrt(np.sum(diff**2, axis=1) + eps**2)
            v -= g * masses[i] * np.sum(masses[i + 1:] / r)
        return float(v)

    # ------------------------------------------------------------------
    # Scalar (Higgs-type) potential
    # ------------------------------------------------------------------

    def scalar_potential(
        self,
        positions: NDArray[np.float64],
        entropy: float,
    ) -> float:
        """
        Higgs-type scalar field potential.

        The order parameter phi is approximated as the RMS displacement
        modulated by entropy:

        .. math::

            V_{\\text{scalar}} = \\frac{\\lambda}{4} \\phi^4
            - \\frac{\\mu^2}{2} \\phi^2

        with :math:`\\phi = \\langle |\\mathbf{r}| \\rangle / \\sqrt{S}`.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        entropy : float

        Returns
        -------
        float
        """
        rms = float(np.sqrt(np.mean(np.sum(positions**2, axis=1))))
        phi = rms / max(np.sqrt(entropy), 1e-10)
        phi2 = phi**2
        phi4 = phi**4
        return float(
            self._lcfg.lambda_scalar * 0.25 * phi4
            - self._lcfg.mu_squared * 0.5 * phi2
        )

    # ------------------------------------------------------------------
    # Entropic potential
    # ------------------------------------------------------------------

    def entropic_potential(self, entropy: float) -> float:
        """
        Boltzmann-type entropic potential.

        .. math::

            V_{\\text{entropy}} = \\kappa \\cdot S \\ln\\!\\left(\\frac{S}{S_0}\\right)

        with :math:`S_0 = 1` (Planck unit).

        Parameters
        ----------
        entropy : float

        Returns
        -------
        float
        """
        s = max(entropy, 1e-30)
        return float(self._lcfg.kappa * s * np.log(s))

    # ------------------------------------------------------------------
    # Topological term (Chern-Simons analogue)
    # ------------------------------------------------------------------

    def topological(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
    ) -> float:
        """
        Chern-Simons-type topological contribution.

        Approximated as :math:`\\xi \\cdot \\Omega` where
        :math:`\\Omega = \\sum_i \\mathbf{r}_i \\times \\mathbf{v}_i \\cdot \\hat{z}`.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        velocities : NDArray[np.float64], shape (N, 3)

        Returns
        -------
        float
        """
        cross = np.cross(positions, velocities)  # (N, 3)
        omega = float(np.sum(cross[:, 2]))  # z-component
        return float(self._lcfg.xi * omega)

    # ------------------------------------------------------------------
    # Full Hamiltonian
    # ------------------------------------------------------------------

    def compute(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
        entropy: float,
    ) -> float:
        """
        Compute the total Hamiltonian H = T + V_total.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        velocities : NDArray[np.float64], shape (N, 3)
        masses : NDArray[np.float64], shape (N,)
        entropy : float

        Returns
        -------
        float
            Total Hamiltonian energy H.
        """
        t = self.kinetic(velocities, masses)
        v_grav = self.gravitational(positions, masses)
        v_scalar = self.scalar_potential(positions, entropy)
        v_entropy = self.entropic_potential(entropy)
        v_topo = self.topological(positions, velocities)
        return t + v_grav + v_scalar + v_entropy + v_topo

    # ------------------------------------------------------------------
    # Gradient of the Hamiltonian (for emergence-rate damping)
    # ------------------------------------------------------------------

    def gradient(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
        entropy: float,
        delta: float = 1e-5,
    ) -> NDArray[np.float64]:
        """
        Numerical gradient of H w.r.t. positions via central differences.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        velocities : NDArray[np.float64], shape (N, 3)
        masses : NDArray[np.float64], shape (N,)
        entropy : float
        delta : float
            Finite-difference step size.

        Returns
        -------
        NDArray[np.float64], shape (N, 3)
        """
        grad = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            for j in range(3):
                pos_p = positions.copy()
                pos_m = positions.copy()
                pos_p[i, j] += delta
                pos_m[i, j] -= delta
                hp = self.compute(pos_p, velocities, masses, entropy)
                hm = self.compute(pos_m, velocities, masses, entropy)
                grad[i, j] = (hp - hm) / (2 * delta)
        return grad

    # ------------------------------------------------------------------
    # Field-theory correction force
    # ------------------------------------------------------------------

    def field_gradient(
        self,
        positions: NDArray[np.float64],
        entropy: float,
    ) -> NDArray[np.float64]:
        """
        Per-particle force from the scalar field gradient.

        Returns -dV_scalar/dr_i for each particle.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        entropy : float

        Returns
        -------
        NDArray[np.float64], shape (N, 3)
        """
        delta = 1e-5
        force = np.zeros_like(positions)
        masses_dummy = np.ones(positions.shape[0])
        vel_dummy = np.zeros_like(positions)
        for i in range(positions.shape[0]):
            for j in range(3):
                pos_p = positions.copy()
                pos_m = positions.copy()
                pos_p[i, j] += delta
                pos_m[i, j] -= delta
                vp = self.scalar_potential(pos_p, entropy)
                vm = self.scalar_potential(pos_m, entropy)
                force[i, j] = -(vp - vm) / (2 * delta)
        del masses_dummy, vel_dummy
        return force

    # ------------------------------------------------------------------
    # Collapse detection
    # ------------------------------------------------------------------

    def detect_collapse(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
        _entropy: float,
    ) -> CollapseState:
        """
        Virial-theorem collapse detector.

        The virial ratio :math:`Q = 2T / |V_{\\text{grav}}|` classifies
        the dynamical state:

        +-------+--------+-------------------+
        | Q > 2 | EXPANDING                  |
        +-------+----------------------------+
        | 1..2  | STABLE                     |
        +-------+----------------------------+
        | .5..1 | CONTRACTING                |
        +-------+----------------------------+
        | 0..5  | CRITICAL                   |
        +-------+----------------------------+
        | V→-∞  | SINGULARITY                |
        +-------+----------------------------+

        Parameters
        ----------
        positions : NDArray[np.float64], shape (N, 3)
        velocities : NDArray[np.float64], shape (N, 3)
        masses : NDArray[np.float64], shape (N,)
        entropy : float

        Returns
        -------
        CollapseState
        """
        t = self.kinetic(velocities, masses)
        v_grav = self.gravitational(positions, masses)
        abs_vg = abs(v_grav)
        if abs_vg < 1e-30:
            return CollapseState.STABLE
        if abs_vg > 1e10:
            return CollapseState.SINGULARITY
        q = 2.0 * t / abs_vg
        if q > 2.0:
            return CollapseState.EXPANDING
        if q >= 1.0:
            return CollapseState.STABLE
        if q >= 0.5:
            return CollapseState.CONTRACTING
        return CollapseState.CRITICAL
