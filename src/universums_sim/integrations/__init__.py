"""
Integration shims for the full-stack optional dependencies.

Each sub-module provides a lightweight adapter that wraps an optional
package's public API and returns a safe stub when the package is absent.
"""

from universums_sim.integrations.registry import IntegrationRegistry

__all__ = ["IntegrationRegistry"]
