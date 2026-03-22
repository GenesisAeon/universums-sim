"""
Integration registry: tracks which optional packages are available at runtime.
"""

from __future__ import annotations

import importlib
from typing import Any

_OPTIONAL_PACKAGES: tuple[str, ...] = (
    "genesis_os",
    "aeon_ai",
    "cosmic_web",
    "fieldtheory",
    "mirror_machine",
    "advanced_weighting_systems",
    "sigillin",
    "entropy_governance",
    "utac_core",
    "mandala_visualizer",
    "sonification",
    "climate_dashboard",
    "implosive_genesis",
    "entropy_table",
)


class IntegrationRegistry:
    """
    Lazy registry of optional package availability.

    Examples
    --------
    >>> reg = IntegrationRegistry()
    >>> available = reg.available_packages()
    >>> isinstance(available, list)
    True
    """

    def __init__(self) -> None:
        self._cache: dict[str, bool] = {}

    def is_available(self, package: str) -> bool:
        """
        Check whether *package* can be imported.

        Parameters
        ----------
        package : str
            Python import name (e.g. 'genesis_os').

        Returns
        -------
        bool
        """
        if package not in self._cache:
            try:
                importlib.import_module(package)
                self._cache[package] = True
            except ImportError:
                self._cache[package] = False
        return self._cache[package]

    def available_packages(self) -> list[str]:
        """Return a list of all optional packages that are importable."""
        return [p for p in _OPTIONAL_PACKAGES if self.is_available(p)]

    def unavailable_packages(self) -> list[str]:
        """Return a list of all optional packages that are NOT importable."""
        return [p for p in _OPTIONAL_PACKAGES if not self.is_available(p)]

    def get_module(self, package: str) -> Any | None:
        """
        Return the imported module for *package*, or None if unavailable.

        Parameters
        ----------
        package : str

        Returns
        -------
        module or None
        """
        if not self.is_available(package):
            return None
        return importlib.import_module(package)

    def status_dict(self) -> dict[str, bool]:
        """Return a dict mapping package name -> availability bool."""
        return {p: self.is_available(p) for p in _OPTIONAL_PACKAGES}
