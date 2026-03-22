"""
Tests for universums_sim.integrations.registry.

Covers:
- IntegrationRegistry availability checking
- status_dict()
- available_packages() / unavailable_packages()
- get_module()
- Contract stubs (without actual packages installed)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from universums_sim.integrations.registry import IntegrationRegistry, _OPTIONAL_PACKAGES


# ---------------------------------------------------------------------------
# Registry basic tests (30 tests)
# ---------------------------------------------------------------------------

class TestIntegrationRegistry:
    def test_creation(self):
        reg = IntegrationRegistry()
        assert reg is not None

    def test_is_available_numpy_true(self):
        reg = IntegrationRegistry()
        assert reg.is_available("numpy") is True

    def test_is_available_nonexistent_false(self):
        reg = IntegrationRegistry()
        assert reg.is_available("nonexistent_package_xyz_123") is False

    def test_is_available_twice_cached(self):
        reg = IntegrationRegistry()
        reg.is_available("numpy")
        reg.is_available("numpy")
        assert "numpy" in reg._cache

    def test_cache_populated_after_check(self):
        reg = IntegrationRegistry()
        reg.is_available("numpy")
        assert "numpy" in reg._cache
        assert reg._cache["numpy"] is True

    def test_unavailable_cached_false(self):
        reg = IntegrationRegistry()
        reg.is_available("totally_fake_module_abc")
        assert reg._cache.get("totally_fake_module_abc") is False

    def test_status_dict_returns_dict(self):
        reg = IntegrationRegistry()
        d = reg.status_dict()
        assert isinstance(d, dict)

    def test_status_dict_has_all_packages(self):
        reg = IntegrationRegistry()
        d = reg.status_dict()
        for pkg in _OPTIONAL_PACKAGES:
            assert pkg in d

    def test_status_dict_bool_values(self):
        reg = IntegrationRegistry()
        d = reg.status_dict()
        for v in d.values():
            assert isinstance(v, bool)

    def test_available_packages_returns_list(self):
        reg = IntegrationRegistry()
        lst = reg.available_packages()
        assert isinstance(lst, list)

    def test_unavailable_packages_returns_list(self):
        reg = IntegrationRegistry()
        lst = reg.unavailable_packages()
        assert isinstance(lst, list)

    def test_available_union_unavailable_equals_all(self):
        reg = IntegrationRegistry()
        avail = set(reg.available_packages())
        unavail = set(reg.unavailable_packages())
        all_pkgs = set(_OPTIONAL_PACKAGES)
        assert avail | unavail == all_pkgs
        assert avail & unavail == set()

    def test_available_packages_subset_of_optional(self):
        reg = IntegrationRegistry()
        for p in reg.available_packages():
            assert p in _OPTIONAL_PACKAGES

    def test_unavailable_packages_subset_of_optional(self):
        reg = IntegrationRegistry()
        for p in reg.unavailable_packages():
            assert p in _OPTIONAL_PACKAGES

    def test_get_module_numpy(self):
        reg = IntegrationRegistry()
        m = reg.get_module("numpy")
        assert m is not None

    def test_get_module_fake_returns_none(self):
        reg = IntegrationRegistry()
        m = reg.get_module("completely_fake_xyz_pkg")
        assert m is None

    def test_get_module_type(self):
        reg = IntegrationRegistry()
        m = reg.get_module("numpy")
        import types
        assert isinstance(m, types.ModuleType)

    def test_optional_packages_tuple(self):
        assert isinstance(_OPTIONAL_PACKAGES, tuple)

    def test_optional_packages_nonempty(self):
        assert len(_OPTIONAL_PACKAGES) > 0

    def test_optional_packages_all_strings(self):
        for p in _OPTIONAL_PACKAGES:
            assert isinstance(p, str)

    def test_genesis_os_in_optional(self):
        assert "genesis_os" in _OPTIONAL_PACKAGES

    def test_aeon_ai_in_optional(self):
        assert "aeon_ai" in _OPTIONAL_PACKAGES

    def test_sigillin_in_optional(self):
        assert "sigillin" in _OPTIONAL_PACKAGES

    def test_entropy_governance_in_optional(self):
        assert "entropy_governance" in _OPTIONAL_PACKAGES

    def test_utac_core_in_optional(self):
        assert "utac_core" in _OPTIONAL_PACKAGES

    def test_mandala_visualizer_in_optional(self):
        assert "mandala_visualizer" in _OPTIONAL_PACKAGES

    def test_sonification_in_optional(self):
        assert "sonification" in _OPTIONAL_PACKAGES

    def test_climate_dashboard_in_optional(self):
        assert "climate_dashboard" in _OPTIONAL_PACKAGES

    def test_implosive_genesis_in_optional(self):
        assert "implosive_genesis" in _OPTIONAL_PACKAGES

    def test_entropy_table_in_optional(self):
        assert "entropy_table" in _OPTIONAL_PACKAGES


# ---------------------------------------------------------------------------
# Contract stubs (mock tests simulating external package contracts)
# ---------------------------------------------------------------------------

class TestContractGenesiOS:
    """Contract tests: genesis_os integration (mocked)."""

    def test_genesis_os_version_attribute(self):
        mock_mod = MagicMock()
        mock_mod.__version__ = "0.2.0"
        with patch.dict(sys.modules, {"genesis_os": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("genesis_os")
            assert m.__version__ == "0.2.0"

    def test_genesis_os_has_init(self):
        mock_mod = MagicMock()
        mock_mod.GenesisOS = MagicMock()
        with patch.dict(sys.modules, {"genesis_os": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("genesis_os")
            assert hasattr(m, "GenesisOS")

    def test_genesis_os_available_when_mocked(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"genesis_os": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("genesis_os") is True


class TestContractAeonAI:
    """Contract tests: aeon_ai integration (mocked)."""

    def test_aeon_ai_version_attribute(self):
        mock_mod = MagicMock()
        mock_mod.__version__ = "0.2.0"
        with patch.dict(sys.modules, {"aeon_ai": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("aeon_ai")
            assert m.__version__ == "0.2.0"

    def test_aeon_ai_available_when_mocked(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"aeon_ai": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("aeon_ai") is True


class TestContractSigillin:
    """Contract tests: sigillin integration (mocked)."""

    def test_sigillin_has_sigil_class(self):
        mock_mod = MagicMock()
        mock_mod.Sigil = MagicMock()
        with patch.dict(sys.modules, {"sigillin": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("sigillin")
            assert hasattr(m, "Sigil")


class TestContractFieldtheory:
    """Contract tests: fieldtheory integration (mocked)."""

    def test_fieldtheory_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"fieldtheory": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("fieldtheory") is True

    def test_fieldtheory_has_lagrangian(self):
        mock_mod = MagicMock()
        mock_mod.Lagrangian = MagicMock()
        with patch.dict(sys.modules, {"fieldtheory": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("fieldtheory")
            assert hasattr(m, "Lagrangian")


class TestContractMirrorMachine:
    def test_mirror_machine_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"mirror_machine": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("mirror_machine") is True


class TestContractEntropyGovernance:
    def test_entropy_governance_policy(self):
        mock_mod = MagicMock()
        mock_mod.Policy = MagicMock()
        with patch.dict(sys.modules, {"entropy_governance": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("entropy_governance")
            assert hasattr(m, "Policy")


class TestContractUTAC:
    def test_utac_core_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"utac_core": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("utac_core") is True


class TestContractMandalaVisualizer:
    def test_mandala_visualizer_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"mandala_visualizer": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("mandala_visualizer") is True

    def test_mandala_visualizer_has_render(self):
        mock_mod = MagicMock()
        mock_mod.render = MagicMock(return_value="<svg/>")
        with patch.dict(sys.modules, {"mandala_visualizer": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("mandala_visualizer")
            assert hasattr(m, "render")


class TestContractSonification:
    def test_sonification_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"sonification": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("sonification") is True


class TestContractClimateDashboard:
    def test_climate_dashboard_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"climate_dashboard": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("climate_dashboard") is True


class TestContractImplosiveGenesis:
    def test_implosive_genesis_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"implosive_genesis": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("implosive_genesis") is True


class TestContractEntropyTable:
    def test_entropy_table_available(self):
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {"entropy_table": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            assert reg.is_available("entropy_table") is True

    def test_entropy_table_has_table(self):
        mock_mod = MagicMock()
        mock_mod.EntropyTable = MagicMock()
        with patch.dict(sys.modules, {"entropy_table": mock_mod}):
            reg = IntegrationRegistry()
            reg._cache = {}
            m = reg.get_module("entropy_table")
            assert hasattr(m, "EntropyTable")
