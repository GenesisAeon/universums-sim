"""
Tests for universums_sim.governance.entropy.

Covers:
- GovernancePolicy validation
- PolicyAction enum
- GovernanceDecision dataclass
- EntropyGovernor.evaluate() all branches
- EntropyGovernor.apply_weight()
"""

from __future__ import annotations

import pytest

from universums_sim.governance.entropy import (
    EntropyGovernor,
    GovernanceDecision,
    GovernancePolicy,
    PolicyAction,
)


# ---------------------------------------------------------------------------
# PolicyAction (5 tests)
# ---------------------------------------------------------------------------

class TestPolicyAction:
    def test_all_actions(self):
        names = {a.name for a in PolicyAction}
        assert names == {"ALLOW", "THROTTLE", "HALT", "RESET"}

    def test_allow(self):
        assert PolicyAction.ALLOW.name == "ALLOW"

    def test_throttle(self):
        assert PolicyAction.THROTTLE.name == "THROTTLE"

    def test_halt(self):
        assert PolicyAction.HALT.name == "HALT"

    def test_reset(self):
        assert PolicyAction.RESET.name == "RESET"


# ---------------------------------------------------------------------------
# GovernancePolicy (20 tests)
# ---------------------------------------------------------------------------

class TestGovernancePolicy:
    def test_default_entropy_warn(self):
        p = GovernancePolicy()
        assert p.entropy_warn == 1_000.0

    def test_default_entropy_halt(self):
        p = GovernancePolicy()
        assert p.entropy_halt == 10_000.0

    def test_default_entropy_reset(self):
        p = GovernancePolicy()
        assert p.entropy_reset == 100_000.0

    def test_default_weight(self):
        p = GovernancePolicy()
        assert p.weight_ethical == 1.0

    def test_custom_warn(self):
        p = GovernancePolicy(entropy_warn=500.0)
        assert p.entropy_warn == 500.0

    def test_custom_halt(self):
        p = GovernancePolicy(entropy_halt=5000.0)
        assert p.entropy_halt == 5000.0

    def test_custom_weight(self):
        p = GovernancePolicy(weight_ethical=2.0)
        assert p.weight_ethical == 2.0

    def test_invalid_warn_zero(self):
        with pytest.raises(Exception):
            GovernancePolicy(entropy_warn=0.0)

    def test_invalid_halt_zero(self):
        with pytest.raises(Exception):
            GovernancePolicy(entropy_halt=0.0)

    def test_invalid_weight_zero(self):
        with pytest.raises(Exception):
            GovernancePolicy(weight_ethical=0.0)

    def test_invalid_weight_above_max(self):
        with pytest.raises(Exception):
            GovernancePolicy(weight_ethical=11.0)

    def test_frozen(self):
        p = GovernancePolicy()
        with pytest.raises(Exception):
            p.entropy_warn = 999.0  # type: ignore[misc]

    def test_weight_max_boundary(self):
        p = GovernancePolicy(weight_ethical=10.0)
        assert p.weight_ethical == 10.0

    def test_weight_min_boundary(self):
        p = GovernancePolicy(weight_ethical=0.001)
        assert p.weight_ethical == 0.001

    def test_large_reset(self):
        p = GovernancePolicy(entropy_reset=1e9)
        assert p.entropy_reset == 1e9

    def test_small_warn(self):
        p = GovernancePolicy(entropy_warn=0.1)
        assert p.entropy_warn == 0.1

    def test_warn_lt_halt_not_enforced(self):
        # Policy doesn't enforce ordering; user responsibility
        p = GovernancePolicy(entropy_warn=9999.0, entropy_halt=1000.0)
        assert p.entropy_warn > p.entropy_halt

    def test_policy_repr(self):
        p = GovernancePolicy()
        assert "GovernancePolicy" in repr(p)

    def test_policy_dict(self):
        p = GovernancePolicy()
        d = p.model_dump()
        assert "entropy_warn" in d

    def test_policy_equality(self):
        p1 = GovernancePolicy()
        p2 = GovernancePolicy()
        assert p1 == p2


# ---------------------------------------------------------------------------
# GovernanceDecision (10 tests)
# ---------------------------------------------------------------------------

class TestGovernanceDecision:
    def _make_decision(self, **kwargs):
        defaults = dict(
            action=PolicyAction.ALLOW,
            entropy=50.0,
            reason="test",
            metadata={},
        )
        defaults.update(kwargs)
        return GovernanceDecision(**defaults)

    def test_creation(self):
        d = self._make_decision()
        assert d.action == PolicyAction.ALLOW

    def test_entropy_stored(self):
        d = self._make_decision(entropy=123.0)
        assert d.entropy == 123.0

    def test_reason_stored(self):
        d = self._make_decision(reason="because")
        assert d.reason == "because"

    def test_metadata_stored(self):
        d = self._make_decision(metadata={"k": "v"})
        assert d.metadata["k"] == "v"

    def test_to_dict_keys(self):
        d = self._make_decision()
        assert set(d.to_dict().keys()) >= {"action", "entropy", "reason", "metadata"}

    def test_to_dict_action_is_string(self):
        d = self._make_decision(action=PolicyAction.HALT)
        assert d.to_dict()["action"] == "HALT"

    def test_frozen(self):
        d = self._make_decision()
        with pytest.raises(Exception):
            d.entropy = 999.0  # type: ignore[misc]

    def test_slots(self):
        d = self._make_decision()
        assert not hasattr(d, "__dict__")

    def test_to_dict_json_safe(self):
        import json
        d = self._make_decision()
        json.dumps(d.to_dict())

    def test_throttle_metadata(self):
        d = self._make_decision(action=PolicyAction.THROTTLE, metadata={"throttle_factor": 0.5})
        assert d.metadata["throttle_factor"] == 0.5


# ---------------------------------------------------------------------------
# EntropyGovernor (40 tests)
# ---------------------------------------------------------------------------

class TestEntropyGovernorInit:
    def test_init_default(self):
        g = EntropyGovernor()
        assert g is not None

    def test_init_with_policy(self, default_policy):
        g = EntropyGovernor(default_policy)
        assert g.policy is default_policy

    def test_policy_property(self, governor):
        assert isinstance(governor.policy, GovernancePolicy)

    def test_policy_frozen(self, governor):
        with pytest.raises(Exception):
            governor.policy.entropy_warn = 1.0  # type: ignore[misc]


class TestEntropyGovernorEvaluate:
    def test_allow_below_warn(self, governor):
        d = governor.evaluate(entropy=50.0)
        assert d.action == PolicyAction.ALLOW

    def test_allow_zero_entropy(self, governor):
        d = governor.evaluate(entropy=0.0)
        assert d.action == PolicyAction.ALLOW

    def test_throttle_at_warn(self, governor):
        d = governor.evaluate(entropy=100.0)
        assert d.action == PolicyAction.THROTTLE

    def test_throttle_above_warn(self, governor):
        d = governor.evaluate(entropy=200.0)
        assert d.action == PolicyAction.THROTTLE

    def test_halt_at_halt(self, governor):
        d = governor.evaluate(entropy=1000.0)
        assert d.action == PolicyAction.HALT

    def test_halt_above_halt(self, governor):
        d = governor.evaluate(entropy=5000.0)
        assert d.action == PolicyAction.HALT

    def test_reset_at_reset(self, governor):
        d = governor.evaluate(entropy=10000.0)
        assert d.action == PolicyAction.RESET

    def test_reset_above_reset(self, governor):
        d = governor.evaluate(entropy=1e6)
        assert d.action == PolicyAction.RESET

    def test_allow_headroom_in_metadata(self, governor):
        d = governor.evaluate(entropy=50.0)
        assert "headroom" in d.metadata

    def test_throttle_factor_in_metadata(self, governor):
        d = governor.evaluate(entropy=500.0)
        assert "throttle_factor" in d.metadata

    def test_decision_entropy_matches(self, governor):
        d = governor.evaluate(entropy=77.0)
        assert d.entropy == 77.0

    def test_decision_reason_nonempty(self, governor):
        d = governor.evaluate(entropy=50.0)
        assert len(d.reason) > 0

    def test_allow_returns_governance_decision(self, governor):
        d = governor.evaluate(entropy=1.0)
        assert isinstance(d, GovernanceDecision)

    def test_halt_returns_governance_decision(self, governor):
        d = governor.evaluate(entropy=5000.0)
        assert isinstance(d, GovernanceDecision)

    def test_reset_returns_governance_decision(self, governor):
        d = governor.evaluate(entropy=1e5)
        assert isinstance(d, GovernanceDecision)


class TestApplyWeight:
    def test_allow_returns_weighted_rate(self, governor):
        rate = governor.apply_weight(1.0, entropy=50.0)
        assert rate == pytest.approx(1.0 * governor.policy.weight_ethical)

    def test_halt_returns_zero(self, governor):
        rate = governor.apply_weight(1.0, entropy=5000.0)
        assert rate == 0.0

    def test_reset_returns_zero(self, governor):
        rate = governor.apply_weight(1.0, entropy=1e6)
        assert rate == 0.0

    def test_throttle_reduces_rate(self, governor):
        rate = governor.apply_weight(1.0, entropy=500.0)
        assert 0.0 <= rate <= 1.0

    def test_zero_rate_unchanged(self, governor):
        rate = governor.apply_weight(0.0, entropy=50.0)
        assert rate == 0.0

    def test_apply_weight_non_negative(self, governor):
        for entropy in [1.0, 100.0, 1000.0, 5000.0, 1e5]:
            assert governor.apply_weight(1.0, entropy) >= 0.0

    def test_custom_weight_applied(self):
        p = GovernancePolicy(weight_ethical=2.0, entropy_warn=1000.0)
        g = EntropyGovernor(p)
        rate = g.apply_weight(1.0, entropy=5.0)
        assert rate == pytest.approx(2.0)

    def test_throttle_factor_proportional(self):
        p = GovernancePolicy(entropy_warn=100.0, entropy_halt=1000.0, weight_ethical=1.0)
        g = EntropyGovernor(p)
        rate1 = g.apply_weight(1.0, entropy=200.0)
        rate2 = g.apply_weight(1.0, entropy=500.0)
        # Higher entropy within throttle zone should give lower or equal rate
        assert rate2 <= rate1 + 1e-9
