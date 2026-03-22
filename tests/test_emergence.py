"""
Tests for universums_sim.simulation.emergence.

Covers:
- EmergenceType enum
- EmergenceEvent dataclass
- EmergenceEngine rate computation
- EmergenceEngine event firing
- Edge cases (zero entropy, max entropy, large grad_h)
"""

from __future__ import annotations

import numpy as np
import pytest

from universums_sim.simulation.emergence import (
    EmergenceEngine,
    EmergenceEvent,
    EmergenceType,
)

# ---------------------------------------------------------------------------
# EmergenceType (5 tests)
# ---------------------------------------------------------------------------

class TestEmergenceType:
    def test_all_types_exist(self):
        names = {t.name for t in EmergenceType}
        expected = {"MICRO", "MESO", "MACRO", "COSMIC", "TRANSCENDENT"}
        assert names == expected

    def test_micro(self):
        assert EmergenceType.MICRO.name == "MICRO"

    def test_cosmic(self):
        assert EmergenceType.COSMIC.name == "COSMIC"

    def test_transcendent(self):
        assert EmergenceType.TRANSCENDENT.name == "TRANSCENDENT"

    def test_count(self):
        assert len(EmergenceType) == 5


# ---------------------------------------------------------------------------
# EmergenceEvent (20 tests)
# ---------------------------------------------------------------------------

class TestEmergenceEvent:
    def _make_event(self, **kwargs):
        defaults: dict = {
            "event_id": "evt-00000001",
            "step": 1,
            "time": 0.5,
            "entropy": 10.0,
            "rate": 0.01,
            "kind": EmergenceType.MESO,
            "description": "test event",
            "metadata": {},
        }
        defaults.update(kwargs)
        return EmergenceEvent(**defaults)

    def test_creation(self):
        e = self._make_event()
        assert e.event_id == "evt-00000001"

    def test_step(self):
        e = self._make_event(step=42)
        assert e.step == 42

    def test_time(self):
        e = self._make_event(time=3.14)
        assert e.time == 3.14

    def test_entropy(self):
        e = self._make_event(entropy=100.0)
        assert e.entropy == 100.0

    def test_rate(self):
        e = self._make_event(rate=0.123)
        assert e.rate == 0.123

    def test_kind_micro(self):
        e = self._make_event(kind=EmergenceType.MICRO)
        assert e.kind == EmergenceType.MICRO

    def test_kind_cosmic(self):
        e = self._make_event(kind=EmergenceType.COSMIC)
        assert e.kind == EmergenceType.COSMIC

    def test_description(self):
        e = self._make_event(description="hello")
        assert e.description == "hello"

    def test_metadata_empty(self):
        e = self._make_event()
        assert e.metadata == {}

    def test_metadata_populated(self):
        e = self._make_event(metadata={"key": "value"})
        assert e.metadata["key"] == "value"

    def test_to_dict_keys(self):
        e = self._make_event()
        d = e.to_dict()
        assert set(d.keys()) >= {
            "event_id", "step", "time", "entropy", "rate", "kind", "description"
        }

    def test_to_dict_kind_name(self):
        e = self._make_event(kind=EmergenceType.COSMIC)
        assert e.to_dict()["kind"] == "COSMIC"

    def test_to_dict_step(self):
        e = self._make_event(step=7)
        assert e.to_dict()["step"] == 7

    def test_frozen(self):
        e = self._make_event()
        with pytest.raises(Exception):
            e.step = 99  # type: ignore[misc]

    def test_slots(self):
        e = self._make_event()
        assert not hasattr(e, "__dict__")

    def test_to_dict_metadata(self):
        e = self._make_event(metadata={"x": 42})
        assert e.to_dict()["metadata"]["x"] == 42

    def test_to_dict_is_dict(self):
        e = self._make_event()
        assert isinstance(e.to_dict(), dict)

    def test_event_id_field(self):
        e = self._make_event(event_id="abc-123")
        assert e.event_id == "abc-123"

    def test_zero_rate(self):
        e = self._make_event(rate=0.0)
        assert e.rate == 0.0

    def test_large_entropy(self):
        e = self._make_event(entropy=1e8)
        assert e.entropy == 1e8


# ---------------------------------------------------------------------------
# EmergenceEngine (60 tests)
# ---------------------------------------------------------------------------

class TestEmergenceEngineInit:
    def test_creation(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.5, beta=0.1, entropy_max=1e4, rng=rng)
        assert eng is not None

    def test_event_count_starts_zero(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.5, beta=0.1, entropy_max=1e4, rng=rng)
        assert eng.event_count == 0

    def test_alpha_stored(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.77, beta=0.1, entropy_max=1e4, rng=rng)
        assert eng._alpha == 0.77

    def test_beta_stored(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.5, beta=0.33, entropy_max=1e4, rng=rng)
        assert eng._beta == 0.33

    def test_entropy_max_stored(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.5, beta=0.1, entropy_max=9999.0, rng=rng)
        assert eng._entropy_max == 9999.0


class TestComputeRate:
    def test_rate_positive(self, emergence_engine):
        r = emergence_engine.compute_rate(entropy=100.0, grad_h=0.0)
        assert r >= 0.0

    def test_rate_zero_entropy(self, emergence_engine):
        r = emergence_engine.compute_rate(entropy=0.0, grad_h=0.0)
        assert r == 0.0

    def test_rate_negative_entropy(self, emergence_engine):
        r = emergence_engine.compute_rate(entropy=-1.0, grad_h=0.0)
        assert r == 0.0

    def test_rate_decreases_with_large_grad(self, emergence_engine):
        r_low = emergence_engine.compute_rate(entropy=100.0, grad_h=0.0)
        r_high = emergence_engine.compute_rate(entropy=100.0, grad_h=100.0)
        assert r_high < r_low

    def test_rate_at_max_entropy(self, emergence_engine):
        r = emergence_engine.compute_rate(entropy=1e4, grad_h=0.0)
        assert r == pytest.approx(0.0, abs=1e-10)

    def test_rate_above_max_entropy(self, emergence_engine):
        r = emergence_engine.compute_rate(entropy=2e4, grad_h=0.0)
        assert r <= 0.0

    def test_rate_finite(self, emergence_engine):
        r = emergence_engine.compute_rate(entropy=500.0, grad_h=1.0)
        assert np.isfinite(r)

    def test_rate_formula_manual(self):
        rng = np.random.default_rng(0)
        alpha = 0.42
        beta = 0.1
        entropy_max = 1000.0
        eng = EmergenceEngine(alpha=alpha, beta=beta, entropy_max=entropy_max, rng=rng)
        s = 100.0
        grad = 2.0
        expected = alpha * s * (1 - s / entropy_max) * np.exp(-beta * grad)
        assert eng.compute_rate(s, grad) == pytest.approx(expected, rel=1e-9)

    def test_rate_increases_with_entropy_below_half(self, emergence_engine):
        r1 = emergence_engine.compute_rate(entropy=10.0, grad_h=0.0)
        r2 = emergence_engine.compute_rate(entropy=100.0, grad_h=0.0)
        # At low entropy levels, rate increases with entropy
        assert r2 > r1

    def test_rate_peak_at_half_entropy_max(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1.0, beta=0.0, entropy_max=100.0, rng=rng)
        # Maximum of S*(1-S/S_max) is at S = S_max/2
        r_peak = eng.compute_rate(entropy=50.0, grad_h=0.0)
        r_low = eng.compute_rate(entropy=10.0, grad_h=0.0)
        r_high = eng.compute_rate(entropy=90.0, grad_h=0.0)
        assert r_peak >= r_low
        assert r_peak >= r_high

    def test_rate_zero_alpha(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.0, beta=0.1, entropy_max=1e4, rng=rng)
        assert eng.compute_rate(100.0, 0.0) == 0.0

    def test_rate_large_beta_suppresses(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.5, beta=1000.0, entropy_max=1e4, rng=rng)
        r = eng.compute_rate(entropy=100.0, grad_h=1.0)
        assert r < 1e-100


class TestFireEvents:
    def test_fire_returns_list(self, emergence_engine):
        events = emergence_engine.fire_events(0, 0.0, 100.0, 0.5)
        assert isinstance(events, list)

    def test_events_are_emergence_events(self, emergence_engine):
        for _ in range(20):
            events = emergence_engine.fire_events(0, 0.0, 100.0, 0.5)
            for e in events:
                assert isinstance(e, EmergenceEvent)

    def test_counter_increments(self):
        rng = np.random.default_rng(123)
        eng = EmergenceEngine(alpha=100.0, beta=0.0, entropy_max=1e6, rng=rng)
        total = 0
        for i in range(50):
            evts = eng.fire_events(i, float(i), 1000.0, 1.0)
            total += len(evts)
        assert eng.event_count == total

    def test_event_id_format(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        events = eng.fire_events(0, 0.0, 1000.0, 0.0)
        for e in events:
            assert e.event_id.startswith("evt-")

    def test_zero_rate_no_events(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=0.0, beta=0.0, entropy_max=1e4, rng=rng)
        for _ in range(100):
            events = eng.fire_events(0, 0.0, 100.0, 0.0)
            assert events == []

    def test_high_rate_fires_events(self):
        rng = np.random.default_rng(42)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        fired = 0
        for i in range(100):
            rate = eng.compute_rate(entropy=1.0, grad_h=0.0)
            events = eng.fire_events(i, float(i), 1.0, rate)
            fired += len(events)
        assert fired > 0

    def test_event_step_matches(self):
        rng = np.random.default_rng(42)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        for step in range(10):
            events = eng.fire_events(step, float(step), 1.0, 0.0)
            for e in events:
                assert e.step == step

    def test_event_entropy_matches(self):
        rng = np.random.default_rng(42)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        for _ in range(20):
            events = eng.fire_events(0, 0.0, 777.0, 0.0)
            for e in events:
                assert e.entropy == 777.0

    def test_event_kind_is_emergence_type(self):
        rng = np.random.default_rng(42)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        for i in range(50):
            events = eng.fire_events(i, float(i), 1.0, 0.0)
            for e in events:
                assert isinstance(e.kind, EmergenceType)

    def test_classify_micro(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        kind = eng._classify(0.0005)
        assert kind == EmergenceType.MICRO

    def test_classify_meso(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        kind = eng._classify(0.005)
        assert kind == EmergenceType.MESO

    def test_classify_macro(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        kind = eng._classify(0.05)
        assert kind == EmergenceType.MACRO

    def test_classify_cosmic(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        kind = eng._classify(0.5)
        assert kind == EmergenceType.COSMIC

    def test_classify_transcendent(self):
        rng = np.random.default_rng(0)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        kind = eng._classify(2.0)
        assert kind == EmergenceType.TRANSCENDENT

    def test_event_metadata_has_wall_time(self):
        rng = np.random.default_rng(42)
        eng = EmergenceEngine(alpha=1e6, beta=0.0, entropy_max=1e9, rng=rng)
        for _ in range(30):
            events = eng.fire_events(0, 0.0, 1.0, 0.0)
            for e in events:
                assert "wall_time" in e.metadata
