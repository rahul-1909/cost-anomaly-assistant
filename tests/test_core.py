"""
test_core.py -- Unit tests for the deterministic core logic.

Tests cover:
- Week assignment (Mondays, Sundays, year boundaries)
- Cost formula (aggregate, not per-shipment ratio average)
- 8-week lookback (strictly before, fewer than 8 weeks, first week)
- Peer average excluding self
- Note matching (wrong route, wrong window, no-impact notes)
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loading import load_config, load_shipments
from src.metrics import (
    compute_weekly_cost,
    compute_vs_own_history,
    compute_vs_similar_routes,
    compute_percentage_change,
    format_vs_own_history,
    format_vs_similar_routes,
)
from src.flagging import should_flag
from src.retrieval import (
    note_applies_to_route,
    note_date_applies_to_week,
    note_text_applies_to_week,
    note_indicates_cost_increase,
)


# ============================================================
# Week assignment tests
# ============================================================

class TestWeekAssignment:
    """Tests for correct week_of (Monday) assignment."""

    def _make_shipment(self, date_str: str) -> pd.DataFrame:
        """Create a minimal shipment DataFrame with a single record."""
        return pd.DataFrame({
            "shipment_id": ["TEST001"],
            "origin": ["Mumbai"],
            "destination": ["Pune"],
            "route_type": ["Short"],
            "material": ["Cement"],
            "quantity_tonnes": [10.0],
            "distance_km": [150.0],
            "freight_cost_inr": [5000],
            "shipment_date": [date_str],
            "transporter": ["Test Co"],
        })

    def test_monday_stays_monday(self):
        """A shipment on Monday should have week_of = that same Monday."""
        df = self._make_shipment("2024-01-01")  # Monday
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        df["week_of"] = df["shipment_date"] - pd.to_timedelta(
            df["shipment_date"].dt.weekday, unit="D"
        )
        assert df["week_of"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_sunday_maps_to_prior_monday(self):
        """A shipment on Sunday should map to the preceding Monday."""
        df = self._make_shipment("2024-01-07")  # Sunday
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        df["week_of"] = df["shipment_date"] - pd.to_timedelta(
            df["shipment_date"].dt.weekday, unit="D"
        )
        assert df["week_of"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_wednesday_maps_to_monday(self):
        """A mid-week shipment should map to the Monday of that week."""
        df = self._make_shipment("2024-01-03")  # Wednesday
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        df["week_of"] = df["shipment_date"] - pd.to_timedelta(
            df["shipment_date"].dt.weekday, unit="D"
        )
        assert df["week_of"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_year_boundary(self):
        """A Sunday at end of 2024 should still map to the correct Monday."""
        df = self._make_shipment("2024-12-29")  # Sunday
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        df["week_of"] = df["shipment_date"] - pd.to_timedelta(
            df["shipment_date"].dt.weekday, unit="D"
        )
        # Dec 29, 2024 is a Sunday; Monday of that week is Dec 23, 2024
        assert df["week_of"].iloc[0] == pd.Timestamp("2024-12-23")

    def test_year_boundary_new_year(self):
        """Jan 1, 2025 (Wednesday) should map to Dec 30, 2024 (Monday)."""
        df = self._make_shipment("2025-01-01")  # Wednesday
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        df["week_of"] = df["shipment_date"] - pd.to_timedelta(
            df["shipment_date"].dt.weekday, unit="D"
        )
        assert df["week_of"].iloc[0] == pd.Timestamp("2024-12-30")

    def test_saturday_maps_to_monday(self):
        """Saturday should map to the preceding Monday."""
        df = self._make_shipment("2024-01-06")  # Saturday
        df["shipment_date"] = pd.to_datetime(df["shipment_date"])
        df["week_of"] = df["shipment_date"] - pd.to_timedelta(
            df["shipment_date"].dt.weekday, unit="D"
        )
        assert df["week_of"].iloc[0] == pd.Timestamp("2024-01-01")


# ============================================================
# Cost formula tests
# ============================================================

class TestCostFormula:
    """Tests for the cost_per_tonne_km computation."""

    def test_single_shipment(self):
        """Cost formula for a single shipment."""
        df = pd.DataFrame({
            "shipment_id": ["S1"],
            "origin": ["A"],
            "destination": ["B"],
            "route_type": ["Short"],
            "route": ["A-B"],
            "quantity_tonnes": [10.0],
            "distance_km": [100.0],
            "freight_cost_inr": [5000],
            "shipment_date": pd.to_datetime(["2024-01-01"]),
            "week_of": pd.to_datetime(["2024-01-01"]),
        })
        weekly = compute_weekly_cost(df)
        # 5000 / (10 * 100) = 5.0
        assert np.isclose(weekly["cost_per_tonne_km"].iloc[0], 5.0)

    def test_aggregate_not_average(self):
        """
        Multiple shipments in same route-week must be aggregated
        (sum/sum), not averaged from per-shipment ratios.
        """
        df = pd.DataFrame({
            "shipment_id": ["S1", "S2"],
            "origin": ["A", "A"],
            "destination": ["B", "B"],
            "route_type": ["Short", "Short"],
            "route": ["A-B", "A-B"],
            "quantity_tonnes": [10.0, 20.0],
            "distance_km": [100.0, 200.0],
            "freight_cost_inr": [5000, 10000],
            "shipment_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "week_of": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        })
        weekly = compute_weekly_cost(df)
        # Aggregate: (5000 + 10000) / (10*100 + 20*200) = 15000 / 5000 = 3.0
        assert np.isclose(weekly["cost_per_tonne_km"].iloc[0], 3.0)

        # Per-shipment average would be: (5000/(10*100) + 10000/(20*200)) / 2
        # = (5.0 + 2.5) / 2 = 3.75 -- WRONG
        assert not np.isclose(weekly["cost_per_tonne_km"].iloc[0], 3.75)

    def test_shipment_count(self):
        """Shipment count should be tracked correctly."""
        df = pd.DataFrame({
            "shipment_id": ["S1", "S2", "S3"],
            "origin": ["A", "A", "A"],
            "destination": ["B", "B", "B"],
            "route_type": ["Short", "Short", "Short"],
            "route": ["A-B", "A-B", "A-B"],
            "quantity_tonnes": [10.0, 15.0, 20.0],
            "distance_km": [100.0, 100.0, 100.0],
            "freight_cost_inr": [5000, 7500, 10000],
            "shipment_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "week_of": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        })
        weekly = compute_weekly_cost(df)
        assert weekly["shipment_count"].iloc[0] == 3


# ============================================================
# Lookback tests
# ============================================================

class TestLookback:
    """Tests for the trailing 8-week rolling average."""

    def _make_weekly(self, n_weeks: int, route: str = "A-B") -> pd.DataFrame:
        """Create a weekly DataFrame with n_weeks of data."""
        dates = pd.date_range("2024-01-01", periods=n_weeks, freq="7D")
        costs = [2.0 + 0.1 * i for i in range(n_weeks)]
        return pd.DataFrame({
            "route": [route] * n_weeks,
            "route_type": ["Short"] * n_weeks,
            "week_of": dates,
            "cost_per_tonne_km": costs,
        })

    def test_first_week_no_history(self):
        """First week should have no own history (NaN)."""
        weekly = self._make_weekly(10)
        result = compute_vs_own_history(weekly, window=8)
        assert np.isnan(result["own_history_avg"].iloc[0])

    def test_strictly_before(self):
        """Lookback must use only weeks strictly before current week."""
        weekly = self._make_weekly(3)
        result = compute_vs_own_history(weekly, window=8)
        # Week 2 (index 1) should use only week 1 (index 0)
        assert np.isclose(result["own_history_avg"].iloc[1], 2.0)
        # Week 3 (index 2) should use weeks 1 and 2
        expected = np.mean([2.0, 2.1])
        assert np.isclose(result["own_history_avg"].iloc[2], expected)

    def test_fewer_than_8_weeks(self):
        """With fewer than 8 prior weeks, use all available prior weeks."""
        weekly = self._make_weekly(5)
        result = compute_vs_own_history(weekly, window=8)
        # Week 5 (index 4) should use all 4 prior weeks
        expected = np.mean([2.0, 2.1, 2.2, 2.3])
        assert np.isclose(result["own_history_avg"].iloc[4], expected)

    def test_full_8_week_window(self):
        """With 8+ prior weeks, use exactly the last 8."""
        weekly = self._make_weekly(12)
        result = compute_vs_own_history(weekly, window=8)
        # Week 12 (index 11) should use weeks 4-11 (indices 3-10)
        expected = np.mean([2.0 + 0.1 * i for i in range(3, 11)])
        assert np.isclose(result["own_history_avg"].iloc[11], expected)

    def test_no_look_ahead(self):
        """Verify that future weeks are never used."""
        weekly = self._make_weekly(10)
        # Set a very high cost in the last week
        weekly.loc[weekly.index[-1], "cost_per_tonne_km"] = 100.0
        result = compute_vs_own_history(weekly, window=8)
        # The second-to-last week's average should NOT include the last week
        penultimate_avg = result["own_history_avg"].iloc[-2]
        assert penultimate_avg < 10.0  # Should be around 2.x, not affected by 100.0

    def test_missing_week_does_not_extend_calendar_window(self):
        weekly = pd.DataFrame({
            "route": ["A-B"] * 3,
            "route_type": ["Short"] * 3,
            "week_of": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-03-04"]),
            "cost_per_tonne_km": [2.0, 4.0, 3.0],
        })
        result = compute_vs_own_history(weekly, window=8)
        # Jan 1 is more than eight calendar weeks before Mar 4.
        assert np.isclose(result.loc[2, "own_history_avg"], 4.0)


# ============================================================
# Peer average tests
# ============================================================

class TestPeerAverage:
    """Tests for the peer average computation."""

    def test_excludes_self(self):
        """Peer average must exclude the route itself."""
        weekly = pd.DataFrame({
            "route": ["A-B", "C-D", "E-F"],
            "route_type": ["Short", "Short", "Short"],
            "week_of": pd.to_datetime(["2024-01-01"] * 3),
            "cost_per_tonne_km": [5.0, 3.0, 4.0],
        })
        result = compute_vs_similar_routes(weekly)
        # A-B's peer avg should be mean of C-D and E-F = (3.0 + 4.0) / 2 = 3.5
        assert np.isclose(result.loc[result["route"] == "A-B", "peer_avg"].iloc[0], 3.5)

    def test_different_route_types_excluded(self):
        """Routes with different route_type should not be peers."""
        weekly = pd.DataFrame({
            "route": ["A-B", "C-D", "E-F"],
            "route_type": ["Short", "Medium", "Short"],
            "week_of": pd.to_datetime(["2024-01-01"] * 3),
            "cost_per_tonne_km": [5.0, 3.0, 4.0],
        })
        result = compute_vs_similar_routes(weekly)
        # A-B (Short) peer avg should only include E-F (Short) = 4.0
        assert np.isclose(result.loc[result["route"] == "A-B", "peer_avg"].iloc[0], 4.0)

    def test_single_route_in_type(self):
        """A route that is the only one of its type has NaN peer avg."""
        weekly = pd.DataFrame({
            "route": ["A-B", "C-D"],
            "route_type": ["Short", "Medium"],
            "week_of": pd.to_datetime(["2024-01-01"] * 2),
            "cost_per_tonne_km": [5.0, 3.0],
        })
        result = compute_vs_similar_routes(weekly)
        # Both routes are alone in their type
        assert np.isnan(result.loc[result["route"] == "A-B", "peer_avg"].iloc[0])
        assert np.isnan(result.loc[result["route"] == "C-D", "peer_avg"].iloc[0])


# ============================================================
# Percentage change and formatting tests
# ============================================================

class TestPercentageChange:
    """Tests for percentage change computation."""

    def test_positive_change(self):
        result = compute_percentage_change(135.5, 100.0)
        assert np.isclose(result, 35.5)

    def test_negative_change(self):
        result = compute_percentage_change(80.0, 100.0)
        assert np.isclose(result, -20.0)

    def test_zero_baseline(self):
        result = compute_percentage_change(5.0, 0.0)
        assert result is None

    def test_nan_baseline(self):
        result = compute_percentage_change(5.0, float("nan"))
        assert result is None

    def test_none_baseline(self):
        result = compute_percentage_change(5.0, None)
        assert result is None


class TestFormatting:
    """Tests for output string formatting."""

    def test_positive_own_history(self):
        result = format_vs_own_history(35.5)
        assert result == "+35.5% vs this route's past average"

    def test_negative_own_history(self):
        result = format_vs_own_history(-10.3)
        assert result == "-10.3% vs this route's past average"

    def test_empty_own_history(self):
        result = format_vs_own_history(None)
        assert result == ""

    def test_positive_similar_routes(self):
        result = format_vs_similar_routes(21.0)
        assert result == "+21.0% vs similar-length routes this week"

    def test_negative_similar_routes(self):
        result = format_vs_similar_routes(-5.2)
        assert result == "-5.2% vs similar-length routes this week"

    def test_rounded_negative_zero_is_rendered_as_zero(self):
        assert format_vs_own_history(-0.04) == "0.0% vs this route's past average"
        assert format_vs_similar_routes(-0.04) == "0.0% vs similar-length routes this week"


# ============================================================
# Flagging tests
# ============================================================

class TestFlagging:
    """Tests for the flag decision logic."""

    def _default_config(self) -> dict:
        return {
            "flag_own_history_rising_threshold": 0.0,
            "flag_own_history_outlier_threshold": 15.0,
            "flag_peer_outlier_threshold": 15.0,
        }

    def test_rising_and_outlier_flags(self):
        """Rising above 15% own history should flag."""
        assert should_flag(20.0, 10.0, self._default_config()) is True

    def test_rising_but_below_threshold_no_flag(self):
        """Rising but below both thresholds should not flag."""
        assert should_flag(5.0, 5.0, self._default_config()) is False

    def test_declining_no_flag(self):
        """Declining cost should never flag."""
        assert should_flag(-10.0, 25.0, self._default_config()) is False

    def test_no_history_no_flag(self):
        """No own history should not flag."""
        assert should_flag(None, 25.0, self._default_config()) is False

    def test_peer_outlier_flags(self):
        """Below own history threshold but above peer threshold should flag."""
        assert should_flag(10.0, 20.0, self._default_config()) is True

    def test_nan_history_no_flag(self):
        """NaN own history should not flag."""
        assert should_flag(float("nan"), 25.0, self._default_config()) is False

    def test_real_config_threshold_boundaries(self):
        """The configured strict thresholds control own and peer flag paths."""
        config = load_config(PROJECT_ROOT / "config.yaml")
        rising = config["flag_own_history_rising_threshold"]
        own_limit = config["flag_own_history_outlier_threshold"]
        peer_limit = config["flag_peer_outlier_threshold"]

        assert should_flag(own_limit, peer_limit, config) is False
        assert should_flag(own_limit + 0.01, peer_limit, config) is True
        assert should_flag(rising + 1.0, peer_limit, config) is False
        assert should_flag(rising + 1.0, peer_limit + 0.01, config) is True


def test_load_shipments_rejects_multiple_route_types(monkeypatch):
    shipments = pd.DataFrame({
        "shipment_id": ["S1", "S2"],
        "origin": ["A", "A"],
        "destination": ["B", "B"],
        "route_type": ["Short", "Medium"],
        "material": ["Cement", "Cement"],
        "quantity_tonnes": [10.0, 20.0],
        "distance_km": [100.0, 100.0],
        "freight_cost_inr": [5000.0, 10000.0],
        "shipment_date": ["2025-01-06", "2025-01-07"],
        "transporter": ["One", "Two"],
    })
    monkeypatch.setattr("src.loading.pd.read_csv", lambda _path: shipments.copy())

    with pytest.raises(AssertionError, match="exactly one route_type"):
        load_shipments(PROJECT_ROOT / "unused.csv")


# ============================================================
# Note matching tests
# ============================================================

class TestNoteMatching:
    """Tests for the deterministic note matching rules."""

    def test_route_match_exact(self):
        assert note_applies_to_route("Chennai-Bangalore", "Chennai-Bangalore") is True

    def test_route_match_all_routes(self):
        assert note_applies_to_route("All Routes", "Mumbai-Pune") is True

    def test_route_no_match(self):
        assert note_applies_to_route("Chennai-Bangalore", "Mumbai-Delhi") is False

    def test_wrong_route_rejected(self):
        """A note for one route must not match a different route."""
        assert note_applies_to_route("Delhi-Jaipur", "Chennai-Bangalore") is False

    def test_note_without_duration_applies_only_to_its_calendar_week(self):
        note_date = pd.Timestamp("2025-02-24")
        week_monday = pd.Timestamp("2025-02-24")
        assert note_date_applies_to_week(note_date, week_monday) is True
        assert note_date_applies_to_week(note_date, pd.Timestamp("2025-03-03")) is False

    def test_date_in_future(self):
        """A note dated after the week's Sunday should not match."""
        note_date = pd.Timestamp("2025-03-10")
        week_monday = pd.Timestamp("2025-02-24")
        # Sunday is Mar 2, note is Mar 10 -- after Sunday
        assert note_date_applies_to_week(note_date, week_monday) is False

    def test_explicit_date_range_controls_applicability(self):
        note_date = pd.Timestamp("2025-02-24")
        text = "Disrupted from Feb 24 to Mar 8, forcing longer detours and higher costs."
        assert note_text_applies_to_week(text, note_date, pd.Timestamp("2025-03-03"))
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-03-10"))

    def test_stated_duration_controls_applicability(self):
        note_date = pd.Timestamp("2025-02-24")
        text = "The disruption lasted for about a week and raised transport costs."
        assert note_text_applies_to_week(text, note_date, pd.Timestamp("2025-02-24"))
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-03-03"))

    def test_long_stated_duration_can_apply_beyond_two_weeks(self):
        note_date = pd.Timestamp("2025-01-01")
        text = "A disruption increased transport costs for two months."
        assert note_text_applies_to_week(text, note_date, pd.Timestamp("2025-02-24"))
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-03-03"))

    def test_festival_week_duration(self):
        note_date = pd.Timestamp("2025-01-20")
        text = "A regional festival week saw a temporary surcharge."
        assert note_text_applies_to_week(text, note_date, pd.Timestamp("2025-01-20"))
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-01-27"))

    def test_quarter_duration(self):
        note_date = pd.Timestamp("2025-09-22")
        text = "Demand remained stable this quarter with no major disruptions."
        assert note_text_applies_to_week(text, note_date, pd.Timestamp("2025-09-22"))
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-10-06"))

    def test_this_week_duration(self):
        note_date = pd.Timestamp("2025-05-05")
        text = "Diesel prices rose nationwide starting this week."
        assert note_text_applies_to_week(text, note_date, pd.Timestamp("2025-05-05"))
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-05-12"))

    def test_duration_does_not_allow_future_note(self):
        note_date = pd.Timestamp("2025-02-24")
        text = "The disruption lasted for two weeks and raised transport costs."
        assert not note_text_applies_to_week(text, note_date, pd.Timestamp("2025-02-17"))

    def test_no_impact_note_rejected(self):
        """Notes saying costs were not affected must be rejected."""
        text = "costs were not significantly affected"
        assert note_indicates_cost_increase(text) is False

    def test_stable_demand_rejected(self):
        """Notes about stable demand should be rejected."""
        text = "overall demand remained stable with no major disruptions reported"
        assert note_indicates_cost_increase(text) is False

    def test_no_rate_change_rejected(self):
        """Notes about costs absorbed without rate change should be rejected."""
        text = "compliance costs were absorbed by transporters without a rate change"
        assert note_indicates_cost_increase(text) is False

    def test_improved_roads_rejected(self):
        """Notes about improved road conditions should be rejected."""
        text = "improved road conditions on the stretch after resurfacing"
        assert note_indicates_cost_increase(text) is False

    def test_return_to_normal_rejected(self):
        """Notes about return to normal should be rejected."""
        text = "route returned to normal conditions after repairs"
        assert note_indicates_cost_increase(text) is False

    def test_flooding_accepted(self):
        """Notes about flooding causing higher costs should be accepted."""
        text = (
            "Heavy flooding disrupted normal truck movement, "
            "forcing longer detours and higher trip costs"
        )
        assert note_indicates_cost_increase(text) is True

    def test_surcharge_accepted(self):
        """Notes about surcharges should be accepted."""
        text = "a temporary surcharge applied due to high demand and limited truck availability"
        assert note_indicates_cost_increase(text) is True

    def test_diesel_price_rise_accepted(self):
        """Notes about diesel price increases should be accepted."""
        text = "Diesel prices rose nationwide, pushing up transportation costs"
        assert note_indicates_cost_increase(text) is True

    def test_not_in_dataset_rejected(self):
        """Notes about routes not in this dataset should be rejected."""
        text = "toll plaza commissioned but the affected routes are not part of this dataset"
        assert note_indicates_cost_increase(text) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
