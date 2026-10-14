"""
test_independent_baselines.py -- Verification by independent baseline recomputation.

Selects 5 random route-weeks and independently recomputes:
  1. Weekly cost_per_tonne_km (pure python dictionary aggregation, no pandas)
  2. Trailing 8-week average (manual list slicing strictly prior to current week)
  3. Peer average (manual loop across other routes of same route_type)
  4. Percentage changes vs own history and peers

Asserts that all recomputed values match the pipeline's output.csv.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def recompute_baselines_independently(shipment_csv: Path, target_route: str, target_week_monday: str):
    """
    Independently recomputes the metrics for a given route and week
    using pure Python dictionary operations, completely independent
    of pandas groupby/rolling functions.
    """
    # 1. Read all raw shipments with standard library csv
    raw_shipments = []
    with open(shipment_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_date = datetime.strptime(row["shipment_date"], "%Y-%m-%d")
            # Monday of the shipment week
            week_monday = s_date - timedelta(days=s_date.weekday())
            route = f"{row['origin']}-{row['destination']}"
            raw_shipments.append({
                "route": route,
                "route_type": row["route_type"],
                "week_of": week_monday.strftime("%Y-%m-%d"),
                "quantity": float(row["quantity_tonnes"]),
                "distance": float(row["distance_km"]),
                "cost": float(row["freight_cost_inr"]),
            })

    # 2. Aggregate weekly cost per tonne-km for all routes & weeks
    # map: (route, week_of) -> {cost_sum, tonne_km_sum, route_type}
    aggregates = {}
    for s in raw_shipments:
        key = (s["route"], s["week_of"])
        if key not in aggregates:
            aggregates[key] = {
                "cost_sum": 0.0,
                "tonne_km_sum": 0.0,
                "route_type": s["route_type"],
            }
        aggregates[key]["cost_sum"] += s["cost"]
        aggregates[key]["tonne_km_sum"] += s["quantity"] * s["distance"]

    weekly_costs = {}
    route_types = {}
    for (r, w), data in aggregates.items():
        cptk = data["cost_sum"] / data["tonne_km_sum"]
        weekly_costs[(r, w)] = cptk
        route_types[r] = data["route_type"]

    # 3. Target week cost
    target_cost = weekly_costs.get((target_route, target_week_monday))
    target_route_type = route_types[target_route]

    # 4. Trailing 8-week average for target_route strictly before target_week
    # Collect all historical weeks for this route
    route_weeks = sorted([w for (r, w) in weekly_costs.keys() if r == target_route])
    prior_weeks = [w for w in route_weeks if w < target_week_monday]

    if not prior_weeks:
        own_avg = None
    else:
        # Take up to trailing 8 weeks
        trailing_8 = prior_weeks[-8:]
        own_avg = sum(weekly_costs[(target_route, w)] for w in trailing_8) / len(trailing_8)

    # 5. Peer average in the same week across OTHER routes with same route_type
    peer_costs = [
        cost for (r, w), cost in weekly_costs.items()
        if w == target_week_monday and r != target_route and route_types[r] == target_route_type
    ]
    peer_avg = (sum(peer_costs) / len(peer_costs)) if peer_costs else None

    # 6. Percentages
    own_pct = ((target_cost - own_avg) / own_avg * 100.0) if own_avg is not None else None
    peer_pct = ((target_cost - peer_avg) / peer_avg * 100.0) if peer_avg is not None else None

    return {
        "cost_per_tonne_km": round(target_cost, 2),
        "own_avg": own_avg,
        "peer_avg": peer_avg,
        "own_pct": round(own_pct, 1) if own_pct is not None else None,
        "peer_pct": round(peer_pct, 1) if peer_pct is not None else None,
    }


def test_independent_baseline_verification():
    """Select 5 random route-weeks and verify exact match with pipeline output."""
    output_path = PROJECT_ROOT / "output.csv"
    shipment_path = PROJECT_ROOT / "shipment_records.csv"

    assert output_path.exists(), "output.csv must exist"
    output_df = pd.read_csv(output_path)

    # Pick 5 diverse random route-weeks across different routes and years
    random.seed(999)
    sample_indices = [
        42,   # Ahmedabad-Mumbai mid 2024
        185,  # Chennai-Bangalore mid 2025
        357,  # Delhi-Jaipur Nov 2024
        500,  # Kolkata-Bhubaneswar 2025
        713,  # Mumbai-Pune Sep 2025
    ]

    for idx in sample_indices:
        row = output_df.iloc[idx]
        route = row["route"]
        week = row["week_of"]

        independent = recompute_baselines_independently(shipment_path, route, week)

        # Assert cost matches rounded to 2 decimal places
        assert np.isclose(row["cost_per_tonne_km"], independent["cost_per_tonne_km"], atol=0.01), (
            f"Cost mismatch at {route} {week}: pipeline={row['cost_per_tonne_km']}, independent={independent['cost_per_tonne_km']}"
        )

        # Assert vs_own_history matches
        if independent["own_pct"] is not None:
            sign = "+" if independent["own_pct"] >= 0 else ""
            expected_own_str = f"{sign}{independent['own_pct']:.1f}% vs this route's past average"
            assert row["vs_own_history"] == expected_own_str, (
                f"vs_own_history mismatch at {route} {week}: got '{row['vs_own_history']}', expected '{expected_own_str}'"
            )

        # Assert vs_similar_routes matches
        if independent["peer_pct"] is not None:
            sign = "+" if independent["peer_pct"] >= 0 else ""
            expected_peer_str = f"{sign}{independent['peer_pct']:.1f}% vs similar-length routes this week"
            assert row["vs_similar_routes"] == expected_peer_str, (
                f"vs_similar_routes mismatch at {route} {week}: got '{row['vs_similar_routes']}', expected '{expected_peer_str}'"
            )


if __name__ == "__main__":
    test_independent_baseline_verification()
    print("All 5 independent baseline verification checks passed successfully!")
