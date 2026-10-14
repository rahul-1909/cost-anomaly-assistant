"""
metrics.py -- Cost metric computation.

Computes weekly cost_per_tonne_km for each route, the trailing 8-week
rolling average (vs_own_history), and the peer average (vs_similar_routes).
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_weekly_cost(shipments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shipments to weekly cost_per_tonne_km per route.

    Formula: cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)

    This aggregates at the route-week level first (summing numerator and denominator
    across all shipments), then divides -- never averaging per-shipment ratios.

    Returns a DataFrame with columns:
        route, route_type, week_of, cost_per_tonne_km, shipment_count,
        total_cost, total_tonne_km
    """
    # Compute per-shipment tonne_km for aggregation
    df = shipments.copy()
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    # Group by route and week
    weekly = (
        df.groupby(["route", "route_type", "week_of"])
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
        .reset_index()
    )

    # Compute the aggregate cost metric
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]

    # Sort for deterministic ordering
    weekly = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

    return weekly


def compute_vs_own_history(
    weekly: pd.DataFrame, window: int = 8
) -> pd.DataFrame:
    """
    Compute the trailing rolling average cost_per_tonne_km for each route.

    For each route-week, vs_own_history is the average cost_per_tonne_km
    over the preceding `window` weeks (strictly before the current week).
    If fewer than `window` prior weeks exist, use all available prior weeks.
    If zero prior weeks exist, leave it NaN.

    Returns the input DataFrame with an added 'own_history_avg' column.
    """
    result = weekly.copy()
    result["own_history_avg"] = np.nan

    for route in result["route"].unique():
        mask = result["route"] == route
        route_data = result.loc[mask].sort_values("week_of")

        for idx in route_data.index:
            # Use the trailing calendar window, not merely the last N observed
            # rows. Missing route-weeks must not pull older weeks into baseline.
            current_week = route_data.loc[idx, "week_of"]
            window_start = current_week - pd.Timedelta(weeks=window)
            prior_rows = route_data[
                (route_data["week_of"] >= window_start)
                & (route_data["week_of"] < current_week)
            ]
            if not prior_rows.empty:
                result.loc[idx, "own_history_avg"] = prior_rows[
                    "cost_per_tonne_km"
                ].mean()

    return result


def compute_vs_similar_routes(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the peer average cost_per_tonne_km for each route-week.

    For each route in a given week, the peer average is the mean cost_per_tonne_km
    across all OTHER routes with the same route_type in that same week.

    Returns the input DataFrame with an added 'peer_avg' column.
    """
    result = weekly.copy()
    result["peer_avg"] = np.nan

    for _, row in result.iterrows():
        same_type_same_week = result[
            (result["route_type"] == row["route_type"])
            & (result["week_of"] == row["week_of"])
            & (result["route"] != row["route"])
        ]
        if len(same_type_same_week) > 0:
            result.loc[row.name, "peer_avg"] = same_type_same_week[
                "cost_per_tonne_km"
            ].mean()

    return result


def compute_percentage_change(
    current: float, baseline: Optional[float]
) -> Optional[float]:
    """
    Compute percentage change from baseline to current.

    Returns None if baseline is None/NaN or zero.
    Returns the percentage as a float (e.g., 35.5 for +35.5%).
    """
    if baseline is None or (isinstance(baseline, float) and np.isnan(baseline)):
        return None
    if baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0


def format_vs_own_history(pct: Optional[float]) -> str:
    """Format the vs_own_history percentage as required by the output spec."""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return ""
    display_pct = round(pct, 1)
    if display_pct == 0:
        return "0.0% vs this route's past average"
    sign = "+" if display_pct > 0 else ""
    return f"{sign}{display_pct:.1f}% vs this route's past average"


def format_vs_similar_routes(pct: Optional[float]) -> str:
    """Format the vs_similar_routes percentage as required by the output spec."""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return ""
    display_pct = round(pct, 1)
    if display_pct == 0:
        return "0.0% vs similar-length routes this week"
    sign = "+" if display_pct > 0 else ""
    return f"{sign}{display_pct:.1f}% vs similar-length routes this week"


def build_metrics_table(shipments: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full metrics pipeline: weekly cost -> own history -> peer average -> percentages.

    Returns a DataFrame with all metric columns ready for flagging.
    """
    window = config.get("own_history_window", 8)

    weekly = compute_weekly_cost(shipments)
    weekly = compute_vs_own_history(weekly, window=window)
    weekly = compute_vs_similar_routes(weekly)

    # Compute percentage changes
    weekly["vs_own_history_pct"] = weekly.apply(
        lambda r: compute_percentage_change(r["cost_per_tonne_km"], r["own_history_avg"]),
        axis=1,
    )
    weekly["vs_peer_pct"] = weekly.apply(
        lambda r: compute_percentage_change(r["cost_per_tonne_km"], r["peer_avg"]),
        axis=1,
    )

    # Format display strings
    weekly["vs_own_history"] = weekly["vs_own_history_pct"].apply(format_vs_own_history)
    weekly["vs_similar_routes"] = weekly["vs_peer_pct"].apply(format_vs_similar_routes)

    # Round cost
    precision = config.get("cost_precision", 2)
    weekly["cost_per_tonne_km"] = weekly["cost_per_tonne_km"].round(precision)

    return weekly
