"""
loading.py -- Data loading and preprocessing.

Reads shipment_records.csv and context_notes.csv, validates data quality,
and assigns each shipment to its Monday-Sunday week.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_shipments(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate shipment_records.csv.

    Returns a DataFrame with shipment_date parsed as datetime and
    week_of (Monday of the shipment's week) added.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = [
        "shipment_id", "origin", "destination", "route_type",
        "material", "quantity_tonnes", "distance_km",
        "freight_cost_inr", "shipment_date", "transporter",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in shipment data: {missing}")

    # Parse dates
    df["shipment_date"] = pd.to_datetime(df["shipment_date"], format="%Y-%m-%d")

    # Validate no nulls
    null_counts = df[required].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    # Validate no duplicates
    dup_count = df["shipment_id"].duplicated().sum()
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate shipment_ids")

    # Validate positive numeric values
    for col in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        if (df[col] <= 0).any():
            raise ValueError(f"Found zero or negative values in {col}")

    # Add route column: "Origin-Destination"
    df["route"] = df["origin"] + "-" + df["destination"]

    route_type_counts = df.groupby("route")["route_type"].nunique()
    inconsistent_routes = route_type_counts[route_type_counts != 1]
    assert inconsistent_routes.empty, (
        "Each route must have exactly one route_type; inconsistent routes: "
        f"{inconsistent_routes.to_dict()}"
    )

    # Assign week_of: the Monday of the shipment's Monday-Sunday week
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(
        df["shipment_date"].dt.weekday, unit="D"
    )

    return df


def load_context_notes(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate context_notes.csv.

    Returns a DataFrame with the date column parsed as datetime.
    """
    df = pd.read_csv(csv_path)

    required = ["note_id", "date", "applies_to", "note"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in context notes: {missing}")

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def load_all(data_dir: Path, config_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load config, shipments, and context notes from the given directory.

    Returns (shipments_df, notes_df, config_dict).
    """
    config = load_config(config_path)
    shipments = load_shipments(data_dir / "shipment_records.csv")
    notes = load_context_notes(data_dir / "context_notes.csv")
    return shipments, notes, config
