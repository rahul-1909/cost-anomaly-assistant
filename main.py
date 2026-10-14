"""
main.py -- Main pipeline runner for the FreightTiger Smart Assistant.

Orchestrates: loading -> metrics -> flagging -> retrieval -> explanation -> output.

Usage:
    python main.py                     # Run full pipeline
    python main.py --config config.yaml # Use custom config
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loading import load_all
from src.metrics import build_metrics_table
from src.flagging import apply_flags
from src.retrieval import build_index, match_note_for_route_week
from src.explain import (
    generate_template_reason_justified,
    generate_template_reason_unexplained,
)


def run_pipeline(
    data_dir: Path,
    config_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Run the complete pipeline and write output.csv.

    Returns the final output DataFrame.
    """
    print(f"[{datetime.now():%H:%M:%S}] Loading data...")
    shipments, notes, config = load_all(data_dir, config_path)
    print(f"  Loaded {len(shipments)} shipments, {len(notes)} notes")

    print(f"[{datetime.now():%H:%M:%S}] Computing weekly metrics...")
    weekly = build_metrics_table(shipments, config)
    print(f"  Computed metrics for {len(weekly)} route-weeks")

    print(f"[{datetime.now():%H:%M:%S}] Applying flag rules...")
    weekly = apply_flags(weekly, config)
    flagged_count = weekly["flagged_bool"].sum()
    print(f"  Flagged {flagged_count} candidate anomalous route-weeks")

    print(f"[{datetime.now():%H:%M:%S}] Building FAISS index...")
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    faiss_index, embeddings, model = build_index(
        notes,
        model_name=embedding_model,
        model_revision=config.get("embedding_model_revision"),
    )
    print(f"  Index built with {len(notes)} notes, {embeddings.shape[1]}-dim embeddings")

    print(f"[{datetime.now():%H:%M:%S}] Matching notes and generating reasons...")
    matched_note_ids = []
    reasons = []
    flagged_strings = []

    for _, row in weekly.iterrows():
        if row["flagged_bool"]:
            # Try to find a matching note using deterministic code rules
            note_id, note_text, verdict_type, rejected_info = match_note_for_route_week(
                route=row["route"],
                week_monday=row["week_of"],
                vs_own_history_pct=row.get("vs_own_history_pct"),
                index=faiss_index,
                model=model,
                notes=notes,
                config=config,
            )

            if verdict_type == "justified":
                flagged_str = "No (justified)"
                note_date_str = notes.loc[
                    notes["note_id"] == note_id, "date"
                ].iloc[0].strftime("%Y-%m-%d")
            else:
                flagged_str = "Yes"
                note_id = None
                note_text = None
                note_date_str = None

            if verdict_type == "justified":
                reason = generate_template_reason_justified(
                    note_id, note_text, note_date_str or ""
                )
            else:
                reason = generate_template_reason_unexplained(rejected_info)

            matched_note_ids.append(note_id or "")
            reasons.append(reason)
            flagged_strings.append(flagged_str)
        else:
            matched_note_ids.append("")
            reasons.append("")
            flagged_strings.append("No")

    weekly["flagged"] = flagged_strings
    weekly["matched_note_id"] = matched_note_ids
    weekly["reason"] = reasons

    # Format week_of as date string YYYY-MM-DD
    weekly["week_of"] = weekly["week_of"].dt.strftime("%Y-%m-%d")

    # Select and order output columns to match sample_output_format_v2.csv EXACTLY
    output_cols = [
        "route", "week_of", "cost_per_tonne_km",
        "vs_own_history", "vs_similar_routes",
        "flagged", "matched_note_id", "reason",
    ]
    output = weekly[output_cols].copy()

    # Write output CSV
    output.to_csv(output_path, index=False)
    print(f"[{datetime.now():%H:%M:%S}] Output written to {output_path}")
    print(f"  Total rows: {len(output)}")
    print(f"  Flagged (Yes): {(output['flagged'] == 'Yes').sum()}")
    print(f"  Justified: {(output['flagged'] == 'No (justified)').sum()}")
    print(f"  Unflagged (No): {(output['flagged'] == 'No').sum()}")

    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FreightTiger Smart Assistant - Shipping Cost Monitor"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root)"
    )
    parser.add_argument(
        "--output", type=str, default="output.csv",
        help="Output CSV path (default: output.csv in project root)"
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT
    config_path = PROJECT_ROOT / args.config
    output_path = PROJECT_ROOT / args.output

    run_pipeline(
        data_dir=data_dir,
        config_path=config_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
