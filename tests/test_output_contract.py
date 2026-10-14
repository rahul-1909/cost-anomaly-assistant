"""Contract and evidence checks for a freshly generated output.csv."""

from pathlib import Path

import pandas as pd

from src.retrieval import (
    note_applies_to_route,
    note_indicates_cost_increase,
    note_text_applies_to_week,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_output_columns_and_inferred_types_match_sample_contract():
    sample = pd.read_csv(PROJECT_ROOT / "sample_output_format_v2.csv")
    output = pd.read_csv(PROJECT_ROOT / "output.csv")

    assert output.columns.tolist() == sample.columns.tolist()
    assert output.dtypes.astype(str).tolist() == sample.dtypes.astype(str).tolist()


def test_justified_rows_cite_applicable_cost_increase_notes():
    output = pd.read_csv(PROJECT_ROOT / "output.csv").fillna("")
    notes = pd.read_csv(PROJECT_ROOT / "context_notes.csv").fillna("")
    notes_by_id = notes.set_index("note_id")

    for _, row in output.iterrows():
        if row["flagged"] == "No (justified)":
            note_id = row["matched_note_id"]
            assert note_id, f"Missing note ID for {row['route']} {row['week_of']}"
            assert note_id in notes_by_id.index, f"Unknown note ID: {note_id}"
            note = notes_by_id.loc[note_id]
            assert note_applies_to_route(note["applies_to"], row["route"])
            assert note_text_applies_to_week(
                note["note"], pd.Timestamp(note["date"]), pd.Timestamp(row["week_of"])
            )
            assert note_indicates_cost_increase(note["note"])
        elif row["flagged"] == "Yes":
            assert row["matched_note_id"] == "", (
                f"Unexplained row has matched note ID for {row['route']} {row['week_of']}"
            )
