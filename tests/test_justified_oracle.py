"""Oracle checks for the only justified route-week-note combinations."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_JUSTIFIED = {
    ("Ahmedabad-Mumbai", "2025-01-20", "N002"),
    ("Chennai-Bangalore", "2025-02-24", "N001"),
    ("Chennai-Bangalore", "2025-03-03", "N001"),
    ("Delhi-Chennai", "2025-05-05", "N003"),
    ("Delhi-Jaipur", "2025-05-05", "N003"),
    ("Kolkata-Bhubaneswar", "2025-05-05", "N003"),
    ("Mumbai-Delhi", "2025-05-05", "N003"),
    ("Mumbai-Pune", "2025-05-05", "N003"),
}


def test_justified_rows_match_the_hardcoded_note_oracle():
    output = pd.read_csv(PROJECT_ROOT / "output.csv").fillna("")
    notes = pd.read_csv(PROJECT_ROOT / "context_notes.csv").fillna("")
    note_ids = set(notes["note_id"])

    justified_rows = output[output["flagged"] == "No (justified)"]
    actual_justified = {
        (row.route, row.week_of, row.matched_note_id)
        for row in justified_rows.itertuples(index=False)
    }
    assert actual_justified <= VALID_JUSTIFIED

    for route, week, note_id in VALID_JUSTIFIED:
        rows = output[(output["route"] == route) & (output["week_of"] == week)]
        assert len(rows) == 1, f"Expected one output row for {route} {week}"
        row = rows.iloc[0]
        assert row["flagged"] == "No (justified)", (
            f"Oracle candidate {route} {week} was not justified"
        )
        assert row["matched_note_id"] == note_id

    for _, row in justified_rows.iterrows():
        assert row["matched_note_id"] in note_ids

    unexplained = output[output["flagged"] == "Yes"]
    assert (unexplained["matched_note_id"] == "").all()
