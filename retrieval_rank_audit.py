"""Report each justified output row's gold-note rank in the FAISS results."""

from pathlib import Path

import pandas as pd

from src.loading import load_config, load_context_notes
from src.retrieval import (
    LexicalIndex,
    build_index,
    build_route_week_query,
    retrieve_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    output = pd.read_csv(PROJECT_ROOT / "output.csv").fillna("")
    notes = load_context_notes(PROJECT_ROOT / "context_notes.csv")
    config = load_config(PROJECT_ROOT / "config.yaml")
    index, embeddings, model = build_index(
        notes,
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
        model_revision=config.get("embedding_model_revision"),
    )
    if isinstance(index, LexicalIndex):
        raise RuntimeError(
            "FAISS rank audit requires the embedding model and FAISS; "
            "the pipeline selected its lexical fallback."
        )

    print(f"FAISS gold-note ranks across {len(notes)} notes:")
    justified = output[output["flagged"] == "No (justified)"]
    for _, row in justified.iterrows():
        own_text = str(row["vs_own_history"])
        own_pct = float(own_text.split("%", 1)[0].replace("+", ""))
        query = build_route_week_query(
            row["route"], pd.Timestamp(row["week_of"]), own_pct
        )
        results = retrieve_candidates(
            query, index, model, notes, top_k=len(notes)
        )
        gold_indices = notes.index[notes["note_id"] == row["matched_note_id"]]
        if len(gold_indices) != 1:
            raise AssertionError(f"Expected one row for gold note {row['matched_note_id']}")
        rank = next(
            (position for position, (note_idx, _score) in enumerate(results, 1)
             if note_idx == gold_indices[0]),
            None,
        )
        print(
            f"{row['route']}, {row['week_of']}: "
            f"{row['matched_note_id']} rank={rank}"
        )

    if embeddings.shape[1] == 0:
        raise AssertionError("FAISS index unexpectedly has no embedding dimensions")


if __name__ == "__main__":
    main()
