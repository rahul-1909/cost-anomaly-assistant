"""The pipeline keeps deterministic retrieval available without model files."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from main import run_pipeline
from src.retrieval import LexicalIndex, build_index, retrieve_candidates


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_embedding_load_failure_warns_and_uses_stable_lexical_ranking(monkeypatch, caplog):
    notes = pd.DataFrame(
        [
            {
                "note_id": "N002",
                "date": pd.Timestamp("2025-01-20"),
                "applies_to": "Ahmedabad-Mumbai",
                "note": "Festival demand caused a transport surcharge.",
            },
            {
                "note_id": "N001",
                "date": pd.Timestamp("2025-02-24"),
                "applies_to": "Chennai-Bangalore",
                "note": "Flooding forced detours and higher trip costs.",
            },
        ]
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    index, embeddings, model = build_index(notes)

    assert isinstance(index, LexicalIndex)
    assert embeddings.shape == (2, 0)
    assert model is None
    assert "using deterministic lexical retrieval" in caplog.text

    ranked_once = retrieve_candidates(
        "Chennai-Bangalore flooding higher trip costs", index, model, notes, top_k=2
    )
    ranked_twice = retrieve_candidates(
        "Chennai-Bangalore flooding higher trip costs", index, model, notes, top_k=2
    )
    assert ranked_once == ranked_twice
    assert ranked_once[0][0] == 1
    assert np.isfinite([score for _, score in ranked_once]).all()


def test_pipeline_fallback_matches_vector_output(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    fallback_output = PROJECT_ROOT / "fallback_test_output.csv"
    try:
        result = run_pipeline(
            PROJECT_ROOT, PROJECT_ROOT / "config.yaml", fallback_output
        )
        expected = pd.read_csv(PROJECT_ROOT / "output.csv").fillna("")
        pd.testing.assert_frame_equal(
            result.fillna(""), expected, check_dtype=False
        )
    finally:
        fallback_output.unlink(missing_ok=True)
