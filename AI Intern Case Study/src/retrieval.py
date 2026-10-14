"""
retrieval.py -- RAG retrieval layer using sentence-transformers and FAISS.

Embeds context_notes.csv and retrieves candidate notes for flagged route-weeks.
All note matching logic is DETERMINISTIC CODE, not LLM-driven.
"""

import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Patterns that indicate a note says costs were NOT affected or decreased
NO_IMPACT_PATTERNS = [
    r"costs?\s+(?:were|was)\s+not\s+(?:significantly\s+)?affected",
    r"not\s+part\s+of\s+this\s+dataset",
    r"no\s+(?:significant\s+)?disruptions?\s+(?:were\s+)?reported",
    r"remained?\s+(?:stable|normal)",
    r"without\s+a\s+rate\s+change",
    r"absorbed\s+by\s+transporters",
    r"improved\s+road\s+conditions",
    r"returned?\s+to\s+normal",
    r"no\s+major\s+disruptions?\s+reported",
]


def build_index(
    notes: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    model_revision: Optional[str] = None,
) -> Tuple[faiss.IndexFlatIP, np.ndarray, SentenceTransformer]:
    """
    Build a FAISS index from context notes embeddings.

    Each note is embedded using the concatenation of its metadata and text.
    Uses inner product (cosine similarity after normalization).

    Returns (faiss_index, embeddings, model).
    """
    model = SentenceTransformer(model_name, revision=model_revision)

    # Build embedding texts: include metadata for better retrieval
    texts = []
    for _, row in notes.iterrows():
        text = (
            f"Note {row['note_id']} dated {row['date'].strftime('%Y-%m-%d')} "
            f"about {row['applies_to']}: {row['note']}"
        )
        texts.append(text)

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    # Build FAISS index with inner product (cosine similarity for normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings, model


def retrieve_candidates(
    query: str,
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    notes: pd.DataFrame,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k candidate note indices and scores for a query.

    Returns list of (note_index, similarity_score) tuples.
    """
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, min(top_k, len(notes)))
    return list(zip(indices[0].tolist(), scores[0].tolist()))


def note_applies_to_route(applies_to: str, route: str) -> bool:
    """
    Check if a note's applies_to field matches the given route.

    A note applies if:
    - applies_to is "All Routes" (case-insensitive), OR
    - applies_to exactly matches the route string (e.g., "Chennai-Bangalore"), OR
    - Both the origin and destination appear in applies_to
    """
    applies_lower = applies_to.strip().lower()
    route_lower = route.strip().lower()

    if applies_lower == "all routes":
        return True

    if applies_lower == route_lower:
        return True

    # Check if origin and destination are both mentioned
    parts = route.split("-")
    if len(parts) == 2:
        origin, dest = parts[0].strip().lower(), parts[1].strip().lower()
        if origin in applies_lower and dest in applies_lower:
            return True

    return False


def note_date_applies_to_week(
    note_date: pd.Timestamp,
    week_monday: pd.Timestamp,
    lookback_days: int = 30,
    effect_window_days: int = 14,
) -> bool:
    """
    Check if a note's date is applicable to the flagged week.

    A note applies if its effect window overlaps with the flagged week:
    - The note's effect runs from note_date to note_date + effect_window_days
    - The flagged week runs from week_monday to week_monday + 6 days (Sunday)
    - There must be an overlap between these two intervals
    - The note date must not be after the week's Sunday (no future notes)
    - The note date must not be more than lookback_days before the week's Monday
    """
    week_sunday = week_monday + timedelta(days=6)
    note_effect_end = note_date + timedelta(days=effect_window_days)

    # Note must not be dated after the week's Sunday (no look-ahead to future events)
    if note_date > week_sunday:
        return False

    # Note must not be too old
    if note_date < week_monday - timedelta(days=lookback_days):
        return False

    # Note's effect window must overlap with the week
    # Overlap exists if note_effect_end >= week_monday and note_date <= week_sunday
    if note_effect_end < week_monday:
        return False

    return True


def note_text_applies_to_week(
    note_text: str,
    note_date: pd.Timestamp,
    week_monday: pd.Timestamp,
    lookback_days: int = 30,
    effect_window_days: int = 14,
) -> bool:
    """Apply explicit event dates in note text before the generic effect cap."""
    if not note_date_applies_to_week(
        note_date, week_monday, lookback_days, effect_window_days
    ):
        return False

    # Notes can give a more precise event interval than the configured default.
    explicit_range = re.search(
        r"\bfrom\s+([A-Za-z]{3,9}\s+\d{1,2})\s+to\s+([A-Za-z]{3,9}\s+\d{1,2})",
        note_text,
        flags=re.IGNORECASE,
    )
    if not explicit_range:
        return True
    def parse_month_day(value: str) -> pd.Timestamp:
        for month_format in ("%B %d %Y", "%b %d %Y"):
            try:
                return pd.to_datetime(
                    f"{value} {note_date.year}", format=month_format, errors="raise"
                )
            except ValueError:
                pass
        raise ValueError(f"Unrecognized explicit date in note: {value}")

    start = parse_month_day(explicit_range.group(1))
    end = parse_month_day(explicit_range.group(2))
    if end < start:
        end = end.replace(year=note_date.year + 1)
    week_sunday = week_monday + timedelta(days=6)
    return start <= week_sunday and end >= week_monday


def note_indicates_cost_increase(note_text: str) -> bool:
    """
    Check if a note indicates costs increased.

    Explicitly rejects notes stating costs were NOT affected or decreased.
    Only accepts notes that describe events causing higher costs.
    """
    text_lower = note_text.lower()

    # Check for explicit no-impact patterns
    for pattern in NO_IMPACT_PATTERNS:
        if re.search(pattern, text_lower):
            return False

    # Check for positive cost impact indicators
    cost_increase_indicators = [
        r"higher\s+(?:trip\s+)?costs?",
        r"surcharge",
        r"pushing\s+up\s+(?:transportation\s+)?costs?",
        r"costs?\s+(?:rose|increased|went\s+up|jumped)",
        r"price[sd]?\s+(?:rose|increased|went\s+up)",
        r"disrupted?\s+(?:normal)?\s*(?:truck)?\s*movement",
        r"forcing\s+(?:longer\s+)?detours",
        r"limited\s+(?:truck\s+)?availability",
        r"high\s+demand",
    ]

    for pattern in cost_increase_indicators:
        if re.search(pattern, text_lower):
            return True

    return False


def match_note_for_route_week(
    route: str,
    week_monday: pd.Timestamp,
    vs_own_history_pct: Optional[float],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    notes: pd.DataFrame,
    config: dict,
) -> Tuple[Optional[str], Optional[str], str, Optional[Dict[str, Any]]]:
    """
    Find a matching note for a flagged route-week.

    Uses vector retrieval to get candidates, then applies deterministic
    filtering rules:
    1. Note's applies_to must match the route
    2. Note's date must be applicable to the week (no future notes)
    3. Note must indicate costs increased (reject negative/neutral notes)

    If multiple notes qualify, picks by deterministic tie-break:
    - Prefer route-specific notes over "All Routes"
    - Then prefer most recent note
    - Then prefer lowest note_id (alphabetical)

    If no note qualifies, also checks if a candidate note was retrieved that
    matched route & date proximity but was rejected by Rule (c). If so, returns
    that rejected note's details to ground the explanation (e.g. sample format row 3).

    Returns (matched_note_id, note_text, verdict_type, rejected_candidate_info).
    verdict_type is "justified" or "unexplained".
    """
    top_k = config.get("rag_top_k", 5)
    lookback = config.get("note_lookback_days", 30)
    effect_window = config.get("note_effect_window_days", 14)

    # Build query from route, week, and cost movement
    pct_str = f"{vs_own_history_pct:+.1f}%" if vs_own_history_pct is not None else "unknown"
    origin, dest = route.split("-", 1)
    query = (
        f"Cost increase on route {route} from {origin} to {dest} "
        f"during week of {week_monday.strftime('%Y-%m-%d')}. "
        f"Cost changed by {pct_str} compared to historical average."
    )

    # Retrieve candidates
    candidates = retrieve_candidates(query, index, model, notes, top_k=top_k)

    qualifying = []
    rejected_applicable = []

    for note_idx, score in candidates:
        if note_idx < 0 or note_idx >= len(notes):
            continue
        note_row = notes.iloc[note_idx]

        # Rule (a): applies_to must match the route
        if not note_applies_to_route(note_row["applies_to"], route):
            continue

        # Rule (b): date must be applicable to the week (or within proximity window)
        # Note: for proximity reporting of non-disruptions (like N006 on 2025-09-22 for 2025-09-15 week),
        # allow within +/- 7 days
        week_sunday = week_monday + timedelta(days=6)
        is_date_applicable = note_text_applies_to_week(
            note_row["note"], note_row["date"], week_monday, lookback, effect_window
        )
        is_near_date = abs((note_row["date"] - week_monday).days) <= 14

        # Rule (c): note must indicate costs increased
        indicates_increase = note_indicates_cost_increase(note_row["note"])

        if is_date_applicable and indicates_increase:
            qualifying.append((note_row, score))
        elif is_near_date and not indicates_increase:
            rejected_applicable.append((note_row, score))

    if qualifying:
        # Deterministic tie-break
        def tie_break_key(item: Tuple) -> Tuple:
            note_row, score = item
            # Prefer route-specific over "All Routes"
            is_specific = 0 if note_row["applies_to"].lower() != "all routes" else 1
            # Prefer most recent note
            recency = -note_row["date"].timestamp()
            # Then by note_id alphabetically
            note_id = note_row["note_id"]
            return (is_specific, recency, note_id)

        qualifying.sort(key=tie_break_key)
        best_note = qualifying[0][0]
        return best_note["note_id"], best_note["note"], "justified", None

    # No qualifying note passed
    rejected_info = None
    if rejected_applicable:
        best_rejected = rejected_applicable[0][0]
        rejected_info = {
            "note_id": best_rejected["note_id"],
            "date": best_rejected["date"].strftime("%Y-%m-%d"),
            "note": best_rejected["note"],
            "applies_to": best_rejected["applies_to"],
        }

    return None, None, "unexplained", rejected_info
