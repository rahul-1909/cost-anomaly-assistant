"""
retrieval.py -- RAG retrieval layer using sentence-transformers and FAISS.

Embeds context_notes.csv and retrieves candidate notes for flagged route-weeks.
All note matching logic is DETERMINISTIC CODE, not LLM-driven.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


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


class LexicalIndex:
    """Small deterministic fallback index used when vector search is unavailable."""

    def __init__(self, note_ids: List[str], documents: List[str]):
        self.note_ids = note_ids
        self.document_tokens = [
            set(re.findall(r"[a-z0-9]+", document.lower())) for document in documents
        ]


def build_index(
    notes: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    model_revision: Optional[str] = None,
) -> Tuple[Any, np.ndarray, Any]:
    """Build a vector index, or a stable lexical index if vector setup fails."""
    # Include metadata in both indexes so retrieval sees route and date context.
    texts = []
    for _, row in notes.iterrows():
        text = (
            f"Note {row['note_id']} dated {row['date'].strftime('%Y-%m-%d')} "
            f"about {row['applies_to']}: {row['note']}"
        )
        texts.append(text)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, revision=model_revision)
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        import faiss

        # Inner product is cosine similarity for normalized vectors.
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings, model
    except Exception as exc:
        logging.warning(
            "Could not load the embedding model or FAISS index (%s); "
            "using deterministic lexical retrieval.",
            exc,
        )
        index = LexicalIndex(notes["note_id"].astype(str).tolist(), texts)
        # A zero-width array signals the fallback to the pipeline's status log.
        return index, np.empty((len(notes), 0), dtype=np.float32), None


def retrieve_candidates(
    query: str,
    index: Any,
    model: Any,
    notes: pd.DataFrame,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k candidate note indices and scores for a query.

    Returns list of (note_index, similarity_score) tuples.
    """
    if isinstance(index, LexicalIndex):
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        query_size = max(len(query_tokens), 1)
        ranked = []
        for note_idx, document_tokens in enumerate(index.document_tokens):
            overlap = len(query_tokens & document_tokens)
            denom = (query_size * max(len(document_tokens), 1)) ** 0.5
            ranked.append((note_idx, overlap / denom))
        ranked.sort(key=lambda item: (-item[1], index.note_ids[item[0]]))
        return ranked[: min(top_k, len(notes))]

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
) -> bool:
    """Use the note's own calendar week when its text states no duration."""
    note_date = pd.Timestamp(note_date).normalize()
    week_monday = pd.Timestamp(week_monday).normalize()
    note_monday = note_date - pd.Timedelta(days=note_date.weekday())
    return note_monday == week_monday


_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _parse_note_date(value: str, default_year: int) -> pd.Timestamp:
    """Parse a month/day date, using the note year when the text omits it."""
    match = re.fullmatch(
        r"([A-Za-z]{3,9})\s+(\d{1,2})(?:,?\s+(\d{4}))?", value.strip()
    )
    if not match:
        raise ValueError(f"Unrecognized date in note: {value}")
    month_text, day_text, year_text = match.groups()
    month = pd.to_datetime(month_text, format="%B", errors="coerce")
    if pd.isna(month):
        month = pd.to_datetime(month_text, format="%b", errors="raise")
    year = int(year_text) if year_text else default_year
    return pd.Timestamp(year=year, month=month.month, day=int(day_text))


def _note_effect_interval(
    note_text: str,
    note_date: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Derive an inclusive event interval from dates/durations stated in a note."""
    note_date = pd.Timestamp(note_date).normalize()

    # Explicit date ranges take precedence over inferred duration phrases.
    date_range = re.search(
        r"\bfrom\s+([A-Za-z]{3,9}\s+\d{1,2}(?:,?\s+\d{4})?)"
        r"\s+to\s+([A-Za-z]{3,9}\s+\d{1,2}(?:,?\s+\d{4})?)",
        note_text,
        flags=re.IGNORECASE,
    )
    if date_range:
        try:
            start = _parse_note_date(date_range.group(1), note_date.year)
            end = _parse_note_date(date_range.group(2), note_date.year)
        except ValueError:
            pass
        else:
            if end < start and not re.search(r"\d{4}", date_range.group(2)):
                end = end.replace(year=end.year + 1)
            return start, end

    duration = re.search(
        r"\b(?:for\s+)?(?:about\s+|approximately\s+)?"
        r"(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(days?|weeks?|months?)\b",
        note_text,
        flags=re.IGNORECASE,
    )
    if duration:
        amount_text, unit = duration.groups()
        amount = _NUMBER_WORDS.get(amount_text.lower(), int(amount_text) if amount_text.isdigit() else 0)
        if amount > 0:
            if unit.lower().startswith("day"):
                end = note_date + pd.Timedelta(days=amount - 1)
            elif unit.lower().startswith("week"):
                end = note_date + pd.Timedelta(days=(amount * 7) - 1)
            else:
                end = note_date + pd.DateOffset(months=amount) - pd.Timedelta(days=1)
            return note_date, pd.Timestamp(end).normalize()

    if re.search(r"\b(?:this|starting this)\s+week\b|\bfestival week\b", note_text, re.IGNORECASE):
        start = note_date - pd.Timedelta(days=note_date.weekday())
        return start, start + pd.Timedelta(days=6)

    if re.search(r"\bthis quarter\b", note_text, re.IGNORECASE):
        quarter_start_month = ((note_date.month - 1) // 3) * 3 + 1
        start = pd.Timestamp(note_date.year, quarter_start_month, 1)
        end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        return start, pd.Timestamp(end).normalize()

    # A note without an explicit duration is limited to the calendar week of
    # its date; the system does not invent a generic multi-week carryover.
    start = note_date - pd.Timedelta(days=note_date.weekday())
    return start, start + pd.Timedelta(days=6)


def note_text_applies_to_week(
    note_text: str,
    note_date: pd.Timestamp,
    week_monday: pd.Timestamp,
) -> bool:
    """Check overlap with the duration/date interval stated in the note."""
    note_date = pd.Timestamp(note_date).normalize()
    week_monday = pd.Timestamp(week_monday).normalize()
    week_sunday = week_monday + timedelta(days=6)
    if note_date > week_sunday:
        return False
    event_start, event_end = _note_effect_interval(note_text, note_date)
    return event_start <= week_sunday and event_end >= week_monday


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
    index: Any,
    model: Any,
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

    If no note qualifies, checks all notes for applicable context that fails the
    cost-increase rule. That rejected context is explanatory only and cannot
    change the verdict.

    Returns (matched_note_id, note_text, verdict_type, rejected_candidate_info).
    verdict_type is "justified" or "unexplained".
    """
    top_k = config.get("rag_top_k", 5)
    query = build_route_week_query(route, week_monday, vs_own_history_pct)

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

        # Rule (b): use event dates/durations stated in note text; future notes
        # and notes outside their event interval cannot support a verdict.
        is_date_applicable = note_text_applies_to_week(
            note_row["note"], note_row["date"], week_monday
        )

        # Rule (c): note must indicate costs increased
        indicates_increase = note_indicates_cost_increase(note_row["note"])

        if is_date_applicable and indicates_increase:
            qualifying.append((note_row, score))

    # Rejected context is selected independently of retrieval backend/ranking,
    # so a vector-model outage cannot change an unexplained row's reason text.
    for _, note_row in notes.iterrows():
        if (
            note_applies_to_route(note_row["applies_to"], route)
            and note_text_applies_to_week(
                note_row["note"], note_row["date"], week_monday
            )
            and not note_indicates_cost_increase(note_row["note"])
        ):
            rejected_applicable.append((note_row, 0.0))

    def tie_break_key(item: Tuple) -> Tuple:
        note_row, _score = item
        is_specific = 0 if note_row["applies_to"].lower() != "all routes" else 1
        recency = -note_row["date"].timestamp()
        return (is_specific, recency, note_row["note_id"])

    if qualifying:
        # Deterministic tie-break
        qualifying.sort(key=tie_break_key)
        best_note = qualifying[0][0]
        return best_note["note_id"], best_note["note"], "justified", None

    # No qualifying note passed
    rejected_info = None
    if rejected_applicable:
        rejected_applicable.sort(key=tie_break_key)
        best_rejected = rejected_applicable[0][0]
        rejected_info = {
            "note_id": best_rejected["note_id"],
            "date": best_rejected["date"].strftime("%Y-%m-%d"),
            "note": best_rejected["note"],
            "applies_to": best_rejected["applies_to"],
        }

    return None, None, "unexplained", rejected_info


def build_route_week_query(
    route: str,
    week_monday: pd.Timestamp,
    vs_own_history_pct: Optional[float],
) -> str:
    """Construct the same retrieval query for matching and rank audits."""
    pct_str = (
        f"{vs_own_history_pct:+.1f}%"
        if vs_own_history_pct is not None
        else "unknown"
    )
    origin, dest = route.split("-", 1)
    return (
        f"Cost increase on route {route} from {origin} to {dest} "
        f"during week of {pd.Timestamp(week_monday).strftime('%Y-%m-%d')}. "
        f"Cost changed by {pct_str} compared to historical average."
    )
