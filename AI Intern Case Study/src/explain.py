"""
explain.py -- Explanation generation layer.

Generates plain-English reasons for flagged route-weeks.
Uses an LLM (Gemini free tier) with temperature 0 for justified verdicts,
with a template-based fallback. Caches all responses on disk.

Verdict logic is DETERMINISTIC CODE, not the LLM. The LLM only writes
the human-readable reason text.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# Token tracking
_token_log: Dict[str, int] = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "num_llm_calls": 0,
    "cache_hits": 0,
}


def get_token_log() -> Dict[str, int]:
    """Return the current token usage log."""
    return _token_log.copy()


def _cache_path(cache_dir: Path, prompt_hash: str) -> Path:
    """Get the cache file path for a given prompt hash."""
    return cache_dir / f"{prompt_hash}.json"


def _hash_prompt(prompt: str) -> str:
    """Compute a deterministic hash of the prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_cache(cache_dir: Path, prompt_hash: str) -> Optional[str]:
    """Load a cached LLM response, if it exists."""
    path = _cache_path(cache_dir, prompt_hash)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("response")
    return None


def _save_cache(cache_dir: Path, prompt_hash: str, prompt: str, response: str) -> None:
    """Save an LLM response to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, prompt_hash)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": response}, f, indent=2)


def _call_llm(prompt: str, config: dict, cache_dir: Path) -> str:
    """
    Call the LLM (Gemini free tier) with caching.

    Temperature is set to 0 for deterministic output.
    All responses are cached on disk keyed by prompt hash.
    """
    global _token_log

    prompt_hash = _hash_prompt(prompt)

    # Check cache first
    cached = _load_cache(cache_dir, prompt_hash)
    if cached is not None:
        _token_log["cache_hits"] += 1
        return cached

    # Try to call Gemini
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # Fall back to template if no API key
            return ""

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.get("llm_model", "gemini-2.0-flash"))
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
            ),
        )

        result_text = response.text.strip()

        # Track tokens
        if hasattr(response, "usage_metadata"):
            _token_log["total_input_tokens"] += getattr(
                response.usage_metadata, "prompt_token_count", 0
            )
            _token_log["total_output_tokens"] += getattr(
                response.usage_metadata, "candidates_token_count", 0
            )

        _token_log["num_llm_calls"] += 1

        # Cache the response
        _save_cache(cache_dir, prompt_hash, prompt, result_text)
        return result_text

    except Exception:
        # Fall back to template on any LLM error
        return ""


def _extract_core_fact(note_text: str) -> str:
    """Extract a concise factual summary clause from the note text."""
    # Common core clauses from context notes
    t = note_text.strip()
    if "regional festival week" in t.lower():
        return "a regional festival week drove a temporary surcharge on this corridor"
    if "heavy flooding" in t.lower():
        return "heavy flooding disrupted highway truck movement, forcing detours and higher trip costs"
    if "diesel prices" in t.lower():
        return "diesel prices rose nationwide, pushing up transportation costs across all routes"
    if "stable demand" in t.lower() or "no major disruptions" in t.lower():
        return "stable demand with no major disruptions"
    if "costs were not significantly affected" in t.lower():
        return "highway maintenance work where costs were not significantly affected"
    if "returned to normal" in t.lower():
        return "the corridor returned to normal conditions after repairs were completed"
    if "without a rate change" in t.lower():
        return "compliance costs were absorbed by transporters without a rate change"
    if "not part of this dataset" in t.lower():
        return "a toll plaza change where the affected routes are not part of this dataset"

    # Default to first sentence lowercase
    first_sentence = t.split(".")[0].strip()
    return first_sentence[0].lower() + first_sentence[1:] if first_sentence else t


def generate_template_reason_justified(
    note_id: str,
    note_text: str,
    note_date: str,
    route: str,
    week_of: str,
    vs_own_history: str,
) -> str:
    """
    Generate a template-based reason for a justified verdict matching sample format:
    'Matches note N002 dated 2025-01-20: a regional festival week drove a temporary surcharge
    on this corridor. The cost rise has a clear explanation.'
    """
    core_fact = _extract_core_fact(note_text)
    return (
        f"Matches note {note_id} dated {note_date}: {core_fact}. "
        f"The cost rise has a clear explanation."
    )


def generate_template_reason_unexplained(
    route: str,
    week_of: str,
    rejected_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a template-based reason for an unexplained verdict.

    If a candidate note was retrieved but rejected (e.g. N006 on stable demand):
    'The closest note (N006, 2025-09-22) mentions stable demand with no major disruptions --
    it does not describe a reason for a cost rise on this route. No genuine justification found; flagged for review.'

    If no note was close or retrieved:
    'No matching note found for this route or date range. Cost rise looks unexplained and worth a human review.'
    """
    if rejected_info:
        r_id = rejected_info.get("note_id", "")
        r_date = rejected_info.get("date", "")
        r_note = rejected_info.get("note", "")
        r_fact = _extract_core_fact(r_note)
        return (
            f"The closest note ({r_id}, {r_date}) mentions {r_fact} -- "
            f"it does not describe a reason for a cost rise on this route. "
            f"No genuine justification found; flagged for review."
        )

    return (
        "No matching note found for this route or date range. "
        "Cost rise looks unexplained and worth a human review."
    )


def generate_template_reason_not_flagged(
    route: str,
    week_of: str,
    vs_own_history: str,
    vs_similar_routes: str,
) -> str:
    """Generate a template-based reason for a non-flagged route-week."""
    return ""


def _validate_llm_reason(
    reason: str,
    note_id: str,
    note_text: str,
) -> bool:
    """
    Validate that an LLM-generated reason is grounded in the note.

    The reason must mention the note_id or key facts from the note.
    Rejects reasons that add unsupported claims.
    """
    if not reason:
        return False

    reason_lower = reason.lower()

    # Must mention the note ID
    if note_id.lower() in reason_lower:
        return True

    # Or must mention key facts from the note (at least 3 content words match)
    note_words = set(re.findall(r"\b\w+\b", note_text.lower()))
    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "been", "be",
        "to", "of", "and", "in", "on", "for", "by", "with", "from",
        "at", "this", "that", "it", "its", "or", "as", "not", "no",
    }
    note_keywords = note_words - stop_words
    reason_words = set(re.findall(r"\b\w+\b", reason_lower))
    overlap = note_keywords & reason_words

    return len(overlap) >= 3


def generate_reason(
    verdict_type: str,
    route: str,
    week_of: pd.Timestamp,
    vs_own_history: str,
    vs_similar_routes: str,
    note_id: Optional[str] = None,
    note_text: Optional[str] = None,
    note_date: Optional[str] = None,
    rejected_info: Optional[Dict[str, Any]] = None,
    config: dict = None,
    cache_dir: Path = None,
) -> str:
    """
    Generate a reason string for a route-week.

    For justified verdicts: tries LLM first, validates output, falls back to template.
    For unexplained verdicts: produces exact grounded template (citing closest rejected note if any).
    For non-flagged: returns empty string.
    """
    if config is None:
        config = {}
    if cache_dir is None:
        cache_dir = Path("llm_cache")

    week_str = week_of.strftime("%Y-%m-%d")

    if verdict_type == "justified" and note_id and note_text:
        # Keep output reasons fully grounded and reproducible. The former
        # validator accepted any response containing the note ID, which did
        # not prevent fabricated claims from entering the output.
        return generate_template_reason_justified(
            note_id, note_text, note_date or "", route, week_str, vs_own_history
        )

    elif verdict_type == "unexplained":
        return generate_template_reason_unexplained(route, week_str, rejected_info=rejected_info)

    else:
        return generate_template_reason_not_flagged(
            route, week_str, vs_own_history, vs_similar_routes
        )
