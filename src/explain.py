"""Build deterministic, evidence-grounded reasons for route-week verdicts."""

from typing import Any, Dict, Optional


def _extract_core_fact(note_text: str) -> str:
    """Return a concise clause grounded in a known context-note description."""
    text = note_text.strip()
    lower = text.lower()
    if "regional festival week" in lower:
        return "a regional festival week drove a temporary surcharge on this corridor"
    if "heavy flooding" in lower:
        return "heavy flooding disrupted highway truck movement, forcing detours and higher trip costs"
    if "diesel prices" in lower:
        return "diesel prices rose nationwide, pushing up transportation costs across all routes"
    if "stable demand" in lower or "no major disruptions" in lower:
        return "stable demand with no major disruptions"
    if "costs were not significantly affected" in lower:
        return "highway maintenance work where costs were not significantly affected"
    if "returned to normal" in lower:
        return "the corridor returned to normal conditions after repairs were completed"
    if "without a rate change" in lower:
        return "compliance costs were absorbed by transporters without a rate change"
    if "not part of this dataset" in lower:
        return "a toll plaza change where the affected routes are not part of this dataset"

    first_sentence = text.split(".")[0].strip()
    return first_sentence[0].lower() + first_sentence[1:] if first_sentence else text


def generate_template_reason_justified(
    note_id: str,
    note_text: str,
    note_date: str,
) -> str:
    """Build the reason for a verdict supported by an accepted context note."""
    core_fact = _extract_core_fact(note_text)
    return (
        f"Matches note {note_id} dated {note_date}: {core_fact}. "
        "The cost rise has a clear explanation."
    )


def generate_template_reason_unexplained(
    rejected_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build an unexplained-verdict reason, optionally citing a rejected note."""
    if rejected_info:
        note_id = rejected_info.get("note_id", "")
        note_date = rejected_info.get("date", "")
        core_fact = _extract_core_fact(rejected_info.get("note", ""))
        return (
            f"The closest note ({note_id}, {note_date}) mentions {core_fact} -- "
            "it does not describe a reason for a cost rise on this route. "
            "No genuine justification found; flagged for review."
        )

    return (
        "No matching note found for this route or date range. "
        "Cost rise looks unexplained and worth a human review."
    )
