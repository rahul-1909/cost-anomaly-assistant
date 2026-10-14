"""
cli.py -- Stretch goal: Plain-English question answering.

A small CLI that answers questions about cost findings by querying
the computed results table and retrieved notes. Replies "no finding"
when the data does not support an answer.
"""

import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_route_from_question(question: str, known_routes: list) -> Optional[str]:
    """Extract a route name from a plain-English question."""
    q_lower = question.lower()
    for route in known_routes:
        if route.lower() in q_lower:
            return route
        # Also check with spaces around the hyphen
        spaced = route.lower().replace("-", " ")
        if spaced in q_lower:
            return route
        # Check origin and destination separately
        parts = route.split("-")
        if len(parts) == 2:
            origin, dest = parts
            if origin.lower() in q_lower and dest.lower() in q_lower:
                return route
    return None


def parse_time_from_question(question: str) -> Optional[str]:
    """
    Extract a time reference from a plain-English question.

    Handles patterns like:
    - "in March" -> "03"
    - "in March 2025" -> "2025-03"
    - "in week of 2025-01-20" -> "2025-01-20"
    - "in January" -> "01"
    """
    q_lower = question.lower()

    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    # Check for specific date
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", q_lower)
    if date_match:
        return date_match.group(1)

    # Check for "month year" pattern
    for month_name, month_num in months.items():
        pattern = rf"\b{month_name}\b\s*(\d{{4}})?"
        match = re.search(pattern, q_lower)
        if match:
            year = match.group(1)
            if year:
                return f"{year}-{month_num}"
            return month_num

    return None


def filter_results(
    results: pd.DataFrame,
    route: Optional[str],
    time_ref: Optional[str],
) -> pd.DataFrame:
    """Filter the results table by route and time reference."""
    filtered = results.copy()

    if route:
        filtered = filtered[filtered["route"] == route]

    if time_ref:
        # Convert week_of to string for matching
        filtered["week_str"] = filtered["week_of"].astype(str)
        if len(time_ref) == 2:
            # Month only - match any year
            filtered = filtered[filtered["week_str"].str[5:7] == time_ref]
        elif len(time_ref) == 7:
            # Year-month
            filtered = filtered[filtered["week_str"].str[:7] == time_ref]
        elif len(time_ref) == 10:
            # Exact date
            filtered = filtered[filtered["week_str"] == time_ref]
        filtered = filtered.drop(columns=["week_str"])

    return filtered


def answer_question(question: str, output_csv: Path) -> str:
    """
    Answer a plain-English question about cost findings.

    Queries the computed output.csv and returns a plain-English answer.
    Returns "no finding" when the data does not support an answer.
    """
    if not output_csv.exists():
        return "No output data found. Please run the pipeline first."

    results = pd.read_csv(output_csv)

    known_routes = results["route"].unique().tolist()

    route = parse_route_from_question(question, known_routes)
    time_ref = parse_time_from_question(question)

    if not route and not time_ref:
        return (
            "No finding: could not identify a specific route or time period "
            "from your question. Try asking about a specific route "
            f"({', '.join(known_routes)}) and/or time period."
        )

    filtered = filter_results(results, route, time_ref)

    if filtered.empty:
        route_str = route or "any route"
        time_str = time_ref or "any period"
        return f"No finding: no data for {route_str} in {time_str}."

    # Focus on flagged rows
    flagged = filtered[filtered["flagged"].str.startswith("Yes") |
                        filtered["flagged"].str.startswith("No (justified)")]

    if flagged.empty:
        route_str = route or "the specified routes"
        time_str = time_ref or "the specified period"
        return (
            f"No finding: {route_str} in {time_str} showed no flagged cost anomalies. "
            f"Costs were within normal range."
        )

    # Build answer from flagged rows
    lines = []
    for _, row in flagged.iterrows():
        line = (
            f"- Week of {row['week_of']}: cost was {row['cost_per_tonne_km']} INR/tonne-km. "
            f"{row['vs_own_history']}. {row['vs_similar_routes']}. "
            f"Status: {row['flagged']}."
        )
        if pd.notna(row.get("reason")) and row["reason"]:
            line += f" Reason: {row['reason']}"
        lines.append(line)

    route_str = route or "Multiple routes"
    time_str = time_ref or "the dataset period"
    header = f"Findings for {route_str} in {time_str}:\n"

    return header + "\n".join(lines)


def main() -> None:
    """CLI entry point for the question-answering assistant."""
    # Determine the project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_csv = project_root / "output.csv"

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("FreightTiger Smart Assistant - Ask a question about shipping costs.")
        print(f"Available data: {output_csv}")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if not question:
                continue
            answer = answer_question(question, output_csv)
            print(f"\n{answer}\n")
        return

    answer = answer_question(question, output_csv)
    print(answer)


if __name__ == "__main__":
    main()
