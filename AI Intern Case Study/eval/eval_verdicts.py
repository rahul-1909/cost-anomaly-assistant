"""
eval_verdicts.py -- Evaluation harness for flagged verdicts and note matching.

Uses a hand-labeled benchmark set of expected verdicts and note IDs across
known ground-truth scenarios (surcharges, flood disruption, fuel price increases,
lingering post-disruption spikes, and negative controls). Computes precision
and recall and prints a detailed report.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Benchmark test cases grounded in context_notes.csv and actual shipment data:
BENCHMARK_CASES = [
    # 1. Sample row 1: Delhi-Jaipur sharp spike (+35.5%), completely unexplained
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "scenario": "Sample Row 1: Severe spike (+35.5% vs own, +21.0% vs peers), no note applies",
    },
    # 2. Sample row 2: Ahmedabad-Mumbai festival surcharge (+29.5%), justified by N002
    {
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N002",
        "scenario": "Sample Row 2: Regional festival surcharge corridor (+29.5% vs own), justified by N002",
    },
    # 3. Sample row 3: Mumbai-Pune creep (+9.2%), peer outlier (+23.6%), unexplained despite nearby N006
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "scenario": "Sample Row 3: Creep (+9.2% vs own, +23.6% vs peers). N006 cites stable demand -> rejected",
    },
    # 4. Chennai-Bangalore flooding week 1 (+31.5% own, +38.7% peers), justified by N001
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-02-24",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "scenario": "Highway flooding week 1 (+31.5% vs own, +38.7% vs peers), justified by N001",
    },
    # 5. Chennai-Bangalore flooding week 2 (+22.8% own, +32.0% peers), justified by N001
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-03",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N001",
        "scenario": "Highway flooding week 2 (+22.8% vs own, +32.0% vs peers, through Mar 8), justified by N001",
    },
    # 6. Mumbai-Pune diesel price increase week (+6.6%), justified by N003
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "scenario": "Nationwide diesel price hike 5-7% (+6.6% on route), justified by N003",
    },
    # 7. Delhi-Chennai diesel price increase week (+6.2%), justified by N003
    {
        "route": "Delhi-Chennai",
        "week_of": "2025-05-05",
        "expected_flagged": "No (justified)",
        "expected_note_id": "N003",
        "scenario": "Nationwide diesel price hike 5-7% (+6.2% on route), justified by N003",
    },
    # 8. Mumbai-Pune week of 2025-09-22 (+2.6% own, +18.2% peers). Note N006 cites stable demand -> must NOT justify
    {
        "route": "Mumbai-Pune",
        "week_of": "2025-09-22",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "scenario": "Peer divergence (+18.2%; own history +2.6%). N006 cites stable demand, no disruptions -> must NOT justify",
    },
    # 9. Chennai-Bangalore post-flood week of 2025-03-17 (+13.7% own). N009 confirms normal -> must NOT justify
    {
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-17",
        "expected_flagged": "Yes",
        "expected_note_id": "",
        "scenario": "Post-flood cost rise (+13.7%). Flood ended Mar 8, N009 confirms normal -> unexplained",
    },
    # 10. Negative control: Mumbai-Delhi maintenance (N005). Small movement stays below thresholds.
    {
        "route": "Mumbai-Delhi",
        "week_of": "2024-07-29",
        "expected_flagged": "No",
        "expected_note_id": "",
        "scenario": "Negative control N005: maintenance delays, but +0.4% own-history and +3.4% peer movements stay below thresholds",
    },
    # 11. Negative control: Delhi-Jaipur toll plaza (N004). Routes not in dataset; movements stay below thresholds.
    {
        "route": "Delhi-Jaipur",
        "week_of": "2024-03-11",
        "expected_flagged": "No",
        "expected_note_id": "",
        "scenario": "Negative control N004: affected routes not in dataset; +2.0% own history and -6.8% peer comparison -> unflagged",
    },
]


def evaluate(output_csv: Path) -> dict:
    """
    Evaluate output.csv against the benchmark test cases.

    Computes verdict accuracy and exact note-ID matching accuracy.
    """
    if not output_csv.exists():
        print(f"ERROR: Output file not found: {output_csv}")
        return {}

    output = pd.read_csv(output_csv)

    results = []
    verdict_correct_count = 0
    note_correct_count = 0

    for case in BENCHMARK_CASES:
        row = output[
            (output["route"] == case["route"]) &
            (output["week_of"] == case["week_of"])
        ]

        if row.empty:
            results.append({
                **case,
                "found": False,
                "actual_flagged": None,
                "actual_note_id": None,
                "verdict_match": False,
                "note_match": False,
                "detail": "Row missing from output.csv",
            })
            continue

        row = row.iloc[0]
        actual_flagged = str(row.get("flagged", ""))
        actual_note = str(row.get("matched_note_id", ""))
        if actual_note == "nan":
            actual_note = ""

        verdict_match = (actual_flagged == case["expected_flagged"])
        note_match = (actual_note == case["expected_note_id"])

        if verdict_match:
            verdict_correct_count += 1
        if note_match:
            note_correct_count += 1

        results.append({
            **case,
            "found": True,
            "actual_flagged": actual_flagged,
            "actual_note_id": actual_note,
            "verdict_match": verdict_match,
            "note_match": note_match,
            "detail": "PASS" if (verdict_match and note_match) else "MISMATCH",
        })

    total = len(BENCHMARK_CASES)
    verdict_acc = verdict_correct_count / total if total > 0 else 0.0
    note_acc = note_correct_count / total if total > 0 else 0.0

    return {
        "results": results,
        "total": total,
        "verdict_correct": verdict_correct_count,
        "note_correct": note_correct_count,
        "verdict_accuracy": verdict_acc,
        "note_accuracy": note_acc,
    }


def print_report(eval_result: dict) -> None:
    """Print formatted evaluation report."""
    if not eval_result:
        print("No evaluation results available.")
        return

    print("=" * 85)
    print("FREIGHTTIGER SMART ASSISTANT -- BENCHMARK EVALUATION REPORT")
    print("=" * 85)

    results = eval_result["results"]
    for r in results:
        status = "[PASS]" if r["verdict_match"] and r["note_match"] else "[FAIL]"
        print(f"\n{status} {r['route']} @ {r['week_of']}")
        print(f"  Scenario: {r['scenario']}")
        print(f"  Expected: flagged='{r['expected_flagged']}', matched_note_id='{r['expected_note_id']}'")
        if r["found"]:
            print(f"  Actual:   flagged='{r['actual_flagged']}', matched_note_id='{r['actual_note_id']}'")
            print(f"  Verdict Match: {r['verdict_match']} | Note Match: {r['note_match']}")
        else:
            print(f"  Status: {r['detail']}")

    print(f"\n{'=' * 85}")
    print(f"EVALUATION SUMMARY")
    print(f"{'=' * 85}")
    print(f"Total Benchmark Cases:  {eval_result['total']}")
    print(f"Verdict Accuracy:       {eval_result['verdict_accuracy']:.2%} ({eval_result['verdict_correct']}/{eval_result['total']})")
    print(f"Note Matching Accuracy: {eval_result['note_accuracy']:.2%} ({eval_result['note_correct']}/{eval_result['total']})")
    all_passed = sum(1 for r in results if r["verdict_match"] and r["note_match"])
    print(f"End-to-End Success:     {all_passed}/{eval_result['total']} cases passed completely")
    print("=" * 85)


if __name__ == "__main__":
    output_csv = PROJECT_ROOT / "output.csv"
    eval_result = evaluate(output_csv)
    print_report(eval_result)
