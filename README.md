# FreightTiger Cost Anomaly Assistant

A Python pipeline that computes weekly freight-cost metrics, flags route-week anomalies, and matches flagged rows to contextual notes using semantic retrieval plus deterministic evidence checks.

## How it works

1. `src/loading.py` validates and loads `shipment_records.csv`, `context_notes.csv`, and `config.yaml`. Shipment dates are assigned to Monday-starting calendar weeks.
2. `src/metrics.py` aggregates cost and tonne-kilometres for each route-week, then calculates the own-history and same-week peer comparisons.
3. `src/flagging.py` flags a route-week when its cost is rising versus its own history and exceeds either configured anomaly threshold. Every route-week in the supplied data has 3–5 shipments.
4. `src/retrieval.py` embeds the context notes with the pinned Sentence Transformers model and searches a FAISS index. Python rules then require route applicability, a matching note-text date/duration interval, and language indicating a cost increase. The rules reject future notes and negative or neutral statements.
5. `src/explain.py` writes fixed reason templates from the accepted note facts. No LLM is called and no model-generated text can change a verdict or reason.

The pipeline is run from the repository root with `python main.py`. It writes `output.csv` using the contract in `sample_output_format_v2.csv`.

## Metric and baseline definitions

- **Route**: a unique `(origin, destination)` pair, associated with its supplied `route_type`.
- **Week**: Monday through Sunday. `week_of` is the Monday date in `YYYY-MM-DD` form.
- **Weekly cost per tonne-kilometre**: `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`, aggregated at the route-week level before division. Per-shipment ratios are not averaged.
- **Own-history comparison**: the arithmetic mean of the route's observations in the strictly prior eight calendar weeks, from `week_of - 8 weeks` up to but excluding `week_of`. Available observations are used when fewer than eight are present; missing calendar weeks are not backfilled. With no prior observation, the comparison is blank.
- **Peer comparison**: the unweighted mean cost for other routes of the same `route_type` in the same week. The route itself is excluded. In this dataset, Short and Long each have two routes, so each has a single peer; Medium has three routes.
- **Flag rule**: own-history change must be greater than 0%, and either the own-history change must be greater than 5% or the peer change must be greater than 15%. These are strict greater-than comparisons.

## Note matching and reason safety

The selected note must apply to the route (or say `All Routes`), overlap the flagged week according to the dates or duration in its text, and describe a cost increase. A note dated after the flagged week's Sunday is rejected. Explicit date ranges such as `from Feb 24 to Mar 8`, stated durations such as `for about a week`, and phrases such as `this week`, `festival week`, and `this quarter` define the note interval. When no date range or duration is stated, applicability is limited to the calendar week containing the note date. The pipeline does not apply a generic multi-week effect window.

When multiple notes qualify, the pipeline prefers a route-specific note, then the most recent date, then the alphabetically first note ID. Explanations are deterministic templates that cite the selected note. An unexplained row has a blank `matched_note_id`; a rejected note may be described in the reason but is not recorded as matched evidence.

## Data and how it was built

The supplied shipment file contains 2,940 records, seven routes, and 728 route-weeks from 2024-01-01 through 2025-12-28. Each route-week has 3–5 shipments. The notes file contains 10 contextual notes, including three cost-increase notes and seven negative controls.

The implementation uses Python, pandas, NumPy, PyYAML, Sentence Transformers, and FAISS. Dependencies are pinned in `requirements.txt`; the embedding model revision is pinned in `config.yaml`. Metric, flag, and reason decisions are implemented in Python. There are no API credentials or external LLM calls required.

## Verification

Run the pipeline first so the tests and evaluation use a freshly generated `output.csv`:

```bash
python main.py
```

Then run the checks from the repository root:

```bash
python -m pytest -q
python tests/test_independent_baselines.py
python eval/eval_verdicts.py
python repro_check.py
```

The unit tests cover week assignment, aggregate cost calculations, strict prior-week baselines, peer self-exclusion, thresholds, note route/date/cost-direction rules, output column order and inferred types, and cited-note validity for every justified row. The independent baseline script recomputes selected route-week values from the raw CSV using Python's standard library. The evaluation script reports verdict and note-matching accuracy on its labeled scenarios. The reproducibility script runs the pipeline three times, compares output hashes, writes the report to `repro_check.txt`, and removes temporary `run1.csv`, `run2.csv`, and `run3.csv` files.

## Running options

Install the pinned dependencies and run from the repository root:

```bash
python -m pip install -r requirements.txt
python main.py
```

Optional pipeline arguments:

```bash
python main.py --config custom_config.yaml
python main.py --output custom_output.csv
```

The plain-English query tool can also be run from the repository root:

```bash
python ask.py "why did route Ahmedabad-Mumbai get pricier in January 2025?"
python ask.py
```

## Known limitations

- Short and Long route types each have only two routes, so each peer average is based on a single other route. Medium route peer averages use two other routes.
- The note interval parser recognizes explicit month/day ranges, common numeric or written durations in days, weeks, or months, and the phrases `this week`, `festival week`, and `this quarter`. Other temporal wording falls back to the note's calendar week and may need an explicit parser rule for broader data.
- The reason templates cover the supplied note patterns. New note wording may need additional cost-direction or reason-template rules.
- The evaluation scenarios are a small fixed benchmark and do not measure the quality of semantic search over a broad production corpus.
- The pinned embedding model must be downloaded once or already present in the local Hugging Face cache. A first run without the cached model requires Hugging Face access.
- The pipeline reports statistical deviations, not confirmed fraud or proof of causation. Human review is required for operational decisions.
