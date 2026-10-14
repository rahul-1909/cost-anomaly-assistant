# FreightTiger Cost Anomaly Assistant

A Python pipeline that computes weekly freight-cost metrics, flags route-week anomalies, and matches flagged rows to contextual notes using note retrieval plus deterministic evidence checks.

## Python support

Tested with Python 3.14.7. The minimum supported version is Python 3.12, the minimum required by the pinned NumPy dependency.

## How it works

1. `src/loading.py` validates and loads `shipment_records.csv`, `context_notes.csv`, and `config.yaml`. Shipment dates are assigned to Monday-starting calendar weeks.
2. `src/metrics.py` aggregates cost and tonne-kilometres for each route-week, then calculates the own-history and same-week peer comparisons.
3. `src/flagging.py` flags a route-week when its cost is rising versus its own history and exceeds either configured anomaly threshold. Every route-week in the supplied data has 3–5 shipments.
4. `src/retrieval.py` embeds the context notes with the pinned Sentence Transformers model and searches a FAISS index. If the model cannot load, it logs a warning and uses deterministic lexical ranking. Python rules then require route applicability, a matching note-text date/duration interval, and language indicating a cost increase. The rules reject future notes and negative or neutral statements.
5. `src/explain.py` writes fixed reason templates from the accepted note facts. No LLM is called and no model-generated text can change a verdict or reason.

The pipeline is run from the repository root with `python main.py`. It writes `output.csv` using the contract in `sample_output_format_v2.csv`.

## Metric and baseline definitions

- **Route**: a unique `(origin, destination)` pair, associated with its supplied `route_type`.
- **Week**: Monday through Sunday. `week_of` is the Monday date in `YYYY-MM-DD` form.
- **Weekly cost per tonne-kilometre**: `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`, aggregated at the route-week level before division. Per-shipment ratios are not averaged.
- **Own-history comparison**: the arithmetic mean of the route's observations in the strictly prior eight calendar weeks, from `week_of - 8 weeks` up to but excluding `week_of`. Available observations are used when fewer than eight are present; missing calendar weeks are not backfilled. With no prior observation, the comparison is blank.
- **Peer comparison**: the unweighted mean cost for other routes of the same `route_type` in the same week. The route itself is excluded. In this dataset, Short and Long each have two routes, so each has a single peer; Medium has three routes.
- **Flag rule**: own-history change must be greater than 0%, and either the own-history change must be greater than 5% or the peer change must be greater than 15%. These are strict greater-than comparisons.

Threshold statistics were recomputed from the unrounded route-week percentage changes using sample standard deviation. Own-history: n=721, mean=+0.42%, sample std=4.25%, median=+0.22%, p75=+2.61%, p90=+4.48%, p92=+4.85%, p95=+5.75%. Peer: n=728, mean=+0.36%, sample std=8.98%, median=-0.38%, p95=+13.43%, p96=+14.32%. The 5.0% own-history threshold is just above p92; the 15.0% peer threshold is above p96. Observed N001 flood weeks rose +31.5% and +22.8% versus own history; N002's festival week rose +29.5%. Delhi-Jaipur (+35.5%) and Mumbai-Pune (+9.2%) are example unexplained spikes in the sample.

## Note matching and reason safety

The selected note must apply to the route (or say `All Routes`), overlap the flagged week according to the dates or duration in its text, and describe a cost increase. A note dated after the flagged week's Sunday is rejected. Explicit date ranges such as `from Feb 24 to Mar 8`, stated durations such as `for about a week`, and phrases such as `this week`, `festival week`, and `this quarter` define the note interval. When no date range or duration is stated, applicability is limited to the calendar week containing the note date. The pipeline does not apply a generic multi-week effect window.

When multiple notes qualify, the pipeline prefers a route-specific note, then the most recent date, then the alphabetically first note ID. Explanations are deterministic templates that cite the selected note. An unexplained row has a blank `matched_note_id`; a rejected note may be described in the reason but is not recorded as matched evidence.

There are only 10 notes in this case-study corpus, so semantic retrieval is a small-corpus candidate-ranking step rather than evidence of broad production search quality. The FAISS rank audit puts the cited notes for all eight justified output rows in positions 1–3, so `rag_top_k` is 5 for this supplied dataset; this small audit says nothing about recall on a larger or new corpus. Deterministic Python rules check retrieved notes for route, time-window, and cost-increase validity; those rules decide the verdict, and fixed Python templates produce the reason. If the embedding model cannot load, the pipeline logs a warning and uses deterministic lexical ranking; the regression test confirms identical output for this dataset. No LLM call can change a verdict or invent a reason.

## Data and how it was built

The supplied shipment file contains 2,940 records, seven routes, and 728 route-weeks from 2024-01-01 through 2025-12-28. Each route-week has 3–5 shipments. The notes file contains 10 contextual notes, including three cost-increase notes and seven negative controls.

The implementation uses Python, pandas, NumPy, PyYAML, Sentence Transformers, and FAISS, with a deterministic lexical retrieval fallback. Dependencies are pinned in `requirements.txt`; the embedding model revision is pinned in `config.yaml`. Metric, flag, note validation, and reason decisions are implemented in Python. There are no API credentials or external LLM calls required.

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
python retrieval_rank_audit.py
python repro_check.py
```

The pytest suite covers week assignment, aggregate cost calculations, strict prior-week baselines, peer self-exclusion, configured threshold boundaries, route-type consistency, note route/date/cost-direction rules, output column order and inferred types, and the hardcoded justified-row oracle. The independent baseline script recomputes five selected route-week metrics from the raw CSV using Python's standard library. Its last-eight-observed baseline equals the eight-calendar-week window for this dataset because no route-weeks are missing. `eval/eval_verdicts.py` is a fixed regression benchmark for verdict and note matching, not an independent label set. The reproducibility script runs the pipeline three times, compares output hashes, writes the report to `repro_check.txt`, and removes temporary `run1.csv`, `run2.csv`, and `run3.csv` files. `retrieval_rank_audit.py` reports each justified row's gold-note position in the FAISS results.

## Repository layout

- `.gitignore` - excludes secrets, virtual environments, and generated caches.
- `FreightTiger_Intern_CaseStudy.pdf` - case-study brief already included in the repository.
- `README.md` - usage, methodology, verification, and limitations.
- `ask.py` - optional plain-English query CLI.
- `config.yaml` - metric, flagging, model, and output settings.
- `context_notes.csv` - contextual notes used to explain eligible anomalies.
- `cost_log.md` - LLM token and cost accounting for a pipeline run.
- `eval/` - regression benchmark directory.
- `eval/eval_verdicts.py` - fixed verdict and note-matching regression benchmark.
- `main.py` - root-level pipeline entry point that writes `output.csv`.
- `output.csv` - generated route-week results.
- `repro_check.py` - three-run SHA-256 reproducibility check.
- `repro_check.txt` - latest saved reproducibility report.
- `requirements.txt` - pinned Python dependencies.
- `retrieval_rank_audit.py` - gold-note rank audit for justified output rows.
- `sample_output_format_v2.csv` - output schema and formatting contract.
- `shipment_records.csv` - shipment-level input data.
- `src/` - pipeline implementation package.
- `src/__init__.py` - source package marker.
- `src/cli.py` - optional plain-English query implementation.
- `src/explain.py` - deterministic reason templates.
- `src/flagging.py` - deterministic anomaly flag rules.
- `src/loading.py` - input and configuration validation.
- `src/metrics.py` - weekly costs and own-history/peer comparisons.
- `src/retrieval.py` - note retrieval and deterministic eligibility checks.
- `tests/` - automated test suite.
- `tests/__init__.py` - test package marker.
- `tests/test_core.py` - unit tests for metrics, flagging, and note rules.
- `tests/test_independent_baselines.py` - independent raw-data metric checks.
- `tests/test_justified_oracle.py` - justified route-week-note oracle checks.
- `tests/test_output_contract.py` - output schema and note-evidence checks.
- `tests/test_retrieval_fallback.py` - deterministic retrieval fallback checks.

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
- FAISS retrieval and the rank-audit script require the pinned embedding model to be available locally or downloadable from Hugging Face. The main pipeline logs a warning and switches to deterministic lexical ranking when model loading fails.
- The pipeline reports statistical deviations, not confirmed fraud or proof of causation. Human review is required for operational decisions.
