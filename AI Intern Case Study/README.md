# FreightTiger Smart Assistant: Shipping Cost Anomaly Detection

A deterministic data engineering and note-retrieval solution that identifies and explains shipping cost anomalies across commercial freight corridors.

---

## 1. System Architecture

The solution uses a hybrid deterministic-AI architecture designed to eliminate hallucinations while maintaining plain-English explainability:

1. **Ingestion & Validation Layer (`src/loading.py`)**:
   - Ingests `shipment_records.csv` and `context_notes.csv`.
   - Validates required columns and shipment nulls, rejects duplicate shipment IDs and non-positive shipment quantities, distances, or costs, and parses shipment and note dates.
   - Derives route identifiers (`Origin-Destination`) and aligns each shipment to its canonical calendar week (`week_of` = Monday).

2. **Deterministic Metric Engine (`src/metrics.py`)**:
   - Aggregates shipments by route and week.
   - Calculates weekly freight cost per tonne-kilometre:
     $$\text{cost\_per\_tonne\_km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$
   - Computes rolling 8-week historical baselines strictly prior to the current week without look-ahead.
   - Computes peer baselines across contemporaneous routes within the same length classification (`route_type`).

3. **Deterministic Flagging Module (`src/flagging.py`)**:
   - Applies transparent, mathematically grounded threshold logic to identify candidate anomalies.
   - Marks route-weeks for review when the own-history percentage is positive, exceeds the configured own-history threshold or peer threshold, and shipment count meets the configured minimum.

4. **Vector Retrieval & Note Guardrail Layer (`src/retrieval.py`)**:
   - Indexes `context_notes.csv` into a FAISS index using `sentence-transformers/all-MiniLM-L6-v2`.
   - Executes semantic retrieval for candidate notes using query embeddings constructed from route, timeline, and cost delta.
   - Applies deterministic code-level filtration: geographical match, temporal window validity (no future notes), and explicit cost-increase validation (rejecting negative/neutral controls).
   - Resolves conflicts using a deterministic tie-breaking protocol.

5. **Explanation & Verification Layer (`src/explain.py`)**:
   - Generates deterministic plain-English explanations for justified anomalies, citing the verified `matched_note_id` and note date.
   - Uses deterministic templates rather than LLM-authored reasons, so a generated sentence cannot add unsupported claims.
   - For unexplained anomalies, reports lack of matching notes or cites the nearest rejected candidate note (e.g. notes reporting normal conditions).

6. **Interactive Assistant CLI (`src/cli.py`, `ask.py`)**:
   - Provides a plain-English query interface enabling users to inspect cost anomalies by corridor and time period.

---

## 2. Metric & Baseline Definitions

All metric formulas and baseline calculations adhere to the exact specification provided in the case study brief:

- **Route Grouping**: A route is defined by unique `(origin, destination)` pairs and associated with its provided `route_type` (Short, Medium, Long). There is a one-to-one mapping between each origin-destination pair and its length bucket.
- **Weekly Alignment (`week_of`)**: Shipment dates are mapped to standard Monday-to-Sunday weeks. `week_of` represents the date of the Monday of that week (ISO format `YYYY-MM-DD`).
- **Cost Metric (`cost_per_tonne_km`)**:
  - Total freight expenditure divided by total tonne-kilometres delivered in that week:
    $$\text{cost\_per\_tonne\_km} = \frac{\sum_{i} \text{freight\_cost\_inr}_i}{\sum_{i} (\text{quantity\_tonnes}_i \times \text{distance\_km}_i)}$$
  - Aggregation is performed at the route-week level before division. Per-shipment unit cost ratios are never averaged together, avoiding distortion from varying shipment weights and trip lengths.
- **Comparison Baseline: `vs_own_history`**:
  - Trailing 8-week rolling arithmetic mean of the route's `cost_per_tonne_km`.
  - Calculated strictly over prior weeks ($t-8$ through $t-1$). Current and future weeks are excluded from the baseline.
  - **Handling of fewer than 8 prior weeks**: If fewer than 8 prior weeks are available, the baseline uses all available prior weeks ($1 \le k < 8$). No padding, backfilling, or extrapolation is applied.
  - **Initial week handling**: If zero prior weeks exist (Week 1 of the dataset), `vs_own_history` is left blank (`NaN` / empty string).
  - Formatted as signed percentage: `+X.X% vs this route's past average`.
- **Comparison Baseline: `vs_similar_routes`**:
  - Unweighted arithmetic mean of `cost_per_tonne_km` during the exact same week across all other routes sharing the same `route_type`.
  - The route itself is strictly excluded from its own peer average calculation.
  - Formatted as signed percentage: `+X.X% vs similar-length routes this week`.

---

## 3. Flagging Rule & Threshold Justification

A route-week is flagged as anomalous if and only if:
1. `vs_own_history_pct > 0` (cost is rising compared to historical trajectory), AND
2. The escalation is outside normal variation versus its own history OR versus peers:
   $$\text{vs\_own\_history\_pct} > 5.0\% \quad \text{OR} \quad \text{vs\_similar\_routes\_pct} > 15.0\%$$
3. Shipment volume meets the minimum threshold (`shipment_count >= 1`).

### Empirical Justification from Data Distribution

Analysis of the 721 route-weeks with an own-history baseline and all 728 peer comparisons across 2024-2025 yields the following distribution parameters:

| Distribution Metric | `vs_own_history` (%) | `vs_similar_routes` (%) |
|---|---|---|
| Mean | +0.42% | +0.36% |
| Standard Deviation ($\sigma$) | 4.25% | 8.98% |
| 50th Percentile (Median) | +0.22% | -0.38% |
| 75th Percentile | +2.61% | +5.85% |
| 90th Percentile | +4.48% | +10.55% |
| 95th Percentile | +5.75% | +13.43% |
| 98th Percentile | +7.50% | +22.68% |
| Maximum | +35.52% | +38.91% |

- **Own History Threshold (5.0%)**: Represents ~1.1 standard deviations above the historical mean and is near the 93rd percentile of valid own-history movements. Setting this threshold at 5.0% ensures detection of:
  - Nationwide diesel price increases (+5% to +7%) documented in note N003.
  - Regional weather disruptions, such as Chennai-Bangalore flooding in N001 (+31.5% on 2025-02-24 and +22.8% on 2025-03-03 versus own history in this run).
  - Corridors experiencing surcharge surges (+29.5%) like Ahmedabad-Mumbai festival week in note N002.
  - Severe unexplained anomalies like Delhi-Jaipur (+35.5%) and Mumbai-Pune (+9.2%).
- **Peer Threshold (15.0%)**: Represents the 96th percentile of peer divergence, identifying routes that deviate markedly from contemporaneous routes of identical length class.
- **Low-Volume Guard**: The supplied data has 3 to 5 shipments in each route-week; no artificial volume padding is necessary.

---

## 4. Anti-Hallucination Architecture

The assistant uses deterministic evidence checks for every note used to justify a flagged route-week:

1. **Deterministic Verdict Decision**:
   - Verdict decisions are executed exclusively by Python rules.
   - A candidate note is accepted as supporting evidence if and only if all three deterministic conditions hold:
     - **Rule A (Geography)**: `applies_to` matches the corridor or specifies "All Routes".
     - **Rule B (Timeline)**: The note date falls within the applicable lookback window and does not occur after the week's end (no future knowledge). The effect window is bounded at 14 days.
     - **Rule C (Cost Direction)**: The note explicitly reports factors causing cost increases. Notes stating that costs were unaffected, normal, or absorbed without rate changes are explicitly rejected.
2. **Deterministic Tie-Breaking**:
   - If multiple notes qualify, the system deterministically selects the most route-specific note, followed by the most recent note date, followed by alphabetical note ID.
3. **Deterministic Reasons**:
   - Reasons are generated from fixed templates and reviewed note facts. No LLM-generated explanation is accepted into output.
4. **Negative Control Rejection Reporting**:
   - When a note is retrieved in proximity to a flagged week but rejected because it describes stable or normal conditions (e.g. N006 on Mumbai-Pune), the assistant cites the rejected note to document why the rise remains unexplained.

---

## 5. Data Inspection Findings

Inspection of the primary data files revealed the following characteristics:

- **`shipment_records.csv`**:
  - Total records: 2,940 shipments across 7 distinct corridors.
  - Missing values / nulls: 0 across all columns.
  - Duplicate `shipment_id`: 0.
  - Negative or zero quantities/distances/costs: 0.
  - Temporal range: 2024-01-01 to 2025-12-28 (104 continuous weeks per route).
  - Corridors: Ahmedabad-Mumbai (Medium), Chennai-Bangalore (Medium), Delhi-Chennai (Long), Delhi-Jaipur (Short), Kolkata-Bhubaneswar (Medium), Mumbai-Delhi (Long), Mumbai-Pune (Short).
  - Each corridor consistently maps to exactly one `route_type`.
- **`context_notes.csv`**:
  - Total notes: 10 entries.
  - 3 notes report cost-increasing events: N001 (Chennai-Bangalore flood), N002 (Ahmedabad-Mumbai festival surcharge), N003 (Nationwide diesel price hike).
  - 7 notes serve as negative controls reporting normal conditions, non-applicable routes, or absorbed compliance costs (N004, N005, N006, N007, N008, N009, N010).
- **`sample_output_format_v2.csv`**:
  - Specifies contract for 8 columns: `route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason`.
  - Requires signed percentage formatting with exact phrasing.

---

## 6. Verification and Correctness

System correctness was validated across four independent test harnesses:

1. **Unit Test Suite (`tests/test_core.py`)**:
   - Tests cover week assignment across year boundaries, Monday-Sunday mappings, aggregate versus ratio calculations, trailing calendar lookback windows, self-exclusion from peer averages, percentage formatting, and note-matching filters.
   - Status: run `pytest -q` for the current result.
2. **Independent Baseline Recomputation (`tests/test_independent_baselines.py`)**:
   - Checks 5 selected route-weeks across different routes and years.
   - Recomputes unit costs, lookbacks, peer averages, and percentage deltas from raw CSV lines using pure Python dictionary and list operations (completely independent of pandas).
   - The script checks 5 selected route-weeks against current `output.csv`.
3. **Benchmark Evaluation Harness (`eval/eval_verdicts.py`)**:
   - Checks verdict and note-ID matching across 11 hand-labeled scenarios including positive justifications, nationwide price rises, negative controls, and sample output rows. It does not assert numeric comparison percentages.
4. **Three-Run Reproducibility Verification (`repro_check.py`)**:
   - Executes the complete pipeline three consecutive times on the raw input data and compares SHA-256 hashes.
   - Run `python repro_check.py` to generate a fresh report; previously recorded hashes are not treated as current verification.

---

## 7. How to Run

### Prerequisites

Python 3.11+ (tested on Python 3.14). Install dependencies:

```bash
pip install -r requirements.txt
```

### 1. Execute Primary Pipeline

Generates `output.csv` matching `sample_output_format_v2.csv`:

```bash
python main.py
```

Options:
- `--no-llm`: Retained compatibility option; explanations are template-based with or without it.
- `--config custom_config.yaml`: Supply an alternative configuration file.
- `--output custom_output.csv`: Specify destination CSV file.

### 2. Run Test Suites

```bash
pytest tests/test_core.py -v
python tests/test_independent_baselines.py
```

### 3. Run Benchmark Evaluation

```bash
python eval/eval_verdicts.py
```

### 4. Verify Three-Run Reproducibility

```bash
python repro_check.py
```

### 5. Query Plain-English CLI Assistant (`ask.py`)

```bash
# Direct question
python ask.py "why did route Ahmedabad-Mumbai get pricier in January 2025?"

# Negative finding check
python ask.py "why did route Mumbai-Delhi get pricier in July 2024?"

# Interactive console mode
python ask.py
```

---

## 8. Known Limitations & Trade-offs

1. **Fixed Temporal Windows for Disruption Events**:
   - External disruptions vary in duration. The system uses a 14-day effect window from the note date. While this accurately models the flood and festival notes in this dataset, real-world deployments would benefit from explicit `event_end_date` metadata in note schemas.
2. **Unweighted Peer Averaging**:
   - Per the brief specification, `vs_similar_routes` is calculated as an unweighted average of route unit costs. Weighting by route volume would be a natural extension for production environments.
3. **Explanations use deterministic templates**:
   - The pipeline makes no LLM API calls; semantic retrieval selects notes and fixed templates form the reason text.
