# Run Cost Log

Measured for one full run over `shipment_records.csv` (2,940 shipments, 728 route-weeks):

| Metric | Value |
|---|---:|
| Total input tokens | 0 |
| Total output tokens | 0 |
| LLM calls | 0 |
| Total tokens | 0 |
| Rough LLM API cost | $0.00 |

The pipeline uses deterministic reason templates and makes no LLM API calls.
Note retrieval runs locally with the pinned sentence-transformer model and FAISS;
its local compute cost is not included in the LLM API estimate.
