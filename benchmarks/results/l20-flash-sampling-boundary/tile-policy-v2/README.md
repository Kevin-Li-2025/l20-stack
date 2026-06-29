# L20 LM-Head FlashSampling Tile Policy V2

This artifact records the L20 tile sweep that followed the first native vLLM
FlashSampling candidate run. The goal was not to claim a serving win; it was to
repair the standalone candidate's launch policy after the c4 serving run showed
a throughput regression.

## Decision

- Batch 1 default: `BLOCK_VOCAB=32`, `BLOCK_HIDDEN=256`.
- Batched default: `BLOCK_VOCAB=64`, `BLOCK_HIDDEN=256`.
- Larger vocab tiles (`128` / `256`) did not improve latency on L20; the
  `256x256` tile exceeds the available shared-memory budget.

## Best Rows

| Shape | Old/default-ish row | Best row | Candidate / full logits | Effect |
| --- | ---: | ---: | ---: | ---: |
| b1 h1024 | `32x64`: 1.080x | `32x256` or `64x256`: 1.011x | still slower | removes most batch-one regression |
| b1 h1536 | `32x64`: 1.102x | `32x256`: 1.017x | still slower | removes most batch-one regression |
| b4 h1024 | `64x128`: 0.930x | `64x256`: 0.911x | faster | improves candidate micro speedup |
| b4 h1536 | `64x128`: 0.949x | `64x256`: 0.930x | faster | improves candidate micro speedup |

Ratios are `candidate_median_ms / full_logits_reference_median_ms`, so lower is
better. Raw per-shape JSON files are under `raw/`, and the aggregated table is
`summary.json`.

## Interpretation

The result is useful but bounded. `BLOCK_HIDDEN=256` fixes an avoidable
standalone-kernel policy issue, but batch-one remains slightly slower than the
full-logits reference. This reinforces the project conclusion: the next real
serving win needs a true LM-head GEMM epilogue path, not a separate replacement
GEMV-style kernel.
