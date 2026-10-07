# Documentation index

Start with the root `README.md` (what it is, quickstart, results). Depth lives here.

| Document | Read it for |
| --- | --- |
| [architecture.md](architecture.md) | Training pipeline and request-flow diagrams; layering rules and how they are enforced |
| [dataset.md](dataset.md) | Data source and licence, the leakage found in the official split, how the split was cleaned, EDA findings |
| [model.md](model.md) | Model and training choices, the training run, calibration, threshold and review band |
| [evaluation.md](evaluation.md) | Test results with CIs, review-band effect, error analysis, latency trade-offs |
| [api.md](api.md) | Endpoints, response fields, error codes, curl examples |
| [MODEL_CARD.md](MODEL_CARD.md) | One-page summary: intended use, data, metrics, limitations |
| [SECURITY.md](SECURITY.md) | STRIDE-lite threat model (threat → mitigation → test), scanner results, residual risks |
| [RUNBOOK.md](RUNBOOK.md) | Deploy, health checks, rollback, model swap, key rotation, logs and metrics |
| [DECISIONS.md](DECISIONS.md) | Every non-trivial decision: alternatives, why, trade-off |

Other folders with their own README: `data/` and `models/`. See
`examples/example_predictions.md` for real API responses.

Figures used by these docs are in `figures/` (copied from `reports/figures/`).
