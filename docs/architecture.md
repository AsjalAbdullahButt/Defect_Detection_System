# Architecture

Two separate halves share one torch-free package, `core/` (class names, preprocessing, model
metadata, hashing):

- **Training** (offline, PyTorch): data audit → split → train → calibrate → evaluate once → export.
- **Serving** (online, ONNX Runtime): a FastAPI app that loads one verified, versioned model.

The only thing that crosses from training to serving is the immutable folder
`models/<version>/` (`model.onnx`, `model_meta.json`, `SHA256SUMS`).

## Training pipeline

![Training pipeline](architecture_training.png)

```mermaid
flowchart LR
    raw[("data/raw<br/>casting 300px + 512px")] --> inspect["inspect<br/>decode every file"]
    inspect --> manifest["manifest<br/>SHA-256 per image"]
    manifest --> split["split<br/>exact dups · same-part copies<br/>(aligned correlation >= 0.99)<br/>group-aware, stratified"]
    split --> audit{"leakage audit<br/>content-based"}
    audit -- fails --> stop(["CI fails"])
    audit -- passes --> splits[("splits.csv<br/>train · val · test · external_test")]
    splits --> train["train<br/>EfficientNet-B0, 2 stages<br/>weighted CE, early stop on val PR-AUC"]
    core1[["core/preprocessing<br/>(shared with serving)"]] -.-> train
    train --> ckpt[("best.pt<br/>weights_only")]
    ckpt --> calibrate["calibrate (VAL only)<br/>temperature · threshold<br/>review band"]
    calibrate --> evaluate["evaluate ONCE<br/>official test + external test<br/>bootstrap CIs"]
    evaluate --> export["export<br/>ONNX opset 17 · checker<br/>PyTorch/ONNX parity"]
    export --> artifact[("models/&lt;version&gt;/<br/>model.onnx · model_meta.json<br/>SHA256SUMS")]
```

## Serving request flow

![Serving request flow](architecture_serving.png)

```mermaid
flowchart TB
    client(["client"]) -->|"POST /v1/predict<br/>X-API-Key"| rid["RequestId<br/>assign / sanitise id"]
    rid --> headers["SecurityHeaders"]
    headers --> access["AccessLog + HTTP metrics"]
    access --> cors["CORS (allow-list, off by default)"]
    cors --> auth{"API-key auth<br/>constant-time, before body"}
    auth -- "missing / wrong" --> e401(["401 · failures throttled -> 429"])
    auth --> rate{"rate limit<br/>per key / IP"}
    rate -- exceeded --> e429(["429 + Retry-After"])
    rate --> size{"body size<br/>while streaming"}
    size -- too big --> e413(["413"])
    size --> route["route (HTTP only)"]
    route --> gate{"InferenceGate<br/>slots + timeout"}
    gate -- full --> e503(["503"])
    gate -- slow --> e504(["504"])
    gate --> validate{"validate image<br/>magic bytes · pixel cap<br/>verify · frames · dims"}
    validate -- bad --> e4xx(["400 / 415"])
    validate --> service["PredictionService<br/>core.preprocess -> logits<br/>softmax(logits/T) · threshold · review band"]
    service --> engine["OnnxPredictor<br/>(Predictor protocol)"]
    engine --> service
    service --> resp(["200 JSON<br/>class · confidence · p(defect)<br/>needs_review · model_version"])
    startup["startup (lifespan)<br/>SHA256SUMS + schema check<br/>refuse to start on mismatch"] -.-> engine
    service -. counters .-> metrics[("Prometheus<br/>internal :9100")]
```

## Layering rules (enforced, not just documented)

| Rule | Enforced by |
| --- | --- |
| `core/` and `serving/` never import torch/torchvision/timm | ruff `TID251` banned-api + `test_serving_never_imports_torch_or_training_stack` |
| Training and serving preprocess identically | `core/preprocessing.py` is the only implementation; `tests/leakage/test_preprocessing_parity.py` |
| Serving depends on the engine only through `Predictor` | `services/prediction.py` type-checks against the protocol (mypy strict) |
| Nothing is learned from test | calibration/threshold read `val` only (asserted in tests); `evaluate` refuses a second run |
| The model artifact is verified before use | `load_verified_meta` (checksums, schema, graph input) at startup |
