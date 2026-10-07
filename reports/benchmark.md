# CPU benchmark

Model `20261007T173514Z_c9fca37c51cb` · 4 threads · 10 warm-up + 100 timed calls per cell · real validation images.

## Latency and throughput

| Batch | Runtime | p50 ms | p95 ms | p99 ms | images/s |
| --- | --- | --- | --- | --- | --- |
| 1 | pytorch_fp32 | 31.65 | 42.89 | 53.69 | 31.6 |
| 1 | onnxruntime_fp32 | 9.87 | 10.82 | 11.14 | 101.3 |
| 1 | onnxruntime_int8 | 184.21 | 235.91 | 247.95 | 5.4 |
| 8 | pytorch_fp32 | 204.93 | 229.48 | 244.05 | 39.0 |
| 8 | onnxruntime_fp32 | 99.43 | 128.75 | 191.77 | 80.5 |
| 8 | onnxruntime_int8 | 1833.71 | 2170.69 | 2326.09 | 4.4 |
| 32 | pytorch_fp32 | 948.64 | 1056.69 | 1117.34 | 33.7 |
| 32 | onnxruntime_fp32 | 416.49 | 487.26 | 544.73 | 76.8 |
| 32 | onnxruntime_int8 | 6428.08 | 8508.05 | 18120.12 | 5.0 |

## Validation accuracy at the operating threshold

| Runtime | PR-AUC | ROC-AUC | Precision | Recall | F1 | max Δp vs PyTorch | decisions changed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch_fp32 | 1.0000 | 1.0000 | 1.0000 | 0.9982 | 0.9991 | 0.00e+00 | 0 |
| onnxruntime_fp32 | 1.0000 | 1.0000 | 1.0000 | 0.9982 | 0.9991 | 2.91e-06 | 0 |
| onnxruntime_int8 | 0.6712 | 0.5727 | 0.7379 | 0.1387 | 0.2335 | 1.00e+00 | 498 |

File sizes (MB): {"onnx_fp32": 16.02, "onnx_int8": 4.31}
