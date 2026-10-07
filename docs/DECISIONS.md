# Decision log

Format: **Decision · Alternatives · Why · Trade-off**. One entry per non-trivial choice.

## V0 — Foundation

### D-001 Python 3.12 (not the system 3.14)

- **Alternatives:** 3.14 (installed), 3.11.
- **Why:** 3.12 has the most mature wheels across torch, onnxruntime, timm and the security tooling. Dev, CI and Docker all use this one version.
- **Trade-off:** misses 3.13/3.14 interpreter speed-ups. uv installs 3.12 automatically, so setup costs nothing extra.

### D-002 `uv pip compile --generate-hashes --universal` for lockfiles

- **Alternatives:** pip-tools `pip-compile`; a uv/Poetry project lock (`uv.lock`).
- **Why:** it writes the same hashed `requirements.txt` format as pip-tools, so `pip --require-hashes` works. It is 10–100x faster. `--universal` adds platform markers, so one lockfile serves Windows dev and the Linux image.
- **Trade-off:** dev machines need uv. `[tool.uv] managed = false` stops uv from also creating a competing `uv.lock`.

### D-003 `requirements/*.in` are the only dependency declaration

- **Alternatives:** list deps in `pyproject.toml` and duplicate them in `.in` files.
- **Why:** `pyproject.toml` reads its extras from the `.in` files (setuptools dynamic metadata), so the two can't drift. I added `base.in` (numpy, pillow, pydantic) for what `core/` needs in both training and serving. This file is not in §2's list.
- **Trade-off:** `.in` files can't contain `-r`/`-c` lines, so layering is done on the command line instead (D-004).

### D-004 Layered lockfiles: serve → train (`-c serve.txt`) → dev (`-c train.txt -c serve.txt`)

- **Alternatives:** independent lockfiles; one combined lockfile.
- **Why:** serving is the smallest and most security-sensitive set, so it resolves first. Training is then held to the same versions of shared packages (numpy, pillow, onnxruntime), so parity tests compare identical libraries.
- **Trade-off:** training can't upgrade a shared package beyond the serving pin without re-locking all three.

### D-005 Architecture boundary enforced by the linter (ruff TID251)

- **Alternatives:** a runtime test only; import-linter.
- **Why:** `torch`, `torchvision` and `timm` are banned project-wide and allowed again only in `training/`, `data/dataset.py` and `data/augment.py`. `ruff check` fails as soon as `core/` or `serving/` imports torch. A runtime `sys.modules` test (V6) backs this up.
- **Trade-off:** a new torch-using module has to be added to the per-file allow-list in `pyproject.toml`.

### D-006 mypy strict flags only for `core/` and `serving/`

- **Alternatives:** `strict = true` globally.
- **Why:** mypy's `strict` setting is global-only, so the strict flags are listed in a per-module override. Every module still needs full annotations. `data/` and `training/` are relaxed about `Any` coming from torch/pandas, whose type stubs are incomplete.
- **Trade-off:** the flags are listed by hand, so new mypy strict flags have to be added manually.

### D-007 GNU make on Windows; recipes call only `uv`

- **Alternatives:** PowerShell task script; `just`; `nox`.
- **Why:** the brief requires a Makefile. Recipes that only call `uv run ...` behave the same under sh (CI, Docker, Git Bash) and cmd.exe, so there's no per-OS branching.
- **Trade-off:** Windows users have to install make once (`winget install ezwinports.make`).

### D-008 Every §2 file exists from V0; unimplemented modules are one-line stubs

- **Alternatives:** create each file only in the version that implements it.
- **Why:** the tree matches the brief from day one. Each stub's docstring states its responsibility and target version, so the layout can be reviewed now.
- **Trade-off:** CLI commands for later versions are placeholders that exit with code 2.

### D-009 Inventory fully decodes every image (`verify()` then `load()`)

- **Alternatives:** `verify()` only (faster); extension-based checks.
- **Why:** `verify()` doesn't decode pixel data, so it misses truncated JPEGs. A full decode is the only reliable corruption check. The format is read from file contents, so mismatches with the extension are reported.
- **Trade-off:** the scan is about 2x slower, which is fine for a one-off inventory of a few thousand images.

### D-010 Repo root is the workspace folder, not `defect-detection/`

- **Alternatives:** a nested `defect-detection/` folder inside the workspace.
- **Why:** an extra nesting level adds nothing. The package and CLI are still named `defect-detection`.
- **Trade-off:** the folder name differs from §2. This is cosmetic only.

## V1 — Manifest, dedupe, split, leakage audit

### D-011 Near-duplicates: 64-bit pHash, Hamming ≤ 4, minimised over 8 rotations/flips — SUPERSEDED by D-031

- **Alternatives:** exact SHA-256 only; aHash/dHash; CNN-embedding similarity; threshold 8–10.
- **Why:** pHash (DCT of a 32×32 grayscale) ignores re-encoding, resizing and small brightness changes. Hashing all 8 dihedral variants catches copies that were rotated or flipped, which is how offline augmentation usually works. A threshold of 4 bits (≥ 94% of bits equal) is strict, because every image of the same part type looks alike.
- **Trade-off:** it can miss crops or larger edits. The audit also reports pairs within 8 bits so the threshold's sensitivity is visible.

### D-012 Exact duplicates collapse to one copy; copies with conflicting labels are all excluded

- **Alternatives:** keep every copy; keep one copy with the majority label.
- **Why:** byte-identical copies add no information and inflate counts and metrics. When the copies carry different labels, the true label can't be known. Every exclusion stays in `splits.csv` with a reason.
- **Trade-off:** a few images are lost. Exclusions are counted in every report.

### D-013 Official test set kept; train images that duplicate a test image are dropped

- **Alternatives:** re-split everything 70/15/15; drop the test copy instead.
- **Why:** keeping the published test set intact keeps results comparable with other work on the same data. Removing the train-side copies is what removes the leak. Validation is carved from the official train set (15%).
- **Trade-off:** train shrinks a little, and val/test proportions follow the dataset rather than 70/15/15. `split.strategy: random` switches to a full re-split.

### D-014 Group-aware stratified allocation written in-house (not `StratifiedGroupKFold`)

- **Alternatives:** sklearn `StratifiedGroupKFold` / `GroupShuffleSplit`.
- **Why:** three-way ratios can't be expressed directly with K-fold. The in-house allocation is about 15 lines: per label, shuffle clusters with the seed and fill splits in order up to their image share. It's deterministic, hits the ratios within 2% (tested), and is easy to explain.
- **Trade-off:** clusters get a single (majority) label, so a mixed-label cluster is stratified approximately. Mixed-label clusters are listed in the report.

### D-015 Audit re-derives leakage from hashes instead of trusting cluster ids

- **Alternatives:** check only that no cluster id spans two splits.
- **Why:** an independent check catches bugs in the clustering itself. A test plants a rotated test image in train under a fresh cluster id: the cluster check passes but the pHash check fails.
- **Trade-off:** an extra O(n²) pass, about 5 s at 7.5k images.

### D-016 `config.py` / `provenance.py` at package root; `data/eda.py` added

- **Alternatives:** parse YAML in each command; put EDA code in the notebook.
- **Why:** a single pydantic schema rejects unknown keys and invalid ratios at load time. Every report records the git commit (with `-dirty` if there are uncommitted changes) and the config hash. EDA logic lives in a tested module that can't load test rows, so the notebook stays thin. These three files are not in §2.
- **Trade-off:** more modules, but each has one job.

## V1 revision — adapting to the real dataset (Kaggle casting, both releases)

### D-031 Near-duplicates by rotation/flip-aligned image correlation, not pHash

- **Alternatives:** keep pHash with a lower threshold; CNN-embedding cosine similarity; SSIM.
- **Why:** on the real data, pHash couldn't separate copies from look-alikes. 95% of train images had another image within 4 bits, the distance histograms had no gap, and pairs at distance 0 included *different* parts with *different* labels (checked visually). Instead I use the Pearson correlation of standardised grayscale images, maximised over the 8 rotations/flips: a 32px pass shortlists candidates and a 128px pass re-scores them, where the nicks and burrs that identify a part are visible. It's plain NumPy matrix maths, with no model needed to decide what counts as a duplicate.
- **Trade-off:** small arbitrary rotations reduce the correlation, so some same-part copies fall below the threshold. That residual is measured (audit info threshold, V4 buckets), not hidden.

### D-032 Threshold 0.99, set by visual verification; shortlist k = 30

- **Alternatives:** 0.97 (catches more copies); 0.995; choosing by cluster statistics alone.
- **Why:** test→train pairs were rendered at 0.9999, 0.999, 0.998, 0.995, 0.99, 0.98, 0.97, 0.955 and 0.94. Down to about 0.97 the *defective* pairs were clearly the same part (matching burrs). But normal images drop out much faster as the threshold loosens, because distinct good parts look alike, so below about 0.98 the score also merges different normal parts. 0.99 is the point where matches were reliably the same part. The shortlist was raised from 10 to 30 after the audit found a link that clustering had missed with k = 10.
- **Trade-off:** same-part copies between 0.95 and 0.99 remain (the audit reports 483 / 715 test images with a train match ≥ 0.97). V4 reports metrics by similarity bucket to show the effect.

### D-033 Clean evaluation sets by direct match; group the training pool by transitive clusters

- **Alternatives:** exclude every train image in the same transitive cluster as a test image (the first implementation).
- **Why:** transitive chains over look-alike normals joined different parts. The largest clusters (209 and 198 images) were all normal, and random member pairs had a median similarity of only 0.89. Excluding whole chains removed 1,058 train images, about 1,000 of them normal. Direct matching (a train image is dropped only if *its own* best similarity to a test image is ≥ 0.99) removes 524. Inside the training pool, chains are kept as groups: that only forces them into one split (train or val), which is the safe direction.
- **Trade-off:** a train image linked to a test image only through an intermediate image survives if its own similarity is < 0.99. That's the residual similarity measured above.

### D-034 `casting_512x512` held out as an external test set

- **Alternatives:** ignore it; add it to training.
- **Why:** it is a separate capture: its median nearest-train similarity is 0.880, against 0.978 for the official test. No image had a ≥ 0.99 match in train or val. It's the best available estimate of performance on genuinely unseen parts and a different capture setup. Images are resized to 224 like everything else.
- **Trade-off:** 1,300 images that could have been training data, and a second number to explain. I think it's worth it: the official test mostly measures recognition of near-copies.

### D-035 Lighting shortcut: measured, mitigated by augmentation, checked in V4

- **Finding (train+val only):** mean brightness alone separates the classes with AUC 0.883 (defective 139 vs normal 150 grey levels, Cohen's d = −1.77); contrast alone reaches 0.812. Most likely the two classes were captured in different sessions.
- **Why no extra change now:** the ±20% brightness/contrast jitter (D-019) is larger than the ~8% class gap, so in training brightness can't be relied on as a cue. Per-image standardisation would remove the cue completely but changes the shared preprocessing. It stays the first remedy if V4 shows the model relies on brightness.
- **Check in V4:** error rates against brightness, and the external test set, where the shortcut may not hold.

## V2 — Shared preprocessing & training

### D-017 Preprocessing: EXIF-upright → RGB → direct bilinear resize to 224 → ImageNet normalisation

- **Alternatives:** resize the shorter side then centre-crop (timm's default eval transform); normalise with dataset statistics.
- **Why:** a centre crop can cut off a defect on the rim. Casting images are square, so a direct resize doesn't distort them. ImageNet mean/std match the pretrained weights, and they aren't fitted to our data, so no statistics leak from any split (§3.5). Applying EXIF orientation makes phone uploads in serving match what training saw.
- **Trade-off:** non-square uploads get stretched. Serving will enforce aspect-ratio limits (V6).

### D-018 Augmentation inserted between the core steps instead of reimplementing preprocessing

- **Alternatives:** a separate torchvision pipeline for training (`Resize` + `ToTensor` + `Normalize`).
- **Why:** training calls the same `prepare_image` / `to_float_array` / `normalize` functions as serving, with augmentations in between. A test proves that with augmentation switched off the output equals `core.preprocess`, and the parity test proves the dataset and the serving decode path are bit-identical.
- **Trade-off:** the noise step is custom code (about 3 lines) rather than a library transform.

### D-019 Mild augmentation only; no random crop or cutout

- **Alternatives:** RandAugment/TrivialAugment, RandomResizedCrop, cutout/random erasing.
- **Why:** defects are often a few pixels (pinholes, burrs). A crop or erase can delete the only defect and turn a "defective" image into a normal-looking one, which is label noise. ±15° rotation, flips, brightness/contrast ±20%, light blur/noise and ±5% shift/scale match plausible camera variation on a fixed rig.
- **Trade-off:** less regularisation than aggressive policies. If V2 shows overfitting, augmentation is the first thing to tune.

### D-020 EfficientNet-B0 (timm, ImageNet `ra_in1k`) with a 2-logit softmax head

- **Alternatives:** ResNet-18 (simpler, slower per FLOP), ConvNeXt-Tiny (7x the parameters), a single-logit sigmoid head.
- **Why:** 4.0 M parameters, strong ImageNet transfer, small ONNX file and fast CPU inference for serving. Two logits match "class-weighted cross-entropy" literally and export cleanly; P(defective) is the softmax of column 1, as fixed in `core/constants.py`.
- **Trade-off:** depthwise convolutions train slowly on CPU (214 ms/img measured for a full fine-tune step).

### D-021 Two-stage fine-tuning; BatchNorm frozen with the backbone in stage 1

- **Alternatives:** single-stage full fine-tuning; layer-wise LR decay.
- **Why:** a randomly initialised head sends large, noisy gradients into pretrained features. Training the head alone first (LR 1e-3, 3 epochs), then everything (LR 1e-4, cosine), avoids destroying them. In stage 1 the backbone runs in eval mode so BatchNorm keeps its ImageNet statistics.
- **Trade-off:** one more hyperparameter pair (head epochs and LR).

### D-022 Class-weighted cross-entropy (weights from train counts only)

- **Alternatives:** `WeightedRandomSampler`; focal loss.
- **Why:** it's a single, transparent change to the loss: `n_total / (2 · n_class)` computed on train only (§3.5). The sampler repeats minority images (more overfitting to duplicates) and changes what an epoch means. Focal loss adds a γ to tune and mainly helps extreme imbalance, which casting data doesn't have (the published release is about 57% defective; to be confirmed by `make data`).
- **Trade-off:** weighting distorts the predicted probabilities. V3's temperature scaling and threshold selection on val correct for this.

### D-023 Model selection: max val PR-AUC, ties broken by lower val loss; early stopping only in stage 2

- **Alternatives:** val loss alone; val accuracy; F1@0.5.
- **Why:** PR-AUC is threshold-free and focused on the defective class, which suits a threshold chosen later (V3). Casting data is easy enough that PR-AUC can hit 1.0 on several epochs, so validation cross-entropy breaks the tie. The un-augmented train metrics of the chosen checkpoint are reported next to val to check for overfitting.
- **Trade-off:** each epoch costs one extra pass over val.

### D-024 Checkpoints: `torch.save` of a state dict + JSON metadata, loaded with `weights_only=True`

- **Alternatives:** pickling the full model; safetensors plus a sidecar JSON.
- **Why:** `weights_only=True` refuses to unpickle arbitrary objects, so a tampered checkpoint can't execute code. Keeping metadata as plain JSON (config, config hash, git commit, epoch, val metrics) keeps it to one self-describing file.
- **Trade-off:** torch-specific. Serving never loads it anyway: it uses ONNX (V5).

## V3 — Calibration & operating point (validation only)

### D-025 Temperature scaling (one parameter) rather than Platt or isotonic calibration

- **Alternatives:** Platt scaling (2 parameters), isotonic regression, no calibration.
- **Why:** dividing logits by a single T can't change the ranking or the argmax, so PR-AUC and ROC-AUC are untouched and only the probabilities move. One parameter fit on about 1k val images can barely overfit. It's needed because class-weighted training deliberately skews the probabilities (D-022), and the review band and the API's `confidence` field need probabilities that mean what they say.
- **Trade-off:** it can't fix a shifted base rate (that would need Platt's bias term). Calibration is measured, so a poor fit would show up in the reliability diagram.

### D-026 Guards on the fit: separable validation keeps T = 1; a fit on its search bound is flagged

- **Alternatives:** always use the fitted value.
- **Why:** if every val defect outscores every normal, NLL keeps decreasing as T → 0, which would make the model arbitrarily overconfident. With zero errors, the data can't support sharpening. A fit stuck at the [0.05, 20] bound means the model carries little signal or val is too small; the report says so instead of hiding it (seen in the synthetic smoke run).
- **Trade-off:** on a separable val set the probabilities stay as trained, i.e. uncalibrated; this is reported.

### D-027 Calibration measured on P(defective) with 15 equal-width bins

- **Alternatives:** top-label ECE (confidence of the predicted class); adaptive (equal-mass) bins.
- **Why:** the threshold and the review band act on P(defective), so that is the quantity whose reliability matters. Equal-width bins match the standard reliability diagram and are easy to read in an interview.
- **Trade-off:** sparse middle bins are noisy on an easy dataset. Bin counts are kept so sparse bins can be discounted.

### D-028 Threshold: best precision subject to val defect recall ≥ 0.99; ties go to the lowest threshold

- **Alternatives:** 0.5; the F1-optimal threshold; minimising expected cost.
- **Why:** recall is the business constraint, since a shipped defect costs far more than a re-inspected good part. Among thresholds meeting it, precision (fewer false alarms) is maximised. When val is nearly separable many thresholds tie; the lowest catches the most defects at no val precision cost. The 0.5 and F1-optimal thresholds are reported for comparison.
- **Trade-off:** "recall ≥ 0.99 on val" is a point estimate (it allows 1 miss per 100 val defects); the test-set result in V4 is the honest check.

### D-029 Review band [low, high) around the threshold

- **Alternatives:** a fixed ±0.1 band; flagging by entropy; no band.
- **Why:** both edges come from business targets, not magic numbers. Below `low`, ≥ 99.9% of val defects would still be flagged. At or above `high`, flagged images are defective with ≥ 99.5% precision. `predicted_class` always uses the threshold; the band only adds `needs_review`. The report shows how many images go to review and how many of the threshold's errors the band catches.
- **Trade-off:** on a near-perfect val set the band can collapse to almost nothing. That is reported rather than padded artificially.

### D-030 Temperature and threshold fit on the same val set; costs are illustrative

- **Alternatives:** split val into a calibration half and a threshold half.
- **Why:** halving an already small val set would make both estimates noisier. Temperature scaling preserves ranking, so threshold selection isn't biased by it beyond the shared sample. The 20 : 1 : 0.25 costs (missed defect : false alarm : review) are labelled illustrative and live in config; they only populate a comparison table.
- **Trade-off:** val metrics at the chosen threshold are optimistic. Test (V4) gives the unbiased estimate.

## V4 — One-shot test evaluation & error analysis

### D-036 "Evaluate once" enforced in code, not by discipline

- **Alternatives:** a convention in the README; re-running is fine because nothing is tuned.
- **Why:** a second look at test is how test results quietly turn into model selection. `evaluate` refuses if the run has a `TEST_EVALUATED.json` marker, if the leakage audit didn't pass, or if the test/external fingerprints recomputed from `splits.csv` differ from the audited ones. Temperature, threshold and band are read from the val-only candidate meta. The PatchCore baseline has the same once-only guard.
- **Trade-off:** fixing a reporting bug afterwards means deleting the marker by hand, which is visible and deliberate.

### D-037 Two test sets, reported side by side, plus metrics by similarity bucket

- **Alternatives:** report only the official test set.
- **Why:** the official test still has a median nearest-train similarity of 0.978 after cleaning, while the 512 external set has 0.880 (D-034). Reporting both, and bucketing errors by each image's similarity to its nearest train image, separates "recognises near-copies" from "generalises to new parts".
- **Trade-off:** two sets of numbers to explain, but each answers a different question.

### D-038 95% percentile bootstrap CIs (1,000 resamples, seed 42)

- **Alternatives:** Wilson intervals per proportion; DeLong for AUC.
- **Why:** one method covers every metric (AUCs, precision, recall, F1, macro-F1) without separate formulas, and needs no normality assumption. Resamples with a single class are skipped and counted.
- **Trade-off:** it ignores the grouped structure (same-part copies inside a test set), so the intervals are somewhat too narrow. With near-perfect scores, percentile intervals also collapse towards the point estimate; the error counts are reported next to them.

### D-039 Grad-CAM via timm's `forward_features` / `forward_head`, no hooks

- **Alternatives:** the `pytorch-grad-cam` package; forward/backward hooks on a named layer.
- **Why:** about 15 lines, no extra dependency, and it works for every timm backbone because they all expose the same two-step forward. The gradient of the defect logit with respect to the last feature map gives a weight per channel.
- **Trade-off:** 7×7 resolution for EfficientNet at 224px, so heatmaps are coarse. They're good enough to see whether the model looks at the part or at the background.

### D-040 PatchCore baseline written in-repo, with documented simplifications

- **Alternatives:** anomalib (pulls in Lightning and a large dependency tree); skip the baseline.
- **Why:** the method is short: pretrained mid-level patch features, a memory bank of normal patches selected with greedy k-center, and an image score equal to its worst patch. It is fit on NORMAL train images only and its threshold is chosen on val with the classifier's recall policy, so the comparison is fair. Simplifications: random 64 patches per image before the coreset, a 5,000-patch bank, and no score reweighting.
- **Trade-off:** probably a few points below a tuned anomalib PatchCore. It is a baseline to argue from, not a competitor to optimise.

## V5 — ONNX export, verification & benchmark

### D-041 TorchScript ONNX exporter (opset 17, dynamic batch); the graph outputs logits

- **Alternatives:** the dynamo-based exporter (torch's default since 2.9); baking softmax and temperature into the graph.
- **Why:** the dynamo exporter needs the extra `onnxscript` package, and the TorchScript exporter is mature for plain CNNs; it exported this model first time. Outputting logits keeps temperature, threshold and review band in `model_meta.json`, so recalibrating never means re-exporting.
- **Trade-off:** torch flags the TorchScript exporter as deprecated ("will be removed"). A future torch upgrade means switching to `dynamo=True` and adding `onnxscript`; the parity test will catch any behaviour change.

### D-042 Model registry: immutable `models/<version>/` with SHA256SUMS; export fails closed

- **Alternatives:** overwrite `models/latest/`; an MLflow registry.
- **Why:** the version is the run name (`<timestamp>_<config-hash>`), so every artifact traces back to its config and commit. Export refuses an existing version and deletes everything it wrote if `onnx.checker` or the parity check fails. Parity: max |logit difference| ≤ 1e-4 on 32 VAL images at batch sizes 1, 7 and 32 (7 checks the dynamic axis with a size the tracer never saw). `SHA256SUMS` uses the `sha256sum -c` format and rejects path-traversal names, so serving can verify integrity before loading anything.
- **Trade-off:** rollback means pointing serving at the previous folder (RUNBOOK, V8) rather than a registry UI.

### D-043 Benchmark method; INT8 dynamic quantisation is an experiment, not shipped by default

- **Alternatives:** time only ONNX; ship INT8 if it is faster.
- **Why:** same real val images, same thread count (4, the physical cores here) for PyTorch and ONNX Runtime, warm-up discarded, p50/p95/p99 over 100 calls, throughput from the median. INT8 is written next to the run, never into `models/`. Its effect on decisions at the operating threshold is measured on the full val split, because a faster model that flips a defect decision isn't an improvement.
- **Trade-off:** dynamic quantisation mainly helps MatMul-heavy models. On a depthwise-conv CNN it may give little speed-up; the table reports whatever it measures.
