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

### D-011 Near-duplicates: 64-bit pHash, Hamming ≤ 4, minimised over 8 rotations/flips

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
