# Repository requirements

- Commit author and committer must be `AsjalAbdullahButt
  <asjal.abdullah.butt@gmail.com>`. Do not add co-author trailers or assistant
  attribution to commits. Do not rewrite or force-push history without explicit approval.
- Before pushing, run lint, formatting, backend and UI type checks, the full test
  suite, the core/serving coverage gate, Bandit, dependency audits for every
  lockfile, and a secret scan of the proposed files and outgoing history.
- Push only after available local checks pass, then monitor GitHub CI until all
  jobs finish. The owner explicitly approved this sequence because GitHub CI
  requires an uploaded commit and this Windows machine has no Docker.
- Keep credentials, real environment files, Streamlit secrets, raw/processed
  datasets, model weights, runs, private reports and local sample images out of
  Git. Verify the staged file list; `.gitignore` does not protect tracked files.
- Preserve the CI quality, security and container checks. Fix failures without
  weakening gates or suppressing findings merely to get a green result.
