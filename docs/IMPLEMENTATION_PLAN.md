# Implementation plan

## Repository audit

The starter has a small FastAPI service, a Streamlit dashboard, basic CSV loading, a single XGBoost pipeline, an Isolation Forest, a SQLite event table, and two minimal tests. `data/raw/fraud_oracle.csv` is present locally but is ignored by Git. Its sampled header contains 33 columns including target `FraudFound_P`. The starter has no Git metadata at this workspace root, so Git status and tracked-file checks are unavailable here.

Verified gaps from the source review:

- Target handling covers only a narrow set of labels, and the loader has no explicit duplicate-column, bad-schema, or size checks.
- There is no leakage review. In particular, the dataset includes `PolicyNumber` (identifier-like) and claimed/accident timing or investigation-adjacent fields that need documented review before use.
- Classifier preprocessing is inside the pipeline, but the separate anomaly preprocessing is fit on all rows before any holdout isolation. No baseline, curves, threshold analysis, training manifest, model/schema validation, or explanation mapping exists.
- The scoring service does not validate missing/extra inference fields and does not calculate SHAP. Its anomaly score is separate from fraud probability, which should be retained.
- API is missing request IDs, batch scoring, summary/global explanation routes, and bounded validated schemas. Persistence omits threshold, model version, separate anomaly flag, and explicit explanation; the API error path risks returning unsafe exception detail.
- The dashboard has four incomplete pages, loads data globally without caching/status detail, lacks explorer/audit/settings, and does not show full model evaluation or explanations.
- Documentation lacks architecture, feature dictionary, model card, privacy/security details, CI, and lint configuration. Existing tests cover only basic target normalization and health.

## Stages

1. **Data and provenance** — strengthen target and schema validation, leakage-aware exclusions, preserve source data, add dataset fingerprint/provenance guidance, and create offline fixture tests. Document source terms as requiring verification by the downloader; do not redistribute the local CSV.
2. **Modeling and artifacts** — stratified fixed-seed split; fit classifier/anomaly transformations on training rows only; evaluate XGBoost and logistic baseline with threshold metrics, PR/ROC data and confusion matrix; persist a reproducibility manifest and schema; validate inference contracts; implement correctly mapped SHAP summaries/local contributions with an explicit unavailable state.
3. **API and audit** — typed request/response models, request ID middleware and structured logs, bounded batch endpoint, analytics/model/global explanation routes, safe consistent errors, and parameterized SQLite audit data including model and threshold fields. Keep narratives isolated, minimized, timed out, and locally available.
4. **Responsible-use documentation and quality** — architecture, data dictionary, model card, security/privacy, cross-platform setup, Ruff, and CI. Add offline tests for the principal loader/model/API/storage/narrative flows.
5. **Streamlit product UI** — implement all eight requested views with a coherent responsive light theme, real-data-only charts, useful empty/error states, export and human-review language; preserve Streamlit.
6. **Verification** — run focused tests at the end of each implementation stage and the full offline test/lint suite; inspect available public-data model readiness without inventing evaluation results. Report unavailable data or environment constraints plainly.

## Working constraints and assumptions

- The local CSV's presence does not establish redistribution rights or a verified licence. Treat its origin/terms as user-provided local data and do not copy it into source control or new artifacts.
- Exclusions will be explicit, conservative, and documented. Sensitive/protected features will not be used for adverse decisions. Dataset-specific meanings and intake availability need confirmation from authoritative dataset documentation; ambiguous fields should be excluded until confirmed.
- Synthetic or hand-built rows are test fixtures only. Public-data model metrics are reported only when training actually completes against the downloaded dataset.
- This is a local-first decision-support tool. It cannot make claim denial decisions and requires qualified human review, appeal/override procedures, and monitoring before operational use.

## Delivered and verified

- Implemented all six stages above: loader/model, API/audit/narrative, documentation/CI, and eight-page Streamlit UI.
- Trained against the locally present CSV and wrote model artifacts/manifests to ignored `models/`. The local file's fingerprint and label counts are in `DATA_DICTIONARY.md`; artifacts are not committed.
- Verified native TreeSHAP global and local contribution paths, transformed-schema mapping, all API routes including single/batch scoring and audit persistence, and each dashboard navigation page.
- Latest offline suite: `python -m pytest -q` (13 tests passed). `python -m compileall -q src dashboard tests` passed.
- Ruff is configured in `pyproject.toml` and CI, but local lint execution was unavailable because Ruff was not installed and package network access was blocked in the execution environment.
