# Model card

## Intended use

Research and local workflow prototyping for prioritizing human review of vehicle insurance claims. The score is not a fraud finding and must not decide claim denial, payment, pricing, investigation closure, or eligibility.

## Models and training

- Main classifier: XGBoost with median/mode imputation and one-hot encoding.
- Baseline: class-weighted Logistic Regression using the same training-only preprocessing pattern.
- Separate anomaly detector: Isolation Forest fitted only on training features. Its score is an anomaly signal, not a probability of fraud.
- Split: fixed-seed (42), stratified 80/20 train/test split.
- Imbalance handling: XGBoost `scale_pos_weight` from training counts and Logistic Regression `class_weight=balanced`; no pre-split resampling.
- Artifacts: local model files and `manifest.json`; the manifest records dataset SHA-256, timestamp, schema, target, split, exclusions, version, and metrics.

The raw training dataset may have label prevalence and sampling bias that differ substantially from deployment. Report precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, PR/ROC curves, and threshold tradeoffs. PR-AUC depends on prevalence. Choose thresholds with investigation capacity and false-positive costs; the configured threshold is a review setting, not a decision boundary for denial.

## Explanations

Local and global explanations use XGBoost's native `pred_contribs` TreeSHAP output from the fitted model in transformed feature space, with transformed names returned alongside values. If contribution generation or mapping fails, the application reports unavailable status and the reason; it never fabricates contributions. Explanations describe model behavior, not causal evidence or verified facts.

## Evaluation

Run `python -m src.training.train` on the authorized local dataset. The command prints observed metrics and writes them to `models/metrics.joblib` and `models/manifest.json`. No benchmark score is claimed in this document. The bundled test fixture is synthetic and used only for offline software-path tests.

## Known limitations and monitoring

Small historical automotive sample, unknown labeling and collection process, potential leakage, temporal drift, class imbalance, and uncertain feature availability limit transfer to production. Validate calibration and subgroup error rates using lawful, representative data; monitor false positives, false negatives, overrides, appeals, and drift. Reassess before changes in data source or workflow.
