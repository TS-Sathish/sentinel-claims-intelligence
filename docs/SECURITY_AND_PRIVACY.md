# Security, privacy, and responsible use

## Intended use and human review

Sentinel provides preliminary review-priority signals only. A qualified person must review primary evidence, apply policy terms, and make any decision. Do not use the score alone to deny, delay, price, or limit coverage. Provide a clear route for claimants to correct information, request human reconsideration, and appeal. Record reviewer overrides and reasons in a governed production system.

## Data handling

- Keep `.env`, raw data, SQLite databases, and model artifacts out of Git. `.gitignore` excludes these local files.
- Use the minimum fields required for a documented purpose. The local benchmark excludes explicit sensitive attributes (including age and marital status), address-related features, and selected process/outcome fields; remaining proxy and label bias risks persist.
- Do not send claim identifiers or excluded personal attributes to an LLM. Optional narrative calls transmit only an allowlisted set of generic descriptors and score/anomaly outputs. Disable the provider for sensitive workflows; review provider terms and retention separately.
- API keys are read from environment variables. Never place secrets in source files, UI, logs, or requests.
- Local SQLite stores only a one-way SHA-256 digest of an optional caller reference, score and anomaly results, threshold, model version, narrative, explanation, and UTC timestamp. Apply operating-system access controls and define retention/deletion practices for any real use.
- CSV input is not executed. This starter does not accept file uploads. If upload support is added, enforce size/type/schema limits and treat all content as untrusted.

## Fairness, labels, and limitations

Historical fraud labels can reflect investigation practices, access to evidence, policyholder behavior, and prior human decisions. They are not guaranteed ground truth. Removing explicit protected fields does not remove proxies. Before any operational use, conduct jurisdiction-specific legal and fairness review, assess false-positive and false-negative rates across relevant groups where lawful, validate representativeness, and monitor drift.

## Operational safeguards

- Bind the local API to loopback (`127.0.0.1`) by default. No permissive CORS is configured.
- Use a trusted environment and trusted local artifacts only; joblib artifacts are executable serialization and must not be loaded from unknown sources.
- Keep API reachable only to authorized local users; add authentication, transport security, rate limiting, retention controls, and security review before network exposure.
- Treat narratives as untrusted display text. Do not render them as unsafe HTML or use them as evidence.
- Maintain a documented human override, correction, appeal, and incident response process before real claim use.
