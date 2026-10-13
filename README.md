# Sentinel Claims Intelligence

Local-first claim risk triage with XGBoost, a separate Isolation Forest anomaly signal, real SHAP explanations, FastAPI, Streamlit, and SQLite audit history. Risk indicators support human review; they do not establish fraud or make claim decisions.

## Results (held-out 20% test split)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.123 | 0.762 | 0.211 | 0.172 | 0.764 |
| Logistic regression baseline | 0.096 | 0.773 | 0.171 | 0.117 | 0.709 |

15,420 vehicle insurance claims with a 6.0% fraud label rate. XGBoost improves PR-AUC by about 47% over the baseline. Precision is low at the 0.50 threshold because fraud is rare, which is why the tool ranks claims for human review rather than deciding them. Age, sex and marital status are excluded from scoring, and `Fault` is excluded because it may be set after intake.

## Dataset setup

Primary source: [Vehicle Insurance Claim Fraud Detection on Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection), commonly `fraud_oracle.csv`, target `FraudFound_P`. Alternatives and the locally observed dataset fingerprint are documented in [Data dictionary](docs/DATA_DICTIONARY.md). Review the dataset's current licence and source terms before use. This repository does not redistribute the CSV and does not bypass Kaggle access controls.

Download through Kaggle's authorized UI, then place the CSV at `data/raw/fraud_oracle.csv`, or set `DATASET_PATH` to another authorized CSV. Optional Kaggle CLI after account setup:

```shell
kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p data/raw --unzip
```

Configure `.env` from `.env.example`; set `TARGET_COLUMN` only when automatic target detection is inappropriate. Target labels must cleanly map to both classes. Raw files remain unchanged; all model artifacts are separate under ignored `models/`.

## Install and run

Python 3.11 is recommended.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.training.train
```

### macOS / Linux

```shell
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m src.training.train
```

Run the local API and dashboard in separate terminals:

```shell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501 · API docs: http://127.0.0.1:8000/docs

Run software tests offline (no public CSV or API key needed):

```shell
python -m pytest -q
```

## Pages

Executive Overview, Claims Explorer, Claim Investigation, Model Performance, Explainability, Data Quality, Audit History, and Settings. Missing dataset/model/artifacts produce explicit setup states. Metrics and charts only render from loaded data or trained artifacts.

## API routes

- `GET /health`
- `GET /api/v1/model/status`
- `GET /api/v1/analytics/summary`
- `POST /api/v1/claims/score`
- `POST /api/v1/claims/score-batch` (maximum 100 claims)
- `GET /api/v1/claims/recent`
- `GET /api/v1/explanations/global`

Scoring requires every feature in the model manifest with no extra fields. The response separates fraud probability from anomaly flag/score and includes threshold, model version, explanation availability, and cautious narrative. The score endpoint accepts scalar JSON feature values; it does not accept uploaded files.

## Optional narrative provider

Default `LLM_PROVIDER=local` works offline without a key. To enable the optional OpenAI-compatible provider, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL` in the local environment. Provider calls have timeouts and fall back to a local note. Only allowlisted generic descriptors and model outputs are sent; review provider terms and data policy before use.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data dictionary and provenance](docs/DATA_DICTIONARY.md)
- [Model card](docs/MODEL_CARD.md)
- [Security and privacy](docs/SECURITY_AND_PRIVACY.md)
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)

## Responsible use

This benchmark has historical labels, unknown sampling and adjudication bias, and a sample prevalence that may not match real portfolios. Do not use this prototype alone for adverse actions. Human evidence review, correction and appeal paths, fairness assessment, access controls, and production monitoring are necessary before real claim workflows.
