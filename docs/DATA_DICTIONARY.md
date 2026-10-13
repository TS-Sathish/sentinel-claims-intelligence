# Data dictionary and provenance

## Provenance and fingerprint

The intended primary source is Kaggle's [Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection), commonly distributed as `fraud_oracle.csv`. An alternative source is Kaggle's [Automobile Insurance Fraud Detection](https://www.kaggle.com/datasets/aashishjhamtani/automobile-insurance-fraud-detection). A reference implementation/data repository is [mwitiderrick/insurancedata](https://github.com/mwitiderrick/insurancedata).

The CSV found in this workspace was inspected locally; origin is not cryptographically or contractually verified. Observed fingerprint: 15,420 rows, 33 columns, SHA-256 `8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6`. Observed `FraudFound_P` counts: 14,497 zeros and 923 ones (5.99% positive). It contains no blank cells or duplicate rows on this file. Recompute fingerprints with training; these values describe only the local file at inspection time.

**Licence/terms:** the Kaggle dataset page and any upstream source terms must be reviewed by the downloader. The licence was not verified from this local file. Sentinel does not bundle or redistribute it. Download manually through Kaggle's authorized workflow, review the displayed licence and terms, then place the CSV under `data/raw/`. Optional CLI after authorized Kaggle setup: `kaggle datasets download -d shivamb/vehicle-claim-fraud-detection -p data/raw --unzip`. Do not use the command to evade access controls or terms.

## Target and known limitations

`FraudFound_P` is conventionally interpreted as 1 = a fraud label found and 0 = no fraud label found. This is a historical dataset label, not independently verified ground truth. The 5.99% share is a property of this benchmark sample, not an estimate of any insurer's prevalence. The dataset is old, automotive-only, and has unknown collection, adjudication, and sampling mechanisms. It may contain process and historical bias and is not representative evidence of present-day performance.

## Columns

The file has 33 total fields. `FraudFound_P` is the target; all other listed entries are source columns. Concise semantic notes below use the column names and conventional interpretation. Exact units, timing, and category coding should be confirmed against upstream metadata before operational use.

| Column | Type in local CSV | Meaning / treatment |
|---|---|---|
| Month | category | Accident month label; candidate predictor after timing review. |
| WeekOfMonth | integer | Week position in a month. |
| DayOfWeek | category | Accident weekday. |
| Make | category | Vehicle make. |
| AccidentArea | category | Accident area classification. |
| DayOfWeekClaimed | category | Claim weekday; claim process timing. |
| MonthClaimed | category | Claim month; claim process timing. |
| WeekOfMonthClaimed | integer | Claim week position. |
| Sex | category | Sensitive personal attribute; excluded from scoring. |
| MaritalStatus | category | Personal attribute; excluded conservatively from scoring. |
| Age | integer | Sensitive personal attribute; excluded from scoring. |
| Fault | category | Liability determination; excluded because post-incident availability is unverified. |
| PolicyType | category | Policy type / vehicle category combination. |
| VehicleCategory | category | Broad vehicle category. |
| VehiclePrice | category | Vehicle price band. |
| FraudFound_P | integer | Target; 1 fraud label, 0 no fraud label. |
| PolicyNumber | integer | Masked policy identifier; excluded. |
| RepNumber | integer | Representative/adjuster code; excluded due to process leakage risk. |
| Deductible | integer | Policy deductible amount. |
| DriverRating | integer | Ordinal rating code; semantics require upstream confirmation. |
| Days_Policy_Accident | category | Time relationship of policy to accident; exact coding requires upstream metadata. |
| Days_Policy_Claim | category | Time relationship of policy to claim; exact coding requires upstream metadata. |
| PastNumberOfClaims | category | Prior-claim count band. |
| AgeOfVehicle | category | Vehicle age band. |
| AgeOfPolicyHolder | category | Age band; sensitive attribute; excluded. |
| PoliceReportFiled | category | Report filed indicator; excluded pending intake-availability review. |
| WitnessPresent | category | Witness indicator; excluded pending intake-availability review. |
| AgentType | category | Agent classification; excluded because operational assignment availability is unverified. |
| NumberOfSuppliments | category | Supplement count band; excluded because it may be post-intake. |
| AddressChange_Claim | category | Address-change timing band relative to claim; excluded as an address proxy with uncertain timing. |
| NumberOfCars | category | Number of cars band. |
| Year | integer | Dataset year field; context may be temporal. |
| BasePolicy | category | Broad policy coverage type. |

## Exclusions

`src/data/loader.py` applies these conservative exclusions before training and scoring:

- `FraudFound_P` and recognized configured target: target leakage by definition.
- Policy/claim IDs, `PolicyNumber`, generic row indexes: identifiers/order artifacts.
- `RepNumber`: operational assignment/process proxy with risk of encoding investigation workflow.
- `Fault`, `PoliceReportFiled`, `WitnessPresent`: post-incident outcome/evidence timing is not established for intake use.
- `Sex`, `Gender`, `Age`, `AgeOfPolicyHolder`: sensitive personal attributes excluded from adverse scoring.
- `MaritalStatus`: personal attribute excluded conservatively.
- `AgentType`: operational assignment/process proxy excluded because intake availability is unverified.
- `NumberOfSuppliments`, `AddressChange_Claim`: potential post-intake or address-related features excluded pending data-timing/fairness review.
- Recognized fraud decisions, investigation results, and settlement amount: direct target proxies or post-adjudication outcomes.
- Identifier-like fields with a unique value per row (such as `*_id`, `*_key`, `*_reference`, or `*number`): dynamic identifier exclusion.

The remaining benchmark fields still require a deployment-specific availability and fairness review. Excluding sensitive fields from model inputs does not guarantee fairness: proxy variables and historical label bias remain possible.
