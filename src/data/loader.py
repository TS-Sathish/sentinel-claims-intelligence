import csv
import re
import pandas as pd
from pathlib import Path
from typing import Any

TARGETS = ["FraudFound_P", "fraud_reported", "fraud_label", "is_fraud", "fraud", "target", "label"]

# Conservative exclusions for the supplied vehicle-claim benchmark. The dataset
# does not define a production intake-time contract, so uncertain fields are
# omitted until availability and fairness have been reviewed.
EXCLUSION_RATIONALE = {
    "policynumber": "masked policy identifier; can encode row/order artifacts",
    "claimnumber": "claim identifier; not a claim characteristic",
    "claimid": "claim identifier; not a claim characteristic",
    "id": "generic record identifier",
    "repnumber": "adjuster/representative assignment may proxy process and encode leakage",
    "fault": "liability determination may be established after intake",
    "policereportfiled": "post-incident process/evidence field; intake availability is unverified",
    "witnesspresent": "post-incident evidence field; intake availability is unverified",
    "sex": "sensitive personal attribute; excluded from adverse scoring",
    "gender": "sensitive personal attribute; excluded from adverse scoring",
    "age": "sensitive personal attribute; excluded from adverse scoring",
    "ageofpolicyholder": "sensitive personal attribute; excluded from adverse scoring",
    "maritalstatus": "sensitive personal attribute; excluded from adverse scoring",
    "addresschangeclaim": "address-related proxy and timing is unverified; excluded conservatively",
    "numberofsupplements": "supplement count may be post-intake; availability is unverified",
    "numberofsupplement": "supplement count may be post-intake; availability is unverified",
    "numberofsuppliments": "supplement count may be post-intake; availability is unverified",
    "agenttype": "operational assignment/process field; intake availability is unverified",
    "fraudfound_p": "target",
    "fraudreported": "target or target proxy",
    "fraud_reported": "target or target proxy",
    "fraudlabel": "target or target proxy",
    "isfraud": "target or target proxy",
    "fraud": "target or target proxy",
    "frauddecision": "target proxy/post-investigation decision",
    "fraudconfirmed": "target proxy/post-investigation decision",
    "investigationresult": "post-investigation outcome",
    "settlementamount": "post-adjudication/settlement outcome",
}


def _key(column: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).strip().lower())


def excluded_columns(columns, target=None):
    """Return excluded source columns mapped to documented reasons."""
    result = {}
    target_key = _key(target) if target else None
    for column in columns:
        key = _key(column)
        if target_key and key == target_key:
            result[str(column)] = "configured prediction target"
        elif key.startswith("unnamed") or key in {"index", "rowid"}:
            result[str(column)] = "generated row index"
        elif key in EXCLUSION_RATIONALE:
            result[str(column)] = EXCLUSION_RATIONALE[key]
    return result

def find_target(frame, configured=None):
    if configured:
        if configured not in frame.columns:
            raise ValueError(f"Configured target '{configured}' is missing.")
        return configured
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for name in TARGETS:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError("Fraud target not detected. Set TARGET_COLUMN in .env. Columns: " + ", ".join(map(str, frame.columns)))

def normalize_target(series):
    mapping = {"1": 1, "1.0": 1, "y": 1, "yes": 1, "true": 1, "fraud": 1,
               "fraudulent": 1, "suspicious": 1, "positive": 1,
               "0": 0, "0.0": 0, "n": 0, "no": 0, "false": 0,
               "legitimate": 0, "non-fraud": 0, "nonfraud": 0, "not fraud": 0,
               "negative": 0, "genuine": 0}
    cleaned = series.astype("string").str.strip().str.lower()
    values = cleaned.map(mapping)
    if values.isna().any() or values.nunique() != 2:
        unexpected = cleaned[values.isna()].dropna().unique().tolist()[:8]
        raise ValueError(f"Target must contain both classes and map to 0/1. Unrecognized labels: {unexpected}")
    return values.astype("int8")

def load_dataset(path, target_column=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Download a public CSV; see README.md.")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            duplicates = sorted({name for name in header if header.count(name) > 1})
            raise ValueError(f"Dataset has duplicate column names: {duplicates}")
        frame = pd.read_csv(path, low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read CSV '{path.name}': {exc}") from exc
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Dataset has duplicate column names: {duplicates}")
    target = find_target(frame, target_column)
    y = normalize_target(frame[target])
    excluded = excluded_columns(frame.columns, target)
    for column in frame.columns:
        name = str(column).strip().lower()
        key = _key(column)
        identifier_name = (bool(re.search(r"(?:^|[^a-z])(id|identifier|key|reference)(?:$|[^a-z])", name))
                           or key.endswith("number"))
        if column != target and identifier_name and frame[column].nunique(dropna=False) == len(frame):
            excluded.setdefault(str(column), "unique identifier-like field")
    X = frame.drop(columns=list(excluded)).copy()
    if X.shape[1] == 0:
        raise ValueError("No usable features remain after target and leakage-risk exclusions.")
    X = X.replace(["?", "NA", "N/A", "null", "None", ""], pd.NA)
    if X.columns.astype(str).duplicated().any():
        raise ValueError("Feature column names must be unique after string conversion.")
    return X, y, frame, target
