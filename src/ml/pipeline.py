from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from xgboost import DMatrix

ARTIFACT_VERSION = "2"
MODEL_FILE = "xgboost_pipeline.joblib"
BASELINE_FILE = "logistic_baseline.joblib"
ANOMALY_FILE = "isolation_forest.joblib"
METRICS_FILE = "metrics.joblib"
MANIFEST_FILE = "manifest.json"


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                 ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _metrics(y_true, probability, threshold=.5):
    pred = (np.asarray(probability) >= threshold).astype(int)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probability)
    fpr, tpr, roc_thresholds = roc_curve(y_true, probability)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist(),
                     "thresholds": pr_thresholds.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                      "thresholds": roc_thresholds.tolist()},
        "threshold_analysis": [],
    }


def _model_metrics(y_true, probability):
    result = _metrics(y_true, probability)
    result["threshold_analysis"] = [
        {"threshold": threshold, **{key: _metrics(y_true, probability, threshold)[key]
                                    for key in ("precision", "recall", "f1")}}
        for threshold in (.1, .2, .35, .5, .65, .8)
    ]
    return result


def _dataset_fingerprint(X: pd.DataFrame, y: pd.Series, dataset_path=None) -> str:
    digest = hashlib.sha256()
    if dataset_path and Path(dataset_path).is_file():
        with open(dataset_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    else:
        digest.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
        digest.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return digest.hexdigest()


def train_models(X: pd.DataFrame, y: pd.Series, model_dir: Path, *, target: str | None = None,
                 dataset_path=None, exclusions: dict[str, str] | None = None):
    if len(X) < 50 or y.nunique() != 2:
        raise ValueError("Need at least 50 labelled rows and both target classes.")
    if list(X.columns).count(None) or X.columns.astype(str).duplicated().any():
        raise ValueError("Feature columns must have unique, non-empty names.")
    if X.isna().all(axis=0).any():
        empty = X.columns[X.isna().all(axis=0)].astype(str).tolist()
        raise ValueError(f"Features contain entirely missing columns: {empty}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y)
    class_counts = y_train.value_counts()
    imbalance_ratio = float(class_counts.min() / class_counts.max())
    pos_weight = float(class_counts.get(0, 1) / class_counts.get(1, 1))

    classifier = Pipeline([("prep", make_preprocessor(X_train)),
                           ("model", XGBClassifier(n_estimators=200, max_depth=4,
                                                    learning_rate=.05, subsample=.85,
                                                    colsample_bytree=.85, eval_metric="logloss",
                                                    scale_pos_weight=pos_weight,
                                                    random_state=42, n_jobs=4))])
    classifier.fit(X_train, y_train)
    xgb_prob = classifier.predict_proba(X_test)[:, 1]

    baseline = Pipeline([("prep", make_preprocessor(X_train)),
                         ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    baseline.fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]

    xgb_metrics = _model_metrics(y_test, xgb_prob)
    baseline_metrics = _model_metrics(y_test, baseline_prob)

    anomaly_preprocessor = make_preprocessor(X_train)
    train_transformed = anomaly_preprocessor.fit_transform(X_train)
    anomaly = IsolationForest(n_estimators=200, contamination="auto", random_state=42,
                              n_jobs=-1).fit(train_transformed)

    metrics = {
        "xgboost": xgb_metrics,
        "baseline": baseline_metrics,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "fraud_rate": float(y.mean()), "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()), "random_seed": 42,
        "split": {"method": "stratified train_test_split", "test_size": .2},
        "imbalance": {"positive_class_weight": pos_weight,
                      "train_min_to_max_class_ratio": imbalance_ratio,
                      "strategy": "XGBoost scale_pos_weight; logistic class_weight=balanced; no resampling"},
        # Top-level aliases keep earlier dashboard/client readers functional.
        "precision": xgb_metrics["precision"], "recall": xgb_metrics["recall"],
        "f1": xgb_metrics["f1"], "pr_auc": xgb_metrics["pr_auc"],
        "roc_auc": xgb_metrics["roc_auc"], "confusion_matrix": xgb_metrics["confusion_matrix"],
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / MODEL_FILE)
    joblib.dump(baseline, model_dir / BASELINE_FILE)
    joblib.dump({"preprocessor": anomaly_preprocessor, "model": anomaly,
                 "feature_columns": list(X.columns)}, model_dir / ANOMALY_FILE)
    joblib.dump(metrics, model_dir / METRICS_FILE)
    try:
        sample = classifier.named_steps["prep"].transform(X_test.iloc[:min(500, len(X_test))])
        shap_values = _native_shap_values(classifier.named_steps["model"], sample)
        joblib.dump({"available": True, "features": _feature_names(classifier.named_steps["prep"]),
                     "mean_abs_shap": np.abs(shap_values).mean(axis=0).astype(float).tolist()},
                    model_dir / "global_explanations.joblib")
    except Exception as exc:
        joblib.dump({"available": False, "reason": type(exc).__name__},
                    model_dir / "global_explanations.joblib")
    manifest = {
        "artifact_version": ARTIFACT_VERSION, "model_version": "sentinel-xgb-2",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": _dataset_fingerprint(X, y, dataset_path),
        "target": target, "feature_columns": [str(c) for c in X.columns],
        "feature_dtypes": {str(c): str(X[c].dtype) for c in X.columns},
        "excluded_columns": exclusions or {}, "split": metrics["split"],
        "random_seed": 42, "metrics": metrics,
    }
    (model_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def load_artifacts(model_dir):
    model_dir = Path(model_dir)
    required = [MODEL_FILE, BASELINE_FILE, ANOMALY_FILE, METRICS_FILE, MANIFEST_FILE]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Model artifacts unavailable (" + ", ".join(missing) + "). Run `python -m src.training.train`.")
    manifest = json.loads((model_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Model artifacts are from an unsupported version. Retrain with this application.")
    return (joblib.load(model_dir / MODEL_FILE), joblib.load(model_dir / ANOMALY_FILE),
            joblib.load(model_dir / METRICS_FILE), manifest)


def validate_schema(frame: pd.DataFrame, expected_columns: list[str], expected_dtypes=None):
    expected, actual = set(expected_columns), set(map(str, frame.columns))
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing features: " + ", ".join(missing))
        if extra:
            parts.append("unexpected features: " + ", ".join(extra))
        raise ValueError("Inference schema mismatch (" + "; ".join(parts) + "). Use the training feature schema.")
    result = frame.loc[:, expected_columns].copy()
    for column in expected_columns:
        expected_type = (expected_dtypes or {}).get(column, "")
        if expected_type.startswith(("int", "float", "uint", "bool")):
            converted = pd.to_numeric(result[column], errors="coerce")
            invalid = result[column].notna() & converted.isna()
            if invalid.any():
                raise ValueError(f"Feature '{column}' must be numeric; received an unparseable value.")
            result[column] = converted.astype(float)
        else:
            result[column] = result[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return result


def _feature_names(preprocessor):
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def explain_global(classifier, limit=25):
    try:
        raise ValueError("Global SHAP values must be loaded from the training artifact.")
    except Exception as exc:
        return {"available": False, "reason": f"Explanation unavailable: {type(exc).__name__}"}


def load_global_explanation(model_dir, limit=25):
    path = Path(model_dir) / "global_explanations.joblib"
    if not path.exists():
        return {"available": False, "reason": "Global SHAP artifact is unavailable. Retrain the model."}
    data = joblib.load(path)
    if not data.get("available"):
        return {"available": False, "reason": "SHAP unavailable during training: " + data.get("reason", "unknown error")}
    pairs = zip(data["features"], data["mean_abs_shap"])
    items = sorted(({"feature": name, "importance": float(value)} for name, value in pairs),
                   key=lambda item: item["importance"], reverse=True)
    return {"available": True, "method": "mean absolute SHAP value on held-out rows",
            "features": items[:limit]}


def explain_local(classifier, row: pd.DataFrame, max_features=12):
    try:
        prep, model = classifier.named_steps["prep"], classifier.named_steps["model"]
        transformed = prep.transform(row)
        names = _feature_names(prep)
        vals = _native_shap_values(model, transformed)[0]
        if len(vals) != len(names):
            return {"available": False, "reason": "SHAP feature mapping did not match the transformed model schema."}
        order = np.argsort(np.abs(vals))[::-1][:max_features]
        return {"available": True, "method": "XGBoost native TreeSHAP contributions (log-odds)",
                "base_value": float(_native_shap_values(model, transformed, include_bias=True)[0, -1]),
                "contributions": [{"feature": names[i], "value": float(vals[i]),
                                   "direction": "increases score" if vals[i] > 0 else "decreases score"}
                                  for i in order]}
    except Exception as exc:
        return {"available": False, "reason": f"SHAP unavailable: {type(exc).__name__}"}


def _native_shap_values(model, transformed, include_bias=False):
    """Return exact TreeSHAP contributions from XGBoost, aligned to input columns."""
    contributions = model.get_booster().predict(DMatrix(transformed), pred_contribs=True)
    contributions = np.asarray(contributions)
    return contributions if include_bias else contributions[:, :-1]
