import pandas as pd
from src.config import MODEL_DIR, RISK_REVIEW_THRESHOLD
from src.ml.pipeline import load_artifacts, validate_schema, explain_local

def score_claim(features):
    classifier, anomaly_bundle, metrics, manifest = load_artifacts(MODEL_DIR)
    row = validate_schema(pd.DataFrame([features]), manifest["feature_columns"],
                          manifest.get("feature_dtypes", {}))
    probability = float(classifier.predict_proba(row)[:, 1][0])
    transformed = anomaly_bundle["preprocessor"].transform(row)
    anomaly_model = anomaly_bundle["model"]
    anomaly_flag = int(anomaly_model.predict(transformed)[0]) == -1
    score = float(-anomaly_model.score_samples(transformed)[0])
    band = "High review priority" if probability >= RISK_REVIEW_THRESHOLD else ("Elevated review priority" if probability >= .35 else "Standard review priority")
    return {"fraud_probability": probability, "risk_band": band, "anomaly_flag": anomaly_flag,
            "anomaly_score": score, "threshold": RISK_REVIEW_THRESHOLD,
            "model_version": manifest["model_version"],
            "explanation": explain_local(classifier, row),
            "metrics": {k: metrics.get(k) for k in ["precision","recall","f1","pr_auc","roc_auc"]},
            "notice": "Risk indicators support human review; they do not establish fraud."}
