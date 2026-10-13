import pandas as pd
import pytest

from src.ml.pipeline import train_models, load_artifacts, validate_schema, load_global_explanation
from src.services.scoring import score_claim


def fixture_data():
    # Small synthetic fixture for offline code-path testing only; never used as
    # public-data evaluation or presented as a real claim.
    X = pd.DataFrame({"vehicle": ["sedan", "truck", "sedan", "van"] * 20,
                      "age_band": ["new", "old", "old", "new"] * 20,
                      "amount": [1.0, 4.0, 2.0, 8.0] * 20})
    y = pd.Series(([0, 1, 0, 1] * 20), dtype="int8")
    return X, y


def test_preprocessing_training_reload_and_score(tmp_path, monkeypatch):
    X, y = fixture_data()
    train_models(X, y, tmp_path, target="label")
    classifier, anomaly, metrics, manifest = load_artifacts(tmp_path)
    row = validate_schema(X.iloc[[0]], manifest["feature_columns"])
    assert 0 <= classifier.predict_proba(row)[0, 1] <= 1
    assert int(anomaly["model"].predict(anomaly["preprocessor"].transform(row))[0]) in (-1, 1)
    assert metrics["xgboost"]["pr_auc"] >= 0
    assert manifest["feature_columns"] == list(X.columns)
    import src.services.scoring as scoring
    monkeypatch.setattr(scoring, "MODEL_DIR", tmp_path)
    result = score_claim(X.iloc[0].to_dict())
    assert result["model_version"] == "sentinel-xgb-2"
    assert "anomaly_score" in result
    assert result["explanation"]["available"] is True
    assert result["explanation"]["method"].startswith("XGBoost native TreeSHAP")
    global_explanation = load_global_explanation(tmp_path)
    assert global_explanation["available"] is True
    assert all(item["feature"] in classifier.named_steps["prep"].get_feature_names_out()
               for item in global_explanation["features"])


def test_schema_mismatch_is_actionable():
    with pytest.raises(ValueError, match="missing features.*unexpected features"):
        validate_schema(pd.DataFrame([{"a": 1, "other": 2}]), ["a", "b"])

def test_numeric_schema_rejects_bad_values():
    with pytest.raises(ValueError, match="Feature 'amount' must be numeric"):
        validate_schema(pd.DataFrame([{"amount": "many"}]), ["amount"], {"amount": "float64"})
