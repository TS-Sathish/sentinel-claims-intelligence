from fastapi.testclient import TestClient
from src.api.main import app
import src.api.main as api
import src.services.scoring as scoring
import src.database.store as store
from src.ml.pipeline import train_models
import pandas as pd

def test_health():
    response=TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_score_endpoint_audits_without_direct_reference(tmp_path, monkeypatch):
    X = pd.DataFrame({"vehicle": ["sedan", "truck", "van", "sedan"] * 15,
                      "amount": [1, 4, 3, 8] * 15})
    y = pd.Series([0, 1, 0, 1] * 15, dtype="int8")
    train_models(X, y, tmp_path, target="label")
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(scoring, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(store, "DB", tmp_path / "audit.db")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    response = TestClient(app).post("/api/v1/claims/score", json={
        "claim_reference": "private-ref", "features": X.iloc[0].to_dict()})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")
    assert response.json()["model_version"] == "sentinel-xgb-2"
    rows = store.recent_events()
    assert len(rows) == 1 and "private-ref" not in str(rows[0])
    batch = TestClient(app).post("/api/v1/claims/score-batch", json={"claims": [
        {"features": X.iloc[1].to_dict()}, {"features": X.iloc[2].to_dict()}]})
    assert batch.status_code == 200 and batch.json()["count"] == 2
    assert len(store.recent_events()) == 3
    invalid = TestClient(app).post("/api/v1/claims/score", json={"features": {"vehicle": "sedan"}})
    assert invalid.status_code == 422
