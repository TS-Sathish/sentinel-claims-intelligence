from fastapi.testclient import TestClient

from src.api.main import app
import src.database.store as store
from src.services.narrative import narrative, _minimized_features


def test_narrative_local_fallback_and_minimization(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    result = {"fraud_probability": .4, "anomaly_flag": False}
    text = narrative(result, {"PolicyNumber": 123, "Age": 40, "VehicleCategory": "Sedan"})
    assert "Limitations:" in text and "Verification steps:" in text
    assert _minimized_features({"PolicyNumber": 123, "VehicleCategory": "Sedan"}) == {"VehicleCategory": "Sedan"}


def test_database_writes_audit_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "audit.db")
    result = {"fraud_probability": .7, "anomaly_score": .9, "anomaly_flag": True,
              "risk_band": "High review priority", "threshold": .65, "model_version": "test",
              "narrative": "Review manually", "explanation": {"available": False}}
    store.save_event("private-reference", result)
    rows = store.recent_events()
    assert len(rows) == 1
    assert rows[0]["threshold"] == .65
    assert rows[0]["model_version"] == "test"
    assert rows[0]["anomaly_flag"] == 1
    assert "private-reference" not in str(rows[0])


def test_api_health_batch_limit_and_request_id():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    too_many = client.post("/api/v1/claims/score-batch", json={"claims": [{"features": {"x": 1}}] * 101})
    assert too_many.status_code == 422
    assert too_many.json()["error"] == "invalid_request"
