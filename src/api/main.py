from __future__ import annotations

from contextvars import ContextVar
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import APP_NAME, DATASET_PATH, MODEL_DIR, RISK_REVIEW_THRESHOLD, TARGET_COLUMN
from src.data.loader import load_dataset
from src.ml.pipeline import load_artifacts, load_global_explanation
from src.services.scoring import score_claim
from src.services.narrative import narrative
from src.database.store import save_event, recent_events, analytics_summary

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sentinel.api")
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
MAX_BATCH_SIZE = 100
MAX_FEATURES = 200

app = FastAPI(title=APP_NAME, version="1.0.0",
              description="Local human-reviewed claim risk triage API. Scores are decision support, not findings.")


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_reference: str | None = Field(default=None, max_length=100)
    features: dict[str, Any] = Field(min_length=1, max_length=MAX_FEATURES,
                                     json_schema_extra={"example": {"VehicleCategory": "Sedan", "AgeOfVehicle": "3 years"}})

    @field_validator("features")
    @classmethod
    def primitive_feature_values(cls, values):
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Feature names must be non-empty strings.")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Feature '{key}' must be a scalar value or null.")
            if isinstance(value, str) and len(value) > 2048:
                raise ValueError(f"Feature '{key}' exceeds the 2048-character limit.")
        return values


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ClaimRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied[:100] if supplied and all(c.isalnum() or c in "-_." for c in supplied) else str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info('{"request_id":"%s","method":"%s","path":"%s","status":%s}',
                    request_id, request.method, request.url.path, response.status_code)
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": "invalid_request",
                        "message": "Request validation failed. Check field names, types, and limits.",
                        "request_id": request_id_var.get()})


@app.exception_handler(HTTPException)
async def http_error_handler(request, exc):
    detail = exc.detail if isinstance(exc.detail, str) else "Request could not be completed."
    return JSONResponse(status_code=exc.status_code, content={"error": "request_failed",
                        "message": detail, "request_id": request_id_var.get()})


@app.exception_handler(Exception)
async def server_error_handler(request, exc):
    logger.exception("unhandled_api_error request_id=%s error_type=%s", request_id_var.get(), type(exc).__name__)
    return JSONResponse(status_code=500, content={"error": "internal_error",
                        "message": "The request could not be completed. Check server logs using the request ID.",
                        "request_id": request_id_var.get()})


def _score(payload: ClaimRequest):
    try:
        result = score_claim(payload.features)
        result["narrative"] = narrative(result, payload.features)
        save_event(payload.claim_reference, result)
        return {"claim_reference": payload.claim_reference, **result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable. Train the model first.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("scoring_failed request_id=%s error_type=%s", request_id_var.get(), type(exc).__name__)
        raise HTTPException(status_code=500, detail="Claim scoring failed. Check server logs using the request ID.") from exc


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}


@app.get("/api/v1/model/status")
def model_status():
    try:
        _, _, _, manifest = load_artifacts(MODEL_DIR)
        return {"trained": True, "model_version": manifest["model_version"],
                "trained_at_utc": manifest["trained_at_utc"], "target": manifest.get("target"),
                "feature_count": len(manifest["feature_columns"]), "threshold": RISK_REVIEW_THRESHOLD}
    except (FileNotFoundError, ValueError):
        return {"trained": False, "model_version": None, "threshold": RISK_REVIEW_THRESHOLD}


@app.get("/api/v1/analytics/summary")
def analytics():
    result = {"audit": analytics_summary(), "dataset": None}
    if DATASET_PATH.exists():
        try:
            X, y, raw, target = load_dataset(DATASET_PATH, TARGET_COLUMN)
            result["dataset"] = {"rows": len(raw), "features": int(X.shape[1]),
                                  "target": target, "label_prevalence": float(y.mean()),
                                  "missing_cells": int(raw.isna().sum().sum()),
                                  "duplicate_rows": int(raw.duplicated().sum())}
        except Exception:
            result["dataset"] = {"available": False, "message": "Configured dataset could not be loaded."}
    return result


@app.post("/api/v1/claims/score")
def score(payload: ClaimRequest):
    return _score(payload)


@app.post("/api/v1/claims/score-batch")
def score_batch(payload: BatchRequest):
    return {"items": [_score(claim) for claim in payload.claims], "count": len(payload.claims)}


@app.get("/api/v1/claims/recent")
def recent(limit: int = 50):
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200.")
    return {"items": recent_events(limit)}


@app.get("/api/v1/explanations/global")
def global_explanations(limit: int = 25):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
    return load_global_explanation(MODEL_DIR, limit)
