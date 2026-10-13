from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

DB = Path(__file__).resolve().parents[2] / "data" / "claims.db"


def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS scoring_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, claim_reference_hash TEXT,
          fraud_probability REAL NOT NULL, anomaly_score REAL, anomaly_flag INTEGER NOT NULL,
          risk_band TEXT NOT NULL, threshold REAL NOT NULL, model_version TEXT NOT NULL,
          narrative TEXT NOT NULL, explanation_json TEXT NOT NULL,
          result_json TEXT NOT NULL, created_at TEXT NOT NULL)''')
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scoring_events)")}
        migrations = {
            "claim_reference_hash": "TEXT", "anomaly_flag": "INTEGER NOT NULL DEFAULT 0",
            "threshold": "REAL NOT NULL DEFAULT 0.65", "model_version": "TEXT NOT NULL DEFAULT 'legacy'",
            "narrative": "TEXT NOT NULL DEFAULT ''", "explanation_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, type_sql in migrations.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE scoring_events ADD COLUMN {name} {type_sql}")


def _reference_hash(reference):
    if not reference:
        return None
    return hashlib.sha256(str(reference).encode("utf-8")).hexdigest()


def save_event(reference, result):
    init_db()
    with sqlite3.connect(DB) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scoring_events)")}
        legacy = "claim_reference" in columns
        insert_columns = ["claim_reference_hash"] + (["claim_reference"] if legacy else []) + [
            "fraud_probability", "anomaly_score", "anomaly_flag", "risk_band", "threshold",
            "model_version", "narrative", "explanation_json", "result_json", "created_at"]
        params = [_reference_hash(reference)] + ([_reference_hash(reference)] if legacy else []) + [
            result["fraud_probability"], result.get("anomaly_score"), int(bool(result.get("anomaly_flag"))),
            result["risk_band"], result["threshold"], result["model_version"], result.get("narrative", ""),
            json.dumps(result.get("explanation", {})), json.dumps(result), datetime.now(timezone.utc).isoformat()]
        placeholders = ",".join("?" for _ in params)
        conn.execute(f"""INSERT INTO scoring_events({','.join(insert_columns)})
          VALUES({placeholders})""", params)


def recent_events(limit=50):
    init_db()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT id,fraud_probability,anomaly_score,anomaly_flag,risk_band,
          threshold,model_version,narrative,explanation_json,result_json,created_at
          FROM scoring_events ORDER BY id DESC LIMIT ?""", (max(1, min(int(limit), 200)),)).fetchall()
        return [dict(row) for row in rows]


def analytics_summary():
    init_db()
    with sqlite3.connect(DB) as conn:
        row = conn.execute("""SELECT COUNT(*), AVG(fraud_probability), SUM(anomaly_flag),
          MAX(created_at) FROM scoring_events""").fetchone()
    return {"scored_claims": row[0], "mean_fraud_probability": row[1],
            "anomaly_flags": row[2] or 0, "last_scored_at": row[3]}
