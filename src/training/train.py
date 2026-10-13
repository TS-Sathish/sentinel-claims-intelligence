import json
from pathlib import Path
from src.config import DATASET_PATH, MODEL_DIR, TARGET_COLUMN
from src.data.loader import load_dataset, excluded_columns
from src.ml.pipeline import train_models

def main():
    X, y, raw, target = load_dataset(DATASET_PATH, TARGET_COLUMN)
    metrics = train_models(X, y, MODEL_DIR, target=target, dataset_path=DATASET_PATH,
                           exclusions=excluded_columns(raw.columns, target))
    summary = {name: {key: result[key] for key in ("precision", "recall", "f1", "pr_auc", "roc_auc", "threshold")}
               for name, result in (("xgboost", metrics["xgboost"]), ("baseline", metrics["baseline"]))}
    print(json.dumps({"dataset": str(DATASET_PATH), "rows": len(raw), "features": X.shape[1],
        "target": target, "train_rows": metrics["train_rows"], "test_rows": metrics["test_rows"],
        "test_label_prevalence": metrics["test_fraud_rate"], "models": summary,
        "manifest": str(Path(MODEL_DIR) / "manifest.json")}, indent=2))

if __name__ == "__main__":
    main()
