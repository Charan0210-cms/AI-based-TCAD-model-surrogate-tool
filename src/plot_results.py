from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import FEATURE_COLUMNS, TARGET_COLUMNS
from utils import DATA_DIR, MODEL_DIR


def predict_bundle(bundle: dict, X_df: pd.DataFrame) -> pd.DataFrame:
    X_scaled = bundle["scaler"].transform(X_df[bundle["feature_columns"]])
    out = {}
    for target, model in bundle["models"].items():
        pred = model.predict(X_scaled)
        if target in set(bundle["log_targets"]):
            pred = 10 ** pred
        out[target] = pred
    return pd.DataFrame(out)


def main() -> None:
    input_path = DATA_DIR / "augmented_device_data.csv"
    if not input_path.exists():
        input_path = DATA_DIR / "raw_device_data.csv"
    df = pd.read_csv(input_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bundle = joblib.load(MODEL_DIR / "surrogate_multioutput.joblib")
    preds = predict_bundle(bundle, X_test)

    for target in TARGET_COLUMNS[:4]:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test[target], preds[target], alpha=0.35)
        low = min(y_test[target].min(), preds[target].min())
        high = max(y_test[target].max(), preds[target].max())
        plt.plot([low, high], [low, high])
        plt.xlabel(f"True {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"Predicted vs True: {target}")
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"pred_vs_true_{target}.png", dpi=200)
        plt.close()

    print("Saved plots to models/")
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        print(json.dumps(json.loads(metrics_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
