from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data_generator import FEATURE_COLUMNS, TARGET_COLUMNS
from utils import DATA_DIR, MODEL_DIR, ensure_dirs

LOG_TARGETS = {"ion_a", "ioff_a", "cgg_f", "ion_over_cgg"}


def prepare_targets(y_df: pd.DataFrame) -> pd.DataFrame:
    y = y_df.copy()
    for col in LOG_TARGETS:
        y[col] = np.log10(np.clip(y[col], 1e-20, None))
    return y


def inverse_target(name: str, values: np.ndarray) -> np.ndarray:
    if name in LOG_TARGETS:
        return 10 ** values
    return values


def build_base_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}
    for idx, name in enumerate(TARGET_COLUMNS):
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))),
            "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
        }
    return metrics


def main() -> None:
    ensure_dirs()
    input_path = DATA_DIR / "augmented_device_data.csv"
    if not input_path.exists():
        input_path = DATA_DIR / "raw_device_data.csv"

    df = pd.read_csv(input_path)
    X = df[FEATURE_COLUMNS]
    y_raw = df[TARGET_COLUMNS]
    y_trainable = prepare_targets(y_raw)

    X_train, X_test, y_train, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=42
    )
    y_trainable_train = prepare_targets(y_train.loc[:, TARGET_COLUMNS])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    pred_cols = []
    for target in TARGET_COLUMNS:
        model = build_base_model()
        model.fit(X_train_scaled, y_trainable_train[target].values)
        pred = inverse_target(target, model.predict(X_test_scaled))
        pred_cols.append(pred)
        models[target] = model

    preds = np.column_stack(pred_cols)
    metrics = compute_metrics(y_test_raw.values, preds)

    bundle = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "log_targets": sorted(LOG_TARGETS),
        "models": models,
    }

    model_path = MODEL_DIR / "surrogate_multioutput.joblib"
    joblib.dump(bundle, model_path)

    metrics_path = MODEL_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model bundle to {model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
