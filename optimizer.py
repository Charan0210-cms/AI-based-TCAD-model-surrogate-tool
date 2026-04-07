from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from data_generator import FEATURE_COLUMNS, TARGET_COLUMNS
from utils import MODEL_DIR


DEFAULTS = {
    "gate_length_nm": 12.0,
    "oxide_thickness_nm": 0.9,
    "channel_thickness_nm": 5.0,
    "sheet_spacing_nm": 11.0,
    "channel_doping_cm3": 1e16,
    "sd_doping_cm3": 1e20,
    "stack_count": 2,
    "vgs_v": 0.7,
    "vds_v": 0.7,
}


def predict_bundle(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    X_scaled = bundle["scaler"].transform(X[bundle["feature_columns"]])
    out = {}
    for target, model in bundle["models"].items():
        pred = model.predict(X_scaled)
        if target in set(bundle["log_targets"]):
            pred = 10 ** pred
        out[target] = pred
    return pd.DataFrame(out)


def build_search_grid(n_per_dim: int = 24) -> pd.DataFrame:
    rows = []
    for corner_radius_nm in np.linspace(0.4, 3.0, n_per_dim):
        for gate_length_nm in np.linspace(10.0, 20.0, n_per_dim):
            row = {"corner_radius_nm": corner_radius_nm, **DEFAULTS}
            row["gate_length_nm"] = gate_length_nm
            rows.append(row)
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize device design using trained surrogate.")
    parser.add_argument("--objective", type=str, default="ion_over_cgg", choices=TARGET_COLUMNS)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    bundle = joblib.load(MODEL_DIR / "surrogate_multioutput.joblib")
    grid = build_search_grid()
    pred_df = predict_bundle(bundle, grid)
    results = pd.concat([grid.reset_index(drop=True), pred_df], axis=1)

    ascending = args.objective == "ioff_a"
    ranked = results.sort_values(args.objective, ascending=ascending).head(args.top_k)
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    main()
