from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from data_generator import FEATURE_COLUMNS
from utils import MODEL_DIR

st.set_page_config(page_title="AI-TCAD Surrogate", layout="wide")
st.title("AI-TCAD Surrogate for Nanosheet FET Optimization")
st.caption("Interactive surrogate model inspired by ML-augmented TCAD workflows.")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_DIR / "surrogate_multioutput.joblib")


def predict_bundle(bundle: dict, X_df: pd.DataFrame) -> pd.DataFrame:
    X_scaled = bundle["scaler"].transform(X_df[bundle["feature_columns"]])
    out = {}
    for target, model in bundle["models"].items():
        pred = model.predict(X_scaled)
        if target in set(bundle["log_targets"]):
            pred = 10 ** pred
        out[target] = pred
    return pd.DataFrame(out)


bundle = load_bundle()

col1, col2 = st.columns(2)
with col1:
    corner_radius_nm = st.slider("Corner radius (nm)", 0.4, 3.0, 1.5, 0.1)
    gate_length_nm = st.slider("Gate length (nm)", 10.0, 22.0, 12.0, 0.5)
    oxide_thickness_nm = st.slider("Oxide thickness (nm)", 0.7, 1.6, 0.9, 0.05)
    channel_thickness_nm = st.slider("Channel thickness (nm)", 4.0, 8.0, 5.0, 0.1)
    sheet_spacing_nm = st.slider("Sheet spacing (nm)", 8.0, 16.0, 11.0, 0.5)
with col2:
    channel_doping_cm3 = st.number_input("Channel doping (cm^-3)", value=1e16, step=1e15, format="%.3e")
    sd_doping_cm3 = st.number_input("Source/drain doping (cm^-3)", value=1e20, step=1e19, format="%.3e")
    stack_count = st.selectbox("Stack count", [1, 2, 3], index=1)
    vgs_v = st.slider("VGS (V)", 0.0, 0.8, 0.7, 0.01)
    vds_v = st.slider("VDS (V)", 0.05, 0.8, 0.7, 0.01)

sample = pd.DataFrame(
    [[
        corner_radius_nm,
        gate_length_nm,
        oxide_thickness_nm,
        channel_thickness_nm,
        sheet_spacing_nm,
        channel_doping_cm3,
        sd_doping_cm3,
        stack_count,
        vgs_v,
        vds_v,
    ]],
    columns=FEATURE_COLUMNS,
)

results = predict_bundle(bundle, sample).iloc[0].to_dict()

metric_cols = st.columns(5)
metric_cols[0].metric("ION (A)", f"{results['ion_a']:.3e}")
metric_cols[1].metric("IOFF (A)", f"{results['ioff_a']:.3e}")
metric_cols[2].metric("Cgg (F)", f"{results['cgg_f']:.3e}")
metric_cols[3].metric("fT (GHz)", f"{results['ft_ghz']:.2f}")
metric_cols[4].metric("ION/Cgg", f"{results['ion_over_cgg']:.3e}")

st.subheader("Model input")
st.dataframe(sample)
