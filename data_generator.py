from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import DATA_DIR, ensure_dirs


@dataclass
class DeviceConfig:
    n_samples: int = 6000
    seed: int = 42


FEATURE_COLUMNS = [
    "corner_radius_nm",
    "gate_length_nm",
    "oxide_thickness_nm",
    "channel_thickness_nm",
    "sheet_spacing_nm",
    "channel_doping_cm3",
    "sd_doping_cm3",
    "stack_count",
    "vgs_v",
    "vds_v",
]

TARGET_COLUMNS = ["ion_a", "ioff_a", "cgg_f", "ft_ghz", "ion_over_cgg"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(config: DeviceConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    corner_radius_nm = rng.uniform(0.4, 3.0, n)
    gate_length_nm = rng.uniform(10.0, 22.0, n)
    oxide_thickness_nm = rng.uniform(0.7, 1.6, n)
    channel_thickness_nm = rng.uniform(4.0, 8.0, n)
    sheet_spacing_nm = rng.uniform(8.0, 16.0, n)
    channel_doping_cm3 = 10 ** rng.uniform(15.0, 18.3, n)
    sd_doping_cm3 = 10 ** rng.uniform(19.2, 20.5, n)
    stack_count = rng.integers(1, 4, n)
    vgs_v = rng.uniform(0.0, 0.8, n)
    vds_v = rng.uniform(0.05, 0.8, n)

    threshold_v = (
        0.26
        + 0.006 * (gate_length_nm - 12.0)
        + 0.03 * (oxide_thickness_nm - 1.0)
        - 0.02 * (corner_radius_nm - 1.2)
        + 0.015 * np.log10(channel_doping_cm3 / 1e16)
        + 0.008 * (channel_thickness_nm - 5.0)
    )

    drive_factor = (
        1.15
        * stack_count
        * (channel_thickness_nm / 5.0) ** 0.6
        * (12.0 / gate_length_nm) ** 0.7
        * (sd_doping_cm3 / 1e20) ** 0.08
        * np.exp(-0.10 * (oxide_thickness_nm - 0.9))
        * np.exp(-0.07 * np.maximum(corner_radius_nm - 1.7, 0.0) ** 2)
    )

    turn_on = sigmoid((vgs_v - threshold_v) / 0.055)
    channel_mod = 1.0 - np.exp(-vds_v / 0.18)

    ion_a = 8e-7 * drive_factor * turn_on * channel_mod
    ion_a *= rng.normal(1.0, 0.035, n)
    ion_a = np.clip(ion_a, 1e-10, None)

    leakage_shape = (
        1.0
        + 0.22 * np.maximum(1.2 - corner_radius_nm, 0.0)
        + 0.18 * (1.2 / oxide_thickness_nm)
        + 0.12 * np.maximum(12.0 - gate_length_nm, 0.0) / 12.0
    )
    ioff_a = 5e-11 * leakage_shape * np.exp(-(corner_radius_nm - 0.5) / 1.15)
    ioff_a *= np.exp(0.4 * np.maximum(threshold_v - vgs_v, -0.15))
    ioff_a *= rng.normal(1.0, 0.05, n)
    ioff_a = np.clip(ioff_a, 1e-13, None)

    cgg_f = (
        3.2e-17
        * stack_count
        * gate_length_nm
        * (channel_thickness_nm + 0.5 * sheet_spacing_nm)
        / oxide_thickness_nm
        * (1.0 + 0.05 * np.maximum(1.5 - corner_radius_nm, 0.0))
    )
    cgg_f *= rng.normal(1.0, 0.025, n)
    cgg_f = np.clip(cgg_f, 1e-17, None)

    ft_ghz = ion_a / (2.0 * np.pi * cgg_f * np.maximum(vgs_v, 0.08)) / 1e9
    ft_ghz *= rng.normal(1.0, 0.03, n)
    ft_ghz = np.clip(ft_ghz, 1.0, None)

    ion_over_cgg = ion_a / cgg_f

    df = pd.DataFrame(
        {
            "corner_radius_nm": corner_radius_nm,
            "gate_length_nm": gate_length_nm,
            "oxide_thickness_nm": oxide_thickness_nm,
            "channel_thickness_nm": channel_thickness_nm,
            "sheet_spacing_nm": sheet_spacing_nm,
            "channel_doping_cm3": channel_doping_cm3,
            "sd_doping_cm3": sd_doping_cm3,
            "stack_count": stack_count,
            "vgs_v": vgs_v,
            "vds_v": vds_v,
            "ion_a": ion_a,
            "ioff_a": ioff_a,
            "cgg_f": cgg_f,
            "ft_ghz": ft_ghz,
            "ion_over_cgg": ion_over_cgg,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic nanosheet FET data.")
    parser.add_argument("--n_samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    df = generate_dataset(DeviceConfig(n_samples=args.n_samples, seed=args.seed))
    out_path = DATA_DIR / "raw_device_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
