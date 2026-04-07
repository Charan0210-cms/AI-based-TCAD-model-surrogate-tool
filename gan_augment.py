from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_generator import FEATURE_COLUMNS, TARGET_COLUMNS
from utils import DATA_DIR, ensure_dirs


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


@dataclass
class AugmentConfig:
    augment_fraction: float = 0.5
    seed: int = 42
    epochs: int = 150
    latent_dim: int = 8


def fit_autoencoder(x: np.ndarray, config: AugmentConfig) -> tuple[Autoencoder, StandardScaler]:
    torch.manual_seed(config.seed)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    model = Autoencoder(input_dim=x.shape[1], latent_dim=config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        recon, _ = model(x_tensor)
        loss = criterion(recon, x_tensor)
        loss.backward()
        optimizer.step()

    return model, scaler


def augment_dataset(df: pd.DataFrame, config: AugmentConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    x = df[FEATURE_COLUMNS + TARGET_COLUMNS].copy()
    x[["ion_a", "ioff_a", "cgg_f", "ion_over_cgg"]] = np.log10(
        x[["ion_a", "ioff_a", "cgg_f", "ion_over_cgg"]]
    )

    model, scaler = fit_autoencoder(x.values, config)
    with torch.no_grad():
        x_scaled = torch.tensor(scaler.transform(x.values), dtype=torch.float32)
        _, z = model(x_scaled)
        z_np = z.numpy()

    n_new = int(len(df) * config.augment_fraction)
    sampled_idx = rng.integers(0, len(df), size=n_new)
    sampled_latents = z_np[sampled_idx] + rng.normal(0.0, 0.18, size=(n_new, z_np.shape[1]))

    with torch.no_grad():
        decoded = model.decoder(torch.tensor(sampled_latents, dtype=torch.float32)).numpy()

    decoded = scaler.inverse_transform(decoded)
    aug = pd.DataFrame(decoded, columns=FEATURE_COLUMNS + TARGET_COLUMNS)

    for col in ["ion_a", "ioff_a", "cgg_f", "ion_over_cgg"]:
        aug[col] = 10 ** aug[col]

    aug["corner_radius_nm"] = aug["corner_radius_nm"].clip(0.4, 3.0)
    aug["gate_length_nm"] = aug["gate_length_nm"].clip(10.0, 22.0)
    aug["oxide_thickness_nm"] = aug["oxide_thickness_nm"].clip(0.7, 1.6)
    aug["channel_thickness_nm"] = aug["channel_thickness_nm"].clip(4.0, 8.0)
    aug["sheet_spacing_nm"] = aug["sheet_spacing_nm"].clip(8.0, 16.0)
    aug["channel_doping_cm3"] = aug["channel_doping_cm3"].clip(1e15, 2e18)
    aug["sd_doping_cm3"] = aug["sd_doping_cm3"].clip(1e19, 4e20)
    aug["stack_count"] = aug["stack_count"].round().clip(1, 3)
    aug["vgs_v"] = aug["vgs_v"].clip(0.0, 0.8)
    aug["vds_v"] = aug["vds_v"].clip(0.05, 0.8)
    aug["ft_ghz"] = aug["ft_ghz"].clip(1.0, 1000.0)

    combined = pd.concat([df, aug], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment synthetic nanosheet data.")
    parser.add_argument("--augment_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    ensure_dirs()
    in_path = DATA_DIR / "raw_device_data.csv"
    df = pd.read_csv(in_path)
    augmented = augment_dataset(
        df,
        AugmentConfig(
            augment_fraction=args.augment_fraction,
            seed=args.seed,
            epochs=args.epochs,
        ),
    )
    out_path = DATA_DIR / "augmented_device_data.csv"
    augmented.to_csv(out_path, index=False)
    print(f"Saved augmented dataset with {len(augmented)} rows to {out_path}")


if __name__ == "__main__":
    main()
