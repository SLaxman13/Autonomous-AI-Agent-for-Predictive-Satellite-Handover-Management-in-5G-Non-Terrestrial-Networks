"""
Stage 2: Feature engineering
- Loads all_passes.csv (from Stage 1b)
- For each pass, builds sliding-window trend features (SNR/elevation
  history + rate of change) -- computed SEPARATELY per pass so we
  never leak information across different satellite passes
- Creates future-SNR prediction targets: snr at t+5s, t+10s, t+20s
- Saves final ML-ready dataset: training_data.csv
"""

import pandas as pd
import numpy as np

STEP_SECONDS = 2
WINDOW_STEPS = 5          # how many past samples to look back (10 sec of history)
HORIZONS_SEC = [5, 10, 20]  # how far ahead to predict

df = pd.read_csv("all_passes.csv")
print(f"Loaded {len(df)} rows across {df['pass_id'].nunique()} passes")

all_engineered = []

for pass_id, group in df.groupby("pass_id"):
    g = group.sort_values("t_sec").reset_index(drop=True)
    n = len(g)

    for i in range(n):
        row = {
            "pass_id": pass_id,
            "satellite": g.loc[i, "satellite"],
            "t_sec": g.loc[i, "t_sec"],
            "elevation_deg": g.loc[i, "elevation_deg"],
            "azimuth_deg": g.loc[i, "azimuth_deg"],
            "range_km": g.loc[i, "range_km"],
            "doppler_hz": g.loc[i, "doppler_hz"],
            "fspl_db": g.loc[i, "fspl_db"],
            "snr_db": g.loc[i, "snr_db"],
        }

        # ---- Trend features: need WINDOW_STEPS of history ----
        if i >= WINDOW_STEPS:
            snr_now = g.loc[i, "snr_db"]
            snr_past = g.loc[i - WINDOW_STEPS, "snr_db"]
            elev_now = g.loc[i, "elevation_deg"]
            elev_past = g.loc[i - WINDOW_STEPS, "elevation_deg"]

            window_sec = WINDOW_STEPS * STEP_SECONDS
            row["snr_delta"] = snr_now - snr_past
            row["snr_slope"] = (snr_now - snr_past) / window_sec
            row["elev_delta"] = elev_now - elev_past
            row["elev_slope"] = (elev_now - elev_past) / window_sec
            row["snr_rolling_mean"] = g.loc[i - WINDOW_STEPS:i, "snr_db"].mean()
            row["snr_rolling_std"] = g.loc[i - WINDOW_STEPS:i, "snr_db"].std()
        else:
            row["snr_delta"] = np.nan
            row["snr_slope"] = np.nan
            row["elev_delta"] = np.nan
            row["elev_slope"] = np.nan
            row["snr_rolling_mean"] = np.nan
            row["snr_rolling_std"] = np.nan

        # ---- Future targets: need HORIZON steps ahead within same pass ----
        for h_sec in HORIZONS_SEC:
            h_steps = h_sec // STEP_SECONDS
            target_idx = i + h_steps
            if target_idx < n:
                row[f"snr_plus_{h_sec}s"] = g.loc[target_idx, "snr_db"]
            else:
                row[f"snr_plus_{h_sec}s"] = np.nan

        all_engineered.append(row)

result = pd.DataFrame(all_engineered)

before = len(result)
result = result.dropna()
after = len(result)
print(f"Dropped {before - after} rows lacking full history/future window (kept {after})")

result.to_csv("training_data.csv", index=False)
print(f"\nSaved training_data.csv with {len(result)} rows, {len(result.columns)} columns")
print(f"\nColumns: {list(result.columns)}")
print(f"\nSample rows:")
print(result[["pass_id", "t_sec", "elevation_deg", "snr_db", "snr_slope",
              "snr_plus_5s", "snr_plus_10s", "snr_plus_20s"]].head(8))
print(f"\nTarget ranges:")
for h in HORIZONS_SEC:
    col = f"snr_plus_{h}s"
    print(f"  {col}: {result[col].min():.1f} - {result[col].max():.1f} dB")