"""
Stage 3: Train and evaluate the model
- Loads training_data.csv (from Stage 2)
- Splits by PASS, not by row -- so the model is tested on satellite
  passes it has never seen, which is the honest way to evaluate this
- Trains one Random Forest Regressor per prediction horizon (5s/10s/20s)
- Evaluates with MAE, RMSE, R^2 -- real numbers, no shortcuts
- Compares against a naive "persistence" baseline (predicted = current SNR)
- Saves the trained models to disk for later use in app.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("training_data.csv")
print(f"Loaded {len(df)} rows, {df['pass_id'].nunique()} passes")

# ---- Features used for prediction ----
FEATURE_COLS = [
    "elevation_deg", "azimuth_deg", "range_km", "doppler_hz", "fspl_db",
    "snr_db", "snr_delta", "snr_slope", "elev_delta", "elev_slope",
    "snr_rolling_mean", "snr_rolling_std"
]
HORIZONS_SEC = [5, 10, 20]

# ---- Split by PASS (not by row!) so test passes are fully unseen ----
all_passes = sorted(df["pass_id"].unique())
np.random.seed(42)
shuffled = np.random.permutation(all_passes)

n_test = max(2, int(len(all_passes) * 0.3))
test_passes = set(shuffled[:n_test])
train_passes = set(shuffled[n_test:])

print(f"\nTrain passes ({len(train_passes)}): {sorted(train_passes)}")
print(f"Test passes  ({len(test_passes)}): {sorted(test_passes)}")

train_df = df[df["pass_id"].isin(train_passes)]
test_df = df[df["pass_id"].isin(test_passes)]
print(f"\nTrain rows: {len(train_df)}, Test rows: {len(test_df)}")

X_train = train_df[FEATURE_COLS]
X_test = test_df[FEATURE_COLS]

results_summary = []
models = {}

for h in HORIZONS_SEC:
    target_col = f"snr_plus_{h}s"
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # ---- Baseline: naive persistence (assume SNR doesn't change) ----
    baseline_preds = test_df["snr_db"]  # "future SNR = current SNR"
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

    print(f"\n--- Horizon +{h}s ---")
    print(f"  Model    -> MAE: {mae:.3f} dB, RMSE: {rmse:.3f} dB, R2: {r2:.3f}")
    print(f"  Baseline -> MAE: {baseline_mae:.3f} dB, RMSE: {baseline_rmse:.3f} dB (persistence)")
    improvement = (baseline_mae - mae) / baseline_mae * 100
    print(f"  Improvement over baseline: {improvement:.1f}%")

    results_summary.append({
        "horizon_sec": h,
        "model_mae": round(mae, 3),
        "model_rmse": round(rmse, 3),
        "model_r2": round(r2, 3),
        "baseline_mae": round(baseline_mae, 3),
        "baseline_rmse": round(baseline_rmse, 3),
        "improvement_pct": round(improvement, 1)
    })

    models[h] = model
    joblib.dump(model, f"model_snr_plus_{h}s.pkl")

# ---- Feature importance (useful for your report/viva) ----
print(f"\n{'='*50}")
print("Feature importance (10s horizon model):")
importances = pd.Series(models[10].feature_importances_, index=FEATURE_COLS)
print(importances.sort_values(ascending=False))

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv("evaluation_results.csv", index=False)
print(f"\n{'='*50}")
print("FINAL EVALUATION SUMMARY")
print(summary_df.to_string(index=False))
print(f"\nModels saved: model_snr_plus_5s.pkl, model_snr_plus_10s.pkl, model_snr_plus_20s.pkl")
print("Evaluation table saved: evaluation_results.csv")