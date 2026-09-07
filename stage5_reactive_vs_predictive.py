"""
Stage 5: Reactive vs Predictive comparison
- Uses training_data.csv (real passes, already has actual SNR +
  predicted SNR at each timestep from the trained models)
- REACTIVE approach: trigger a handover warning only when CURRENT
  SNR crosses the critical threshold (no prediction, just reacts)
- PREDICTIVE approach: trigger a warning when the model's predicted
  SNR (+10s horizon) crosses the critical threshold -- potentially
  earlier, since it's forecasting ahead
- Only evaluates the DESCENDING side of each pass (after peak
  elevation) -- the genuine handover-relevant degradation scenario,
  not the trivial start-of-pass low-elevation condition.
"""

import pandas as pd
import numpy as np

CRITICAL_THRESHOLD_DB = 9.5
PREDICTION_HORIZON_SEC = 10

df = pd.read_csv("training_data.csv")
print(f"Loaded {len(df)} rows across {df['pass_id'].nunique()} passes")

results = []

for pass_id, group in df.groupby("pass_id"):
    g = group.sort_values("t_sec").reset_index(drop=True)

    peak_idx = g["elevation_deg"].idxmax()
    g = g.loc[peak_idx:].reset_index(drop=True)

    if len(g) < 3:
        results.append({
            "pass_id": pass_id, "went_critical": False,
            "reactive_trigger_t": None, "predictive_trigger_t": None,
            "lead_time_sec": None, "outcome": "descending_segment_too_short"
        })
        continue

    reactive_critical = g[g["snr_db"] < CRITICAL_THRESHOLD_DB]
    reactive_trigger_t = reactive_critical["t_sec"].min() if len(reactive_critical) > 0 else None

    predictive_critical = g[g["snr_plus_10s"] < CRITICAL_THRESHOLD_DB]
    predictive_trigger_t = predictive_critical["t_sec"].min() if len(predictive_critical) > 0 else None

    if reactive_trigger_t is None and predictive_trigger_t is None:
        results.append({
            "pass_id": pass_id, "went_critical": False,
            "reactive_trigger_t": None, "predictive_trigger_t": None,
            "lead_time_sec": None, "outcome": "no_critical_event"
        })
        continue

    if reactive_trigger_t is not None and predictive_trigger_t is not None:
        raw_diff = reactive_trigger_t - predictive_trigger_t
        lead_time = raw_diff
        outcome = "predictive_earlier" if lead_time > 0 else (
            "predictive_later" if lead_time < 0 else "same_time")
        results.append({
            "pass_id": pass_id, "went_critical": True,
            "reactive_trigger_t": reactive_trigger_t, "predictive_trigger_t": predictive_trigger_t,
            "lead_time_sec": lead_time, "outcome": outcome
        })
    elif predictive_trigger_t is not None and reactive_trigger_t is None:
        # Check if this trigger is near the END of available data -- if so,
        # we can't actually verify whether the prediction was wrong, since
        # the satellite may have set (or recording ended) before we could
        # confirm it. This is a data-boundary limitation, not necessarily
        # a genuine false alarm from the model.
        max_t = g["t_sec"].max()
        unverifiable = (max_t - predictive_trigger_t) < PREDICTION_HORIZON_SEC
        outcome = "unverifiable_near_data_end" if unverifiable else "false_positive"
        results.append({
            "pass_id": pass_id, "went_critical": False,
            "reactive_trigger_t": None, "predictive_trigger_t": predictive_trigger_t,
            "lead_time_sec": None, "outcome": outcome
        })
    else:
        results.append({
            "pass_id": pass_id, "went_critical": True,
            "reactive_trigger_t": reactive_trigger_t, "predictive_trigger_t": None,
            "lead_time_sec": None, "outcome": "missed_detection"
        })

results_df = pd.DataFrame(results)
results_df.to_csv("reactive_vs_predictive_results.csv", index=False)

print(f"\n{'='*60}")
print("PER-PASS RESULTS")
print(f"{'='*60}")
print(results_df.to_string(index=False))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

n_total = len(results_df)
n_earlier = (results_df["outcome"] == "predictive_earlier").sum()
n_later = (results_df["outcome"] == "predictive_later").sum()
n_same = (results_df["outcome"] == "same_time").sum()
n_false_pos = (results_df["outcome"] == "false_positive").sum()
n_unverifiable = (results_df["outcome"] == "unverifiable_near_data_end").sum()
n_missed = (results_df["outcome"] == "missed_detection").sum()
n_no_event = (results_df["outcome"] == "no_critical_event").sum()

print(f"Total passes analyzed: {n_total}")
print(f"  Predictive detected EARLIER than reactive: {n_earlier}")
print(f"  Predictive detected LATER than reactive:   {n_later}")
print(f"  Same detection time:                       {n_same}")
print(f"  Missed detections: {n_missed}")
print(f"False positives (genuine, verifiable): {n_false_pos}")
print(f"Unverifiable predictions (too close to end of recorded data to confirm): {n_unverifiable}")
print(f"Passes with no critical event at all: {n_no_event}")

valid_lead_times = results_df[results_df["lead_time_sec"].notna()]["lead_time_sec"]
if len(valid_lead_times) > 0:
    print(f"\nAverage lead time (predictive vs reactive): {valid_lead_times.mean():.1f} sec")
    print(f"Median lead time: {valid_lead_times.median():.1f} sec")
    print(f"Max lead time: {valid_lead_times.max():.1f} sec")
    print(f"Min lead time: {valid_lead_times.min():.1f} sec")

print(f"\nSaved detailed results to reactive_vs_predictive_results.csv")