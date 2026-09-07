# Autonomous-AI-Agent-for-Predictive-Satellite-Handover-Management-in-5G-Non-Terrestrial-Networks

An autonomous decision-making agent that predicts satellite link degradation before it happens and proactively manages handovers in a 5G NTN (Non-Terrestrial Network) system — built on real Starlink orbital data, real physics-based RF modeling, and a trained, honestly-evaluated ML forecasting layer.

![Dashboard Screenshot](dashboard.png)

## The Problem

Traditional satellite handover systems are **reactive** — they switch to a new satellite only after the current link has already degraded. In Low Earth Orbit (LEO) constellations like Starlink, satellites move fast and visibility windows are short, so reactive systems risk brief outages before they respond.

**Research question:** can a system forecast link degradation a few seconds ahead, and use that forecast — through an explainable, safety-constrained autonomous agent rather than a black-box model — to make earlier, more reliable handover decisions?

## Project Structure

| File | Purpose |
|---|---|
| `stage1b_collect_passes.py` | Collects real satellite pass data via TLE + orbital propagation |
| `stage2_feature_engineering.py` | Builds sliding-window trend features and future-SNR targets |
| `stage3_train_model.py` | Trains and evaluates the SNR forecasting models |
| `stage5_reactive_vs_predictive.py` | Reactive vs. predictive detection-timing comparison |
| `live_tracker.py` | Real-time multi-satellite TLE tracking module |
| `agent.py` | The autonomous learning agent (decision loop + safety layer) |
| `app_real.py` | FastAPI backend serving telemetry and agent decisions |
| `index.html` | Live dashboard (map, forecasts, agent state) |

## How It Was Built, Stage by Stage

### Stage 1 — Real Orbital Data Pipeline
Pulls live Starlink TLE (Two-Line Element) data from Celestrak and uses `skyfield` to propagate real satellite positions over a fixed ground station — **Chennai Gateway (13.08°N, 80.27°E)**. For each timestep (every 2 seconds through a real satellite pass), the pipeline computes:
- Elevation, azimuth, and range (from real orbital mechanics, not simulated)
- Doppler shift (from measured range-rate)
- Free Space Path Loss (FSPL) and a link budget, producing a physically-derived SNR (including realistic atmospheric scintillation and receiver noise, so the signal isn't artificially noise-free)

**25 real satellite passes** were collected across a range of elevations (10.7°–87.9°) and durations, to ensure the dataset reflects genuinely varied real-world conditions rather than one narrow scenario.

### Stage 2 — Feature Engineering
Raw per-timestep readings alone aren't enough to predict *future* link quality — the model needs to see trends. For each pass (computed independently per pass, so no data leaks across different satellites), this stage builds:
- Rate-of-change features: ΔSNR, ΔElevation over a 10-second sliding window
- Rolling mean/std of SNR
- **Prediction targets:** the actual SNR value 5, 10, and 20 seconds into the future (only possible because this is historical/simulated data where "the future" is already known — a real deployment would only have this after the fact, for training)

### Stage 3 — Model Training and Evaluation
Three separate Random Forest Regressors were trained — one per forecast horizon (+5s, +10s, +20s). Critically, the train/test split was done **by satellite pass, not by row** — the model is tested only on satellite passes it has never seen, avoiding the false confidence of testing on near-identical adjacent timesteps.

**Results (evaluated on held-out real passes):**

| Horizon | MAE | RMSE | R² | Improvement over naive baseline |
|---|---|---|---|---|
| +5s | 0.325 dB | 0.413 dB | 0.984 | 30.1% |
| +10s | 0.354 dB | 0.444 dB | 0.981 | 44.1% |
| +20s | 0.394 dB | 0.510 dB | 0.975 | 62.6% |

The baseline used for comparison is **persistence** (naively assuming future SNR equals current SNR) — a standard, honest baseline in forecasting research. Feature importance analysis showed the model relies heavily on `range_km` and `fspl_db` (deterministic orbital geometry) rather than the noisier instantaneous SNR reading — physically sensible behavior, since geometry is far more predictable than a noisy signal measurement.

### Stage 4 — Live Multi-Satellite Tracking and Real-Time Agent
`live_tracker.py` tracks up to 1,000 satellites simultaneously (sampled across the full ~11,000-satellite constellation, not just the first N, since consecutive TLE entries share the same orbital plane and would otherwise cluster together in the sky). At any moment, it reports every satellite currently visible above a 10° elevation mask, each with full real telemetry and rolling trend history.

### Stage 5 — Reactive vs. Predictive Comparison
To honestly test the core research claim, a comparison harness measures **detection timing** across all 25 real passes: when does a reactive approach (current SNR crosses a critical threshold) versus a predictive approach (forecast SNR crosses that threshold) first flag trouble — restricted to the genuine descending/degradation side of each pass (not the trivial low-elevation condition every pass starts at).

**Result:** across 11 verifiable critical events, the predictive approach was never later than reactive, matched it in 4 cases, and detected it up to **10 seconds earlier** in 7 cases, with **zero missed detections and zero genuine false positives**.

### Stage 6 — The Autonomous Agent
`agent.py` implements a full agent decision loop, not just an if/else script:

- **Observe:** reads live telemetry, maintains a persistent "locked" serving satellite (doesn't silently flicker between satellites every poll)
- **Predict:** calls the trained forecasting models
- **Evaluate:** scores every visible candidate satellite using a utility function (weighted combination of predicted SNR, elevation, and remaining visibility duration)
- **Plan + Safety Check:** decides KEEP / PREPARE_HANDOVER / HANDOVER / HANDOVER_BLOCKED, enforcing a **handover cooldown** (prevents rapid back-and-forth switching) and a **minimum utility-gain threshold** (won't switch for a marginal improvement)
- **Execute:** performs the handover and logs the decision with full reasoning

### Stage 7 — Learning Element (Adaptive Utility Weights)
The agent doesn't use fixed weights forever. After each handover, a **critic** measures the real outcome (did SNR actually improve?), and a **learning element** nudges the utility weights toward whatever factors were associated with good outcomes, and away from factors associated with bad ones — a simple, fully traceable reward-weighted update (every weight change is logged with the exact reward that caused it). This follows the classic Russell & Norvig learning-agent framework (performance element + critic + learning element), deliberately without full reinforcement learning or an LLM in the decision loop — both would sacrifice explainability for a real-time safety-relevant system without a clear benefit at this scope.

### Stage 8 — Live Dashboard
A FastAPI backend serves real-time telemetry and agent decisions to a frontend built with Leaflet.js, showing:
- A real map with the serving satellite's actual ground-track position (from `skyfield`-derived lat/lon) and visible candidates
- Live link quality, forecast, and handover status
- The agent's current utility weights and a log of how they've evolved
- A system event log of every decision and handover

## Honest Limitations

- Satellite sampling (1,000 of ~11,000 total) means a handover candidate isn't always visible at every moment — the agent correctly reports this rather than forcing a bad decision.
- The SNR noise model (scintillation + receiver noise) is a simplified approximation, not a full 3GPP channel model.
- The learning mechanism is a lightweight, single-step reward-weighted heuristic — genuinely a form of online learning, but not full reinforcement learning with a trained value function.
- Weight updates are based on single-event outcomes, so they are somewhat sensitive to the underlying SNR noise; averaging over multiple outcomes before updating is a natural extension.

## Tech Stack

Python · Skyfield (orbital mechanics) · scikit-learn (Random Forest regression) · FastAPI · Leaflet.js · Chart.js · pandas · numpy · real Starlink TLE data via Celestrak

## Running It Locally

```bash
pip install -r requirements.txt

# Rebuild the training pipeline (optional -- trained models are included)
python stage1b_collect_passes.py
python stage2_feature_engineering.py
python stage3_train_model.py

# Run the live system
uvicorn app_real:app --reload
```
