"""
5G NTN Predictive Handover Engine -- LEARNING AGENT VERSION
NetworkOperationsAgent adapts its own utility weights based on
observed handover outcomes. Now also exposes real ground-track
subpoints (lat/lon) for map display.
"""

import time
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from live_tracker import LiveTracker
from agent import NetworkOperationsAgent

app = FastAPI(title="5G NTN Predictive Handover Engine - Learning Agent")

ELEVATION_MASK_DEG = 10.0

MODEL_HORIZONS = [5, 10, 20]
models = {h: joblib.load(f"model_snr_plus_{h}s.pkl") for h in MODEL_HORIZONS}

FEATURE_COLS = [
    "elevation_deg", "azimuth_deg", "range_km", "doppler_hz", "fspl_db",
    "snr_db", "snr_delta", "snr_slope", "elev_delta", "elev_slope",
    "snr_rolling_mean", "snr_rolling_std"
]


def predict_future_snr(telemetry: dict):
    row = pd.DataFrame([{col: telemetry[col] for col in FEATURE_COLS}])
    preds = {}
    for h in MODEL_HORIZONS:
        pred_val = float(models[h].predict(row)[0])
        preds[f"p{h}"] = round(pred_val, 1)

    def classify(val):
        if val >= 13.0:
            return "GOOD", "🟢"
        if val >= 9.5:
            return "WARNING", "🟡"
        return "CRITICAL", "🔴"

    st5, icon5 = classify(preds["p5"])
    st10, icon10 = classify(preds["p10"])
    st20, icon20 = classify(preds["p20"])

    return {
        "p5": preds["p5"], "status_5s": st5, "icon_5s": icon5,
        "p10": preds["p10"], "status_10s": st10, "icon_10s": icon10,
        "p20": preds["p20"], "status_20s": st20, "icon_20s": icon20,
        "history_ready": telemetry.get("ready", False),
    }


def ttg_from_elevation(elevation_deg: float) -> int:
    return max(0, int((elevation_deg - ELEVATION_MASK_DEG) * 8.0))


class EventLog:
    def __init__(self):
        self.events = []

    def log(self, level, msg):
        self.events.insert(0, {"time": time.strftime("%H:%M:%S"), "level": level, "msg": msg})
        self.events = self.events[:15]


event_log = EventLog()
tracker = LiveTracker()
agent = NetworkOperationsAgent(predict_fn=predict_future_snr)

event_log.log("INFO", "System initialized. Ground Station: Chennai Gateway (13.08N, 80.27E)")
event_log.log("INFO", "Learning agent active: adaptive utility weights based on handover outcomes")

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


@app.get("/api/telemetry")
def get_telemetry():
    all_visible = tracker.get_all_visible()
    serving, candidates = agent.observe(all_visible)

    if serving is None:
        return {
            "ground_station": "Chennai Gateway (13.08°N, 80.27°E)",
            "gateway_lat": CHENNAI_LAT, "gateway_lon": CHENNAI_LON,
            "serving_sat": None,
            "message": "No satellites currently visible above elevation mask.",
            "event_log": event_log.events,
        }

    serving_pred = agent.predict(serving)
    ttg_sec = ttg_from_elevation(serving["elevation_deg"])

    scored_candidates = agent.evaluate_candidates(candidates, ttg_from_elevation)

    action, reason, target_name, target_components = agent.plan(serving, serving_pred, scored_candidates, ttg_sec)

    ho_status_map = {
        "KEEP": ("STABLE", "emerald"),
        "PREPARE_HANDOVER": ("WARNING", "amber"),
        "HANDOVER": ("CRITICAL", "red"),
        "HANDOVER_BLOCKED": ("CRITICAL", "red"),
    }
    ho_status, ho_color = ho_status_map.get(action, ("STABLE", "emerald"))
    ho_message = reason

    if action == "HANDOVER" and target_name:
        record = agent.execute_handover(target_name, reason, snr_before=serving["snr_db"], components=target_components)
        event_log.log("SUCCESS", f"Handover executed: {record.from_sat} -> {record.to_sat} ({reason})")
    elif action == "HANDOVER_BLOCKED":
        event_log.log("WARN", f"Handover BLOCKED by safety layer: {reason}")

    memory = agent.get_memory_summary()

    return {
        "ground_station": "Chennai Gateway (13.08°N, 80.27°E)",
        "gateway_lat": CHENNAI_LAT, "gateway_lon": CHENNAI_LON,
        "serving_sat": serving["name"],
        "elevation_deg": round(serving["elevation_deg"], 1),
        "azimuth_deg": round(serving["azimuth_deg"], 1),
        "range_km": round(serving["range_km"], 1),
        "fspl_db": round(serving["fspl_db"], 1),
        "snr_db": round(serving["snr_db"], 1),
        "snr_trend": round(serving.get("snr_delta", 0.0), 2),
        "doppler_hz": round(serving["doppler_hz"], 1),
        "sub_lat": round(serving["sub_lat"], 3),
        "sub_lon": round(serving["sub_lon"], 3),
        "ttg_sec": ttg_sec,
        "handover_status": ho_status,
        "handover_message": ho_message,
        "handover_color": ho_color,
        "agent_action": action,
        "target_sat": target_name if target_name else "N/A",
        "ai_prediction": serving_pred,
        "candidates": [
            {
                "id": c["id"], "elevation": c["elevation"], "snr": c["snr"],
                "predicted_snr_10s": c["predicted_snr_10s"], "utility": c["utility"],
                "sub_lat": round(c["raw"]["sub_lat"], 3), "sub_lon": round(c["raw"]["sub_lon"], 3),
            } for c in scored_candidates
        ],
        "agent_memory": memory,
        "event_log": event_log.events,
        "note": "Predictions unreliable until history_ready=true (~10s after startup)" if not serving_pred["history_ready"] else None,
    }


class ControlSchema(BaseModel):
    action: str


@app.post("/api/control")
def control_simulation(data: ControlSchema):
    event_log.log("INFO", f"Control action received: {data.action} (live agent mode)")
    return {"status": "ok", "mode": "LIVE_LEARNING_AGENT"}


app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def read_root():
    return FileResponse("index.html")