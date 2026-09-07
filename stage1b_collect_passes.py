"""
Stage 1b: Collect MULTIPLE real satellite passes over Chennai.
One pass isn't enough data to train/test a model properly.
"""

from skyfield.api import load, wgs84
import math
import pandas as pd
import numpy as np

np.random.seed(42) 

CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707
CHENNAI_ELEV_M = 6.0
CARRIER_FREQ_HZ = 2.0e9
SPEED_OF_LIGHT = 299792458.0
MIN_ELEVATION_DEG = 10.0
STEP_SECONDS = 2

TARGET_NUM_PASSES = 25
SEARCH_WINDOW_HOURS = 72
MAX_SATS_TO_SCAN = 800

EIRP_DBW = 20.0
RX_GAIN_DB = 34.0
NOISE_POWER_DBW = -116.0

print("Loading TLE data...")
stations_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
satellites = load.tle_file(stations_url)
print(f"Loaded {len(satellites)} Starlink satellites")

ts = load.timescale()
chennai = wgs84.latlon(CHENNAI_LAT, CHENNAI_LON, elevation_m=CHENNAI_ELEV_M)


def find_passes_for_sat(sat, t_start, window_hours, max_passes=3):
    t_end = ts.tt_jd(t_start.tt + window_hours / 24.0)
    try:
        times, events = sat.find_events(chennai, t_start, t_end, altitude_degrees=MIN_ELEVATION_DEG)
    except Exception:
        return []
    passes = []
    for i in range(len(events) - 2):
        if events[i] == 0 and events[i+1] == 1 and events[i+2] == 2:
            passes.append((times[i], times[i+2]))
            if len(passes) >= max_passes:
                break
    return passes


def propagate_pass(sat, rise_t, set_t, pass_id):
    duration_sec = (set_t.tt - rise_t.tt) * 86400.0
    n_steps = int(duration_sec // STEP_SECONDS)

    rows = []
    prev_range_km = None

    for i in range(n_steps + 1):
        t = ts.tt_jd(rise_t.tt + (i * STEP_SECONDS) / 86400.0)
        difference = sat - chennai
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()

        elevation_deg = alt.degrees
        azimuth_deg = az.degrees
        range_km = distance.km

        if prev_range_km is not None:
            range_rate_km_s = (range_km - prev_range_km) / STEP_SECONDS
            radial_velocity_m_s = range_rate_km_s * 1000.0
            doppler_hz = -(radial_velocity_m_s / SPEED_OF_LIGHT) * CARRIER_FREQ_HZ
        else:
            doppler_hz = 0.0

        d_m = range_km * 1000.0
        fspl_db = 20 * math.log10(d_m) + 20 * math.log10(CARRIER_FREQ_HZ) - 147.55

        pr_dbw = EIRP_DBW - fspl_db + RX_GAIN_DB
        snr_clean = pr_dbw - NOISE_POWER_DBW

        # Realistic stochastic variation: atmospheric scintillation +
        # receiver noise jitter. Without this, SNR is a pure deterministic
        # function of range, making prediction trivially easy (R^2 ~0.999)
        # and unrepresentative of a real NTN link.
        scintillation_std_db = 0.35
        receiver_noise_std_db = 0.15
        noise = np.random.normal(0, scintillation_std_db) + np.random.normal(0, receiver_noise_std_db)

        snr_db = snr_clean + noise
        snr_db = max(0.0, min(30.0, snr_db))

        rows.append({
            "pass_id": pass_id,
            "satellite": sat.name,
            "t_sec": i * STEP_SECONDS,
            "elevation_deg": elevation_deg,
            "azimuth_deg": azimuth_deg,
            "range_km": range_km,
            "doppler_hz": doppler_hz,
            "fspl_db": fspl_db,
            "snr_db": snr_db
        })
        prev_range_km = range_km

    return rows


t0 = ts.now()
all_rows = []
pass_count = 0
pass_summaries = []

print(f"\nScanning up to {MAX_SATS_TO_SCAN} satellites for passes in next {SEARCH_WINDOW_HOURS}h...")

for sat in satellites[:MAX_SATS_TO_SCAN]:
    if pass_count >= TARGET_NUM_PASSES:
        break
    passes = find_passes_for_sat(sat, t0, SEARCH_WINDOW_HOURS, max_passes=1)
    for rise_t, set_t in passes:
        pass_count += 1
        pass_id = f"pass_{pass_count:02d}"
        rows = propagate_pass(sat, rise_t, set_t, pass_id)
        all_rows.extend(rows)

        max_elev = max(r["elevation_deg"] for r in rows)
        pass_summaries.append({
            "pass_id": pass_id,
            "satellite": sat.name,
            "n_points": len(rows),
            "max_elevation": round(max_elev, 1),
            "duration_sec": rows[-1]["t_sec"]
        })
        print(f"  {pass_id}: {sat.name}, max_elev={max_elev:.1f}°, {len(rows)} points")

        if pass_count >= TARGET_NUM_PASSES:
            break

if not all_rows:
    raise RuntimeError("No passes found — widen SEARCH_WINDOW_HOURS or MAX_SATS_TO_SCAN")

df = pd.DataFrame(all_rows)
df.to_csv("all_passes.csv", index=False)

print(f"\n{'='*50}")
print(f"Collected {pass_count} passes, {len(df)} total rows")
print(f"Saved to all_passes.csv")
print(f"{'='*50}")
print(pd.DataFrame(pass_summaries).to_string(index=False))
print(f"\nOverall SNR range: {df['snr_db'].min():.1f} - {df['snr_db'].max():.1f} dB")
print(f"Overall elevation range: {df['elevation_deg'].min():.1f} - {df['elevation_deg'].max():.1f}°")