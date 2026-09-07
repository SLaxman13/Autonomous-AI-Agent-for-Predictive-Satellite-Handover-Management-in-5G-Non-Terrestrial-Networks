"""
Stage 1: Real orbital data pipeline
- Downloads real Starlink TLE data
- Picks one satellite
- Finds a real visible pass over Chennai
- Propagates position every 2 seconds through that pass
- Computes elevation, distance, Doppler, FSPL, SNR at each timestep
- Saves as a CSV time-series (this is our real dataset for one pass)
"""

from skyfield.api import load, wgs84, EarthSatellite
import math
import pandas as pd
import numpy as np

# ---- CONFIG ----
CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707
CHENNAI_ELEV_M = 6.0
CARRIER_FREQ_HZ = 2.0e9
SPEED_OF_LIGHT = 299792458.0
MIN_ELEVATION_DEG = 10.0
STEP_SECONDS = 2

# ---- STEP A: Load real TLE data ----
stations_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
satellites = load.tle_file(stations_url)
print(f"Loaded {len(satellites)} Starlink satellites from Celestrak")

ts = load.timescale()
chennai = wgs84.latlon(CHENNAI_LAT, CHENNAI_LON, elevation_m=CHENNAI_ELEV_M)

# ---- STEP B: Find a satellite with a good upcoming pass ----
t0 = ts.now()
search_window_hours = 6

def find_pass(sat, t_start, window_hours):
    t_end = ts.tt_jd(t_start.tt + window_hours / 24.0)
    times, events = sat.find_events(chennai, t_start, t_end, altitude_degrees=MIN_ELEVATION_DEG)
    for i in range(len(events) - 2):
        if events[i] == 0 and events[i+1] == 1 and events[i+2] == 2:
            return times[i], times[i+2]
    return None, None

chosen_sat = None
rise_t, set_t = None, None

for sat in satellites[:200]:
    try:
        r, s = find_pass(sat, t0, search_window_hours)
        if r is not None:
            chosen_sat = sat
            rise_t, set_t = r, s
            break
    except Exception:
        continue

if chosen_sat is None:
    raise RuntimeError("No valid pass found — widen search_window_hours or satellite range")

print(f"Chosen satellite: {chosen_sat.name}")
print(f"Pass window: {rise_t.utc_iso()} -> {set_t.utc_iso()}")

# ---- STEP C: Propagate every 2 seconds through the pass ----
duration_sec = (set_t.tt - rise_t.tt) * 86400.0
n_steps = int(duration_sec // STEP_SECONDS)
print(f"Pass duration: {duration_sec:.0f} sec, {n_steps} steps at {STEP_SECONDS}s resolution")

rows = []
prev_range_km = None

for i in range(n_steps + 1):
    t = ts.tt_jd(rise_t.tt + (i * STEP_SECONDS) / 86400.0)
    difference = chosen_sat - chennai
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()

    elevation_deg = alt.degrees
    azimuth_deg = az.degrees
    range_km = distance.km

    if prev_range_km is not None:
        dt = STEP_SECONDS
        range_rate_km_s = (range_km - prev_range_km) / dt
        radial_velocity_m_s = range_rate_km_s * 1000.0
        doppler_hz = -(radial_velocity_m_s / SPEED_OF_LIGHT) * CARRIER_FREQ_HZ
    else:
        doppler_hz = 0.0

    d_m = range_km * 1000.0
    fspl_db = 20 * math.log10(d_m) + 20 * math.log10(CARRIER_FREQ_HZ) - 147.55

    eirp_dbw = 20.0
    rx_gain_db = 34.0
    noise_power_dbw = -116.0
    pr_dbw = eirp_dbw - fspl_db + rx_gain_db
    snr_db = pr_dbw - noise_power_dbw
    snr_db = max(0.0, min(30.0, snr_db))

    rows.append({
        "t_sec": i * STEP_SECONDS,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
        "range_km": range_km,
        "doppler_hz": doppler_hz,
        "fspl_db": fspl_db,
        "snr_db": snr_db
    })

    prev_range_km = range_km

df = pd.DataFrame(rows)
df.to_csv("pass_data.csv", index=False)
print(f"\nSaved {len(df)} rows to pass_data.csv")
print(df.head(10))
print("...")
print(df.describe())