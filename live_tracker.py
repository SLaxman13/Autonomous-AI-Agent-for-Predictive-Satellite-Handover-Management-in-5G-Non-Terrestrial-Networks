"""
Stage 4a: Live multi-satellite tracking module
- Loads real Starlink TLE data once at startup
- Returns all currently visible satellites with full telemetry
- Maintains a short rolling SNR history per satellite for trend features
- Includes real ground-track subpoint (lat/lon) for map display
- app.py manages which satellite is "serving" (locked) vs "candidate"
"""

from skyfield.api import load, wgs84
import math
import time as time_module
import numpy as np
from collections import defaultdict, deque

# ---- CONFIG (must match training data generation) ----
CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707
CHENNAI_ELEV_M = 6.0
CARRIER_FREQ_HZ = 2.0e9
SPEED_OF_LIGHT = 299792458.0
MIN_ELEVATION_DEG = 10.0

EIRP_DBW = 20.0
RX_GAIN_DB = 34.0
NOISE_POWER_DBW = -116.0

SCINTILLATION_STD_DB = 0.35
RECEIVER_NOISE_STD_DB = 0.15

HISTORY_WINDOW = 5   # samples kept for trend features (matches training)
TOTAL_SATS_TO_TRACK = 1000   # spread across the full constellation for orbital-plane diversity


class LiveTracker:
    def __init__(self):
        print("Loading TLE data for live tracker...")
        stations_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
        all_sats = load.tle_file(stations_url)
        # Sample spread across the FULL list (not just the first N) so we
        # cover multiple orbital planes -- consecutive TLE entries are from
        # the same launch batch and cluster together in the sky.
        step = max(1, len(all_sats) // TOTAL_SATS_TO_TRACK)
        self.satellites = all_sats[::step][:TOTAL_SATS_TO_TRACK]
        print(f"Live tracker loaded {len(self.satellites)} satellites")

        self.ts = load.timescale()
        self.chennai = wgs84.latlon(CHENNAI_LAT, CHENNAI_LON, elevation_m=CHENNAI_ELEV_M)

        self.history = defaultdict(lambda: deque(maxlen=HISTORY_WINDOW + 1))
        self.prev_range = {}
        self.prev_time = {}   # real wall-clock time of last measurement, per satellite

    def _compute_telemetry(self, sat, t):
        difference = sat - self.chennai
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()

        elevation_deg = alt.degrees
        azimuth_deg = az.degrees
        range_km = distance.km

        # Real ground-track subpoint (lat/lon) for map display -- computed
        # from skyfield's own geocentric position, so it's genuine orbital
        # data, not a placeholder or decorative marker.
        geocentric = sat.at(t)
        subpoint = wgs84.subpoint(geocentric)
        sub_lat = subpoint.latitude.degrees
        sub_lon = subpoint.longitude.degrees

        name = sat.name
        now_wall = time_module.time()
        prev_r = self.prev_range.get(name)
        prev_t = self.prev_time.get(name)

        if prev_r is not None and prev_t is not None:
            dt = now_wall - prev_t
            if dt > 0.1:
                range_rate_km_s = (range_km - prev_r) / dt
                radial_velocity_m_s = range_rate_km_s * 1000.0
                doppler_hz = -(radial_velocity_m_s / SPEED_OF_LIGHT) * CARRIER_FREQ_HZ
            else:
                doppler_hz = 0.0
        else:
            doppler_hz = 0.0

        self.prev_range[name] = range_km
        self.prev_time[name] = now_wall

        d_m = range_km * 1000.0
        fspl_db = 20 * math.log10(d_m) + 20 * math.log10(CARRIER_FREQ_HZ) - 147.55

        pr_dbw = EIRP_DBW - fspl_db + RX_GAIN_DB
        snr_clean = pr_dbw - NOISE_POWER_DBW
        noise = np.random.normal(0, SCINTILLATION_STD_DB) + np.random.normal(0, RECEIVER_NOISE_STD_DB)
        snr_db = max(0.0, min(30.0, snr_clean + noise))

        return {
            "name": name,
            "elevation_deg": elevation_deg,
            "azimuth_deg": azimuth_deg,
            "range_km": range_km,
            "doppler_hz": doppler_hz,
            "fspl_db": fspl_db,
            "snr_db": snr_db,
            "sub_lat": sub_lat,
            "sub_lon": sub_lon,
        }

    def _trend_features(self, name, current):
        hist = self.history[name]
        hist.append((current["snr_db"], current["elevation_deg"]))

        if len(hist) > HISTORY_WINDOW:
            snr_past, elev_past = hist[0]
            snr_now, elev_now = hist[-1]
            window_sec = HISTORY_WINDOW * 2

            snrs = [h[0] for h in hist]
            return {
                "snr_delta": snr_now - snr_past,
                "snr_slope": (snr_now - snr_past) / window_sec,
                "elev_delta": elev_now - elev_past,
                "elev_slope": (elev_now - elev_past) / window_sec,
                "snr_rolling_mean": float(np.mean(snrs)),
                "snr_rolling_std": float(np.std(snrs)),
                "ready": True,
            }
        else:
            return {
                "snr_delta": 0.0, "snr_slope": 0.0,
                "elev_delta": 0.0, "elev_slope": 0.0,
                "snr_rolling_mean": current["snr_db"], "snr_rolling_std": 0.0,
                "ready": False,
            }

    def get_visible_satellites(self):
        """Kept for standalone testing. app.py uses get_all_visible() instead."""
        visible = self.get_all_visible()
        if not visible:
            return None, []
        visible_sorted = sorted(visible, key=lambda x: x["elevation_deg"], reverse=True)
        return visible_sorted[0], visible_sorted[1:]

    def get_all_visible(self):
        """Returns a flat list of ALL currently visible satellites with full telemetry."""
        t = self.ts.now()
        visible = []

        for sat in self.satellites:
            telem = self._compute_telemetry(sat, t)
            if telem["elevation_deg"] >= MIN_ELEVATION_DEG:
                trend = self._trend_features(telem["name"], telem)
                telem.update(trend)
                visible.append(telem)

        return visible


if __name__ == "__main__":
    tracker = LiveTracker()
    print("\nPolling live visible satellites...")
    serving, candidates = tracker.get_visible_satellites()
    if serving:
        print(f"Serving: {serving['name']}, elev={serving['elevation_deg']:.1f}, snr={serving['snr_db']:.1f}, "
              f"subpoint=({serving['sub_lat']:.2f}, {serving['sub_lon']:.2f})")
        print(f"Candidates visible: {len(candidates)}")
    else:
        print("No satellites currently visible above elevation mask.")