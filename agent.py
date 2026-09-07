"""
NetworkOperationsAgent -- now a LEARNING agent (Option A: adaptive utility
weights), not just a fixed-formula utility-based agent.

Architecture (Russell & Norvig learning-agent framework):
 - Performance element: observe -> predict -> evaluate -> plan -> execute
   (same as before)
 - Critic: after each handover, measures the outcome (did SNR actually
   improve?) using the existing outcome-tracked memory
 - Learning element: adjusts the utility function's weights based on
   the critic's feedback -- factors that were high when a handover
   turned out well get reinforced; factors present during a bad
   outcome get down-weighted
 - Problem generator: NOT implemented (would mean deliberately trying
   suboptimal actions to gather more data -- out of scope; this agent
   only learns from outcomes of decisions it would make anyway)

This is a lightweight, fully explainable learning rule (a single-step
reward-weighted update, similar in spirit to a linear bandit / simple
policy-gradient-style adjustment) -- deliberately NOT full RL (no
value function, no training loop, no exploration policy). It is
scoped to be honestly defensible: every weight change can be traced
to a specific observed outcome.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


# ---- Initial utility function weights (now ADAPTIVE, not fixed) ----
INITIAL_WEIGHTS = {
    "w_snr": 0.45,
    "w_elevation": 0.20,
    "w_visibility": 0.15,
}
HANDOVER_COST_PENALTY = 0.20   # kept fixed -- a safety constant, not learned

# ---- Learning parameters ----
LEARNING_RATE = 0.05
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.70
LEARNABLE_WEIGHT_SUM = sum(INITIAL_WEIGHTS.values())  # renormalize to conserve this total

# ---- Safety / hysteresis constraints (unchanged) ----
HANDOVER_COOLDOWN_SEC = 15
MIN_UTILITY_GAIN_TO_SWITCH = 0.08
MIN_ELEVATION_FOR_CANDIDATE = 10.0


@dataclass
class HandoverRecord:
    timestamp: float
    from_sat: str
    to_sat: str
    reason: str
    snr_before: float
    # Component values (normalized 0-1) that contributed to picking this
    # candidate -- needed so the learning element knows what to credit
    # or blame when the outcome is later measured.
    components: dict = field(default_factory=dict)
    snr_after: Optional[float] = None
    evaluated: bool = False


@dataclass
class AgentState:
    current_serving: Optional[str] = None
    last_handover_time: float = field(default_factory=lambda: 0.0)
    handover_history: list = field(default_factory=list)
    total_handovers: int = 0
    total_polls: int = 0
    weights: dict = field(default_factory=lambda: dict(INITIAL_WEIGHTS))
    weight_history: list = field(default_factory=list)  # for reporting how weights evolved


class NetworkOperationsAgent:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.state = AgentState()
        self._record_weight_snapshot("initial")

    def _record_weight_snapshot(self, trigger: str):
        snap = dict(self.state.weights)
        snap["_trigger"] = trigger
        snap["_time"] = time.time()
        self.state.weight_history.append(snap)
        self.state.weight_history = self.state.weight_history[-50:]

    # ---------------- OBSERVE ----------------
    def observe(self, all_visible: list):
        self.state.total_polls += 1
        visible_by_name = {v["name"]: v for v in all_visible}

        if self.state.current_serving is None or self.state.current_serving not in visible_by_name:
            if not all_visible:
                return None, []
            best = max(all_visible, key=lambda x: x["elevation_deg"])
            self.state.current_serving = best["name"]

        serving = visible_by_name.get(self.state.current_serving)
        candidates = [v for v in all_visible if v["name"] != self.state.current_serving]
        return serving, candidates

    # ---------------- PREDICT ----------------
    def predict(self, telemetry: dict):
        return self.predict_fn(telemetry)

    # ---------------- EVALUATE (utility function, now with adaptive weights) ----------------
    def _utility_components(self, pred_snr_10s: float, elevation_deg: float, ttg_sec: float):
        """Returns the raw normalized components (0-1), before weighting.
        These get stored so the learning element can later credit/blame
        the right factors."""
        return {
            "norm_snr": max(0.0, min(1.0, pred_snr_10s / 25.0)),
            "norm_elevation": max(0.0, min(1.0, elevation_deg / 90.0)),
            "norm_visibility": max(0.0, min(1.0, ttg_sec / 300.0)),
        }

    def _utility(self, components: dict, is_handover: bool):
        w = self.state.weights
        handover_penalty = HANDOVER_COST_PENALTY if is_handover else 0.0
        utility = (
            w["w_snr"] * components["norm_snr"]
            + w["w_elevation"] * components["norm_elevation"]
            + w["w_visibility"] * components["norm_visibility"]
            - handover_penalty
        )
        return round(utility, 4)

    def evaluate_candidates(self, candidates: list, ttg_fn):
        scored = []
        for c in candidates:
            pred = self.predict(c)
            ttg = ttg_fn(c["elevation_deg"])
            components = self._utility_components(pred["p10"], c["elevation_deg"], ttg)
            utility = self._utility(components, is_handover=True)
            scored.append({
                "id": c["name"],
                "elevation": round(c["elevation_deg"], 1),
                "snr": round(c["snr_db"], 1),
                "predicted_snr_10s": pred["p10"],
                "utility": utility,
                "components": components,
                "raw": c,
            })
        scored.sort(key=lambda x: x["utility"], reverse=True)
        return scored

    # ---------------- PLAN + SAFETY CHECK ----------------
    def plan(self, serving: dict, serving_pred: dict, scored_candidates: list, ttg_sec: int):
        serving_components = self._utility_components(serving_pred["p10"], serving["elevation_deg"], ttg_sec)
        current_utility = self._utility(serving_components, is_handover=False)

        best_candidate = scored_candidates[0] if scored_candidates else None

        link_critical = serving_pred["p10"] < 9.5 or serving["elevation_deg"] < 11.0 or ttg_sec <= 10
        link_warning = (not link_critical) and (serving_pred["p10"] < 13.0 or serving["elevation_deg"] < 16.0)

        if not link_critical and not link_warning:
            return "KEEP", "Serving link healthy, no action needed", None, None

        if best_candidate is None:
            return ("HANDOVER_BLOCKED" if link_critical else "KEEP",
                    "Link degrading but no candidate currently visible", None, None)

        time_since_last = time.time() - self.state.last_handover_time
        if time_since_last < HANDOVER_COOLDOWN_SEC:
            return ("PREPARE_HANDOVER" if link_critical else "KEEP",
                    f"Handover cooldown active ({time_since_last:.0f}s < {HANDOVER_COOLDOWN_SEC}s), "
                    f"deferring even though link is degrading", best_candidate["id"], None)

        utility_gain = best_candidate["utility"] - current_utility
        if utility_gain < MIN_UTILITY_GAIN_TO_SWITCH and not link_critical:
            return "KEEP", f"Candidate utility gain too small ({utility_gain:.3f}), not worth switching", None, None

        if best_candidate["elevation"] < MIN_ELEVATION_FOR_CANDIDATE:
            return ("HANDOVER_BLOCKED" if link_critical else "KEEP",
                    "Best candidate below minimum elevation -- rejected by safety layer", None, None)

        if link_critical:
            if utility_gain < 0:
                gain_note = (f"accepting utility loss ({utility_gain:.3f}) because current link "
                             f"is critical -- any visible candidate is safer than losing the link")
            else:
                gain_note = f"utility gain={utility_gain:.3f}"
            return "HANDOVER", (
                f"Link critical (predicted +10s SNR={serving_pred['p10']}dB). "
                f"Switching to {best_candidate['id']}, {gain_note}"
            ), best_candidate["id"], best_candidate["components"]
        else:
            return "PREPARE_HANDOVER", (
                f"Link degrading, preparing handover to {best_candidate['id']}"
            ), best_candidate["id"], best_candidate["components"]

    # ---------------- EXECUTE ----------------
    def execute_handover(self, target_name: str, reason: str, snr_before: float, components: dict):
        old_name = self.state.current_serving
        self._evaluate_last_handover_outcome(snr_before)

        record = HandoverRecord(
            timestamp=time.time(),
            from_sat=old_name,
            to_sat=target_name,
            reason=reason,
            snr_before=snr_before,
            components=components or {},
        )
        self.state.handover_history.append(record)
        self.state.handover_history = self.state.handover_history[-20:]

        self.state.current_serving = target_name
        self.state.last_handover_time = time.time()
        self.state.total_handovers += 1
        return record

    # ---------------- CRITIC + LEARNING ELEMENT ----------------
    def _evaluate_last_handover_outcome(self, current_snr: float):
        """CRITIC: measure the outcome of the previous handover, then
        trigger the LEARNING ELEMENT to adjust weights based on it."""
        for rec in reversed(self.state.handover_history):
            if not rec.evaluated:
                rec.snr_after = current_snr
                rec.evaluated = True
                reward = rec.snr_after - rec.snr_before  # positive = good outcome
                self._learn(rec.components, reward)
                break

    def _learn(self, components: dict, reward: float):
        """LEARNING ELEMENT: reward-weighted update.
        If the outcome was good (reward > 0), increase the weight of
        whichever factor was HIGH at decision time (it was "right" to
        trust that factor). If the outcome was bad (reward < 0), decrease
        weights for factors that were high (they misled the decision).
        This is a simple, fully explainable linear credit-assignment
        rule -- not full RL, but a genuine adaptive learning step.
        """
        if not components:
            return

        # Normalize reward to a small, bounded learning signal so a single
        # extreme outcome can't destabilize the weights.
        signal = max(-1.0, min(1.0, reward / 5.0))  # +/-5dB swing = full-strength update

        w = self.state.weights
        w["w_snr"] += LEARNING_RATE * signal * components.get("norm_snr", 0.0)
        w["w_elevation"] += LEARNING_RATE * signal * components.get("norm_elevation", 0.0)
        w["w_visibility"] += LEARNING_RATE * signal * components.get("norm_visibility", 0.0)

        # Clip to sane bounds
        for k in w:
            w[k] = max(WEIGHT_MIN, min(WEIGHT_MAX, w[k]))

        # Renormalize so the three learnable weights conserve their total
        # (keeps the utility function's overall scale stable and comparable
        # over time -- otherwise weights could all drift upward together)
        total = sum(w.values())
        if total > 0:
            scale = LEARNABLE_WEIGHT_SUM / total
            for k in w:
                w[k] = round(w[k] * scale, 4)

        self._record_weight_snapshot(f"learned (reward={reward:.2f}dB)")

    def get_memory_summary(self):
        evaluated = [r for r in self.state.handover_history if r.evaluated]
        if not evaluated:
            return {
                "total_handovers": self.state.total_handovers,
                "avg_snr_improvement": None,
                "history": [],
                "current_weights": self.state.weights,
                "weight_history": self.state.weight_history,
            }

        improvements = [r.snr_after - r.snr_before for r in evaluated]
        return {
            "total_handovers": self.state.total_handovers,
            "avg_snr_improvement": round(sum(improvements) / len(improvements), 2),
            "history": [
                {
                    "from": r.from_sat, "to": r.to_sat,
                    "snr_before": round(r.snr_before, 1),
                    "snr_after": round(r.snr_after, 1) if r.snr_after is not None else None,
                    "improvement": round(r.snr_after - r.snr_before, 1) if r.snr_after is not None else None,
                    "reason": r.reason,
                }
                for r in evaluated[-10:]
            ],
            "current_weights": self.state.weights,
            "weight_history": self.state.weight_history,
        }