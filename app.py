"""
Flask web app for interactive pitch sequence optimization.

Run: python app.py
Open: http://localhost:5000
"""

import io
import base64
from flask import Flask, render_template, request, jsonify, session

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt

from dynamic_pitcher import (
    load_model, new_game_state, update_state, recommend_next_pitch,
)

# ═══════════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = "pso-dev-key"

# Load model once at startup
(MODEL, ENCODERS, PITCHER_PITCH_AVGS, BATTER_INFO, LE_OUTCOME,
 ZONE_LOCATION_AVGS, PITCHER_INFO, BATTER_STATS,
 LEAGUE_AVG_BATTER_STATS, BATTER_VS_PITCH_TYPE_SPLITS,
 LEAGUE_AVG_BATTER_VS_PITCH, PITCHER_STATS, LEAGUE_AVG_PITCHER_STATS,
 PARK_FACTORS) = load_model()

ZONES = ENCODERS["zone_label"].classes_.tolist()

# Sorted pitcher list for the UI
PITCHERS = sorted(PITCHER_INFO.keys())

# Build team → batter list and sorted team list for the UI
# BATTER_INFO is {name: {"stand": "R", "team": "HOU"}, ...}
TEAMS = sorted({info["team"] for info in BATTER_INFO.values() if info.get("team")})
BATTERS = sorted(BATTER_INFO.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_state():
    """Get game state from session, or create a fresh one."""
    if "game_state" not in session:
        session["game_state"] = new_game_state()
    return session["game_state"]


def save_state(state):
    """Persist game state to session."""
    session["game_state"] = state
    session.modified = True


def get_recommendations(state):
    """Get top 10 recommendations and return as list of dicts."""
    recs = recommend_next_pitch(
        state, MODEL, ENCODERS, PITCHER_PITCH_AVGS, top_n=10,
        le_outcome=LE_OUTCOME,
        zone_location_avgs=ZONE_LOCATION_AVGS,
        pitcher_info=PITCHER_INFO,
        batter_stats=BATTER_STATS,
        league_avg_batter_stats=LEAGUE_AVG_BATTER_STATS,
        batter_vs_pitch_type_splits=BATTER_VS_PITCH_TYPE_SPLITS,
        league_avg_batter_vs_pitch=LEAGUE_AVG_BATTER_VS_PITCH,
        pitcher_stats=PITCHER_STATS,
        league_avg_pitcher_stats=LEAGUE_AVG_PITCHER_STATS,
        park_factors=PARK_FACTORS,
    )
    return recs[["combo", "pitch_type", "zone_label", "re24_score",
                 "called_strike", "whiff", "foul", "out", "hit", "ball"]].to_dict(orient="records")


def make_chart(recs_list):
    """Generate a horizontal bar chart of recommendations, return as base64 PNG."""
    if not recs_list:
        return ""

    combos = [r["combo"] for r in recs_list][::-1]
    scores = [r["re24_score"] for r in recs_list][::-1]

    _, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2563eb" if i < len(combos) - 1 else "#16a34a" for i in range(len(combos))]
    colors = colors[::-1]  # top entry gets green
    ax.barh(combos, scores, color=colors[::-1])
    ax.set_xlabel("RE24 Score", fontsize=11)
    ax.set_title("Pitch Recommendations (RE24)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Render the main page."""
    state = get_state()
    recs = get_recommendations(state)
    chart = make_chart(recs)

    # Pitch types for the currently selected pitcher
    pitcher_name = state.get("pitcher_name", "nathan eovaldi")
    pitch_types = PITCHER_INFO.get(pitcher_name, {}).get("pitch_types", [])

    return render_template(
        "index.html",
        state=state,
        recs=recs,
        chart=chart,
        pitch_types=pitch_types,
        zones=ZONES,
        teams=TEAMS,
        batters=BATTERS,
        batter_info=BATTER_INFO,
        pitchers=PITCHERS,
        pitcher_info=PITCHER_INFO,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    """AJAX: update game state from controls, return new recommendations."""
    data = request.get_json()
    state = get_state()

    # Update state from UI controls
    state["balls"]   = int(data.get("balls", state["balls"]))
    state["strikes"] = int(data.get("strikes", state["strikes"]))
    state["outs"]    = int(data.get("outs", state["outs"]))
    state["inning"]  = int(data.get("inning", state["inning"]))
    state["on_1b"]   = data.get("on_1b", state["on_1b"])
    state["on_2b"]   = data.get("on_2b", state["on_2b"])
    state["on_3b"]   = data.get("on_3b", state["on_3b"])

    # Update pitcher
    pitcher = data.get("pitcher_name", state.get("pitcher_name", "nathan eovaldi"))
    state["pitcher_name"] = pitcher

    # Update batter — look up handedness automatically
    batter = data.get("batter_name", state.get("batter_name", "unknown"))
    state["batter_name"] = batter
    info = BATTER_INFO.get(batter, {})
    state["stand"] = info.get("stand", state.get("stand", "R"))

    # New situational fields
    state["score_differential"] = int(data.get("score_differential", state.get("score_differential", 0)))
    state["home_team"] = data.get("home_team", state.get("home_team", ""))

    save_state(state)

    recs = get_recommendations(state)
    chart = make_chart(recs)

    return jsonify({"recs": recs, "chart": chart, "state": state})


@app.route("/simulate", methods=["POST"])
def simulate():
    """AJAX: simulate a pitch, update state, return new state + recommendations."""
    data = request.get_json()
    state = get_state()

    pitch_type = data["pitch_type"]
    zone_label = data["zone_label"]
    outcome    = data["outcome"]

    # Record count before the pitch for history
    count_before = f"{state['balls']}-{state['strikes']}"

    state = update_state(state, outcome, pitch_type, zone_label)
    save_state(state)

    recs = get_recommendations(state)
    chart = make_chart(recs)

    # Build history entry
    count_after = f"{state['balls']}-{state['strikes']}"
    history_entry = {
        "pitch_type": pitch_type,
        "zone_label": zone_label,
        "outcome": outcome,
        "count_before": count_before,
        "count_after": count_after,
    }

    return jsonify({
        "recs": recs,
        "chart": chart,
        "state": state,
        "history_entry": history_entry,
    })


@app.route("/reset", methods=["POST"])
def reset():
    """AJAX: reset to a fresh game state."""
    data = request.get_json() or {}

    pitcher = data.get("pitcher_name", "nathan eovaldi")
    batter = data.get("batter_name", "unknown")
    info = BATTER_INFO.get(batter, {})
    stand = info.get("stand", "R")
    home_team = data.get("home_team", "")

    state = new_game_state(
        batter_name=batter,
        batter_stand=stand,
        pitcher_name=pitcher,
        home_team=home_team,
    )
    save_state(state)

    # Clear pitch history in session
    session.pop("pitch_history", None)

    recs = get_recommendations(state)
    chart = make_chart(recs)

    return jsonify({"recs": recs, "chart": chart, "state": state})


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading model...")
    print(f"Pitchers: {len(PITCHERS)}")
    print(f"Zones: {ZONES}")
    print("Starting server at http://localhost:5000")
    app.run(debug=False, port=5000)
