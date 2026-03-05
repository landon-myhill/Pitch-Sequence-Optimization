"""
Flask web app for interactive pitch sequence optimization.

Run: python app.py
Open: http://localhost:5000
"""

import json
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
MODEL, ENCODERS, PITCH_AVGS = load_model()
PITCH_TYPES = list(PITCH_AVGS.keys())
ZONES = ENCODERS["zone_label"].classes_.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_state():
    """Get game state from session, or create a fresh one."""
    if "game_state" not in session:
        session["game_state"] = new_game_state("R")
    return session["game_state"]


def save_state(state):
    """Persist game state to session."""
    session["game_state"] = state
    session.modified = True


def get_recommendations(state):
    """Get top 10 recommendations and return as list of dicts."""
    recs = recommend_next_pitch(state, MODEL, ENCODERS, PITCH_AVGS, top_n=10)
    return recs[["combo", "pitch_type", "zone_label", "re24_score",
                 "strike", "out", "hit", "ball"]].to_dict(orient="records")


def make_chart(recs_list):
    """Generate a horizontal bar chart of recommendations, return as base64 PNG."""
    if not recs_list:
        return ""

    combos = [r["combo"] for r in recs_list][::-1]
    scores = [r["re24_score"] for r in recs_list][::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2563eb" if i < len(combos) - 1 else "#16a34a" for i in range(len(combos))]
    colors = colors[::-1]  # top entry gets green
    ax.barh(combos, scores, color=colors[::-1])
    ax.set_xlabel("RE24 Score", fontsize=11)
    ax.set_title("Pitch Recommendations (RE24)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
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
    return render_template(
        "index.html",
        state=state,
        recs=recs,
        chart=chart,
        pitch_types=PITCH_TYPES,
        zones=ZONES,
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
    state["stand"]   = data.get("stand", state["stand"])
    state["on_1b"]   = data.get("on_1b", state["on_1b"])
    state["on_2b"]   = data.get("on_2b", state["on_2b"])
    state["on_3b"]   = data.get("on_3b", state["on_3b"])

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
    stand = data.get("stand", "R")

    state = new_game_state(batter_stand=stand)
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
    print(f"Pitch types: {PITCH_TYPES}")
    print(f"Zones: {ZONES}")
    print("Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)
