"""
Interactive game simulation for real-time pitch recommendations.

Tracks full game state including batter handedness, pitch sequencing,
and scores recommendations using RE24 run expectancy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from itertools import product

from utils import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    compute_re24_score,
)
from pitcher_pso import encode_categoricals, build_feature_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(path="model_for_pitcher_nathan_eovaldi.pkl"):
    """Load trained model artifact."""
    artifact = joblib.load(path)
    return (
        artifact["model"],
        artifact["label_encoders"],
        artifact["eovaldi_pitch_avgs"],
        artifact.get("batter_info", {}),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Game state
# ═══════════════════════════════════════════════════════════════════════════════

def new_game_state(batter_name="unknown", batter_stand="R"):
    """Initialize a fresh game state."""
    return {
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "inning": 1,
        "on_1b": False,
        "on_2b": False,
        "on_3b": False,
        "batter_name": batter_name,
        "stand": batter_stand,
        "prev_pitch_type": "none",
        "prev_zone_label": "none",
        "pitch_history": [],
        "batters_faced": 0,
        "times_through_order": 1,
    }


def update_state(state, outcome, pitch_type, zone_label):
    """
    Update game state after a pitch.

    Handles count changes, outs, and resets at-bat sequencing
    when a plate appearance ends.
    """
    # Record this pitch
    state["pitch_history"].append({
        "pitch_type": pitch_type,
        "zone_label": zone_label,
        "outcome": outcome,
    })
    state["prev_pitch_type"] = pitch_type
    state["prev_zone_label"] = zone_label

    # Apply outcome
    if outcome == "strike":
        state["strikes"] += 1
        if state["strikes"] >= 3:
            _end_plate_appearance(state, is_out=True)
    elif outcome == "ball":
        state["balls"] += 1
        if state["balls"] >= 4:
            _end_plate_appearance(state, is_out=False)
    elif outcome == "out":
        _end_plate_appearance(state, is_out=True)
    elif outcome == "hit":
        _end_plate_appearance(state, is_out=False)

    # Full turn through the order
    if state["batters_faced"] >= 9:
        state["times_through_order"] = min(state["times_through_order"] + 1, 4)
        state["batters_faced"] = 0

    return state


def _end_plate_appearance(state, is_out):
    """Reset count and sequencing for the next batter."""
    if is_out:
        state["outs"] = min(state["outs"] + 1, 3)
    state["balls"] = 0
    state["strikes"] = 0
    state["batters_faced"] += 1
    state["prev_pitch_type"] = "none"
    state["prev_zone_label"] = "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_next_pitch(state, model, label_encoders, pitch_avgs, top_n=10):
    """
    Generate all pitch x zone combos for the current state,
    predict outcome probabilities, score with RE24, return top N.
    """
    pitch_types = list(pitch_avgs.keys())
    zones = label_encoders["zone_label"].classes_.tolist()

    # Build scenario DataFrame
    scenario = pd.DataFrame(
        list(product(pitch_types, zones)),
        columns=["pitch_type", "zone_label"],
    )

    # Fill in current game state
    scenario["balls"]           = str(state["balls"])
    scenario["strikes"]         = str(state["strikes"])
    scenario["outs_when_up"]    = str(state["outs"])
    scenario["inning"]          = str(state["inning"])
    scenario["on_1b_occupied"]  = str(state["on_1b"])
    scenario["on_2b_occupied"]  = str(state["on_2b"])
    scenario["on_3b_occupied"]  = str(state["on_3b"])
    scenario["batter_name"]     = state.get("batter_name", "unknown")
    scenario["stand"]           = state["stand"]
    scenario["p_throws"]        = "R"  # Eovaldi
    scenario["prev_pitch_type"] = state["prev_pitch_type"]
    scenario["prev_zone_label"] = state["prev_zone_label"]

    # Eovaldi's pitch physics
    for col in NUMERIC_FEATURES:
        scenario[col] = scenario["pitch_type"].apply(
            lambda pt: pitch_avgs.get(pt, {}).get(col, 0.0)
        )

    # Encode & predict
    encoded, _ = encode_categoricals(scenario, label_encoders=label_encoders, fit=False)
    X = build_feature_matrix(encoded)
    probs = model.predict_proba(X)
    prob_df = pd.DataFrame(probs, columns=model.classes_)

    # Attach probabilities
    outcome_cols = ("called_strike", "whiff", "foul", "out", "hit", "ball", "walk_hbp")
    for col in outcome_cols:
        scenario[col] = prob_df[col] if col in prob_df.columns else 0.0

    # Score with RE24 (fully context-aware)
    scenario["re24_score"] = scenario.apply(
        lambda r: compute_re24_score(
            {"called_strike": r["called_strike"], "whiff": r["whiff"],
             "foul": r["foul"], "ball": r["ball"], "out": r["out"],
             "hit": r["hit"], "walk_hbp": r.get("walk_hbp", 0)},
            balls=state["balls"], strikes=state["strikes"], outs=state["outs"],
            on_1b=state["on_1b"], on_2b=state["on_2b"], on_3b=state["on_3b"],
        ),
        axis=1,
    )
    scenario["combo"] = scenario["pitch_type"] + "_" + scenario["zone_label"]

    return scenario.nlargest(top_n, "re24_score")


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_recommendations(recs, title):
    """Horizontal bar chart of pitch recommendations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(recs["combo"], recs["re24_score"], color="steelblue")
    ax.set_xlabel("RE24 Score (Run Prevention Value)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def print_state(state):
    """Print a human-readable game state summary."""
    print(
        f"  Count: {state['balls']}-{state['strikes']} | "
        f"Outs: {state['outs']} | Inning: {state['inning']} | "
        f"Batter: {state['stand']}HB | "
        f"Prev: {state['prev_pitch_type']} @ {state['prev_zone_label']} | "
        f"TTO: {state['times_through_order']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, encoders, pitch_avgs, batter_info = load_model()

    # ── Scenario 1: vs RHB, 0-0 count ────────────────────────────────────
    game = new_game_state(batter_stand="R")
    print("=== Initial state (vs RHB) ===")
    print_state(game)

    recs = recommend_next_pitch(game, model, encoders, pitch_avgs)
    print("\nTop 10 Recommendations:")
    print(recs[["combo", "re24_score", "strike", "out", "hit"]].to_string(index=False))
    plot_recommendations(recs, "Eovaldi vs RHB — 0-0 count")

    # ── Simulate: SL at middle_right → strike ─────────────────────────────
    game = update_state(game, "strike", "SL", "middle_right")
    print("\n=== After SL @ middle_right → strike ===")
    print_state(game)

    recs = recommend_next_pitch(game, model, encoders, pitch_avgs)
    print("\nTop 10 Recommendations:")
    print(recs[["combo", "re24_score", "strike", "out", "hit"]].to_string(index=False))
    plot_recommendations(
        recs,
        f"After SL → strike | {game['balls']}-{game['strikes']} count",
    )

    # ── Scenario 2: vs LHB ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    game_lhb = new_game_state(batter_stand="L")
    print("=== Fresh AB vs LHB ===")
    print_state(game_lhb)

    recs_lhb = recommend_next_pitch(game_lhb, model, encoders, pitch_avgs)
    print("\nTop 10 Recommendations:")
    print(recs_lhb[["combo", "re24_score", "strike", "out", "hit"]].to_string(index=False))
    plot_recommendations(recs_lhb, "Eovaldi vs LHB — 0-0 count")
