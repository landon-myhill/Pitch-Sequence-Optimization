"""
Interactive game simulation for real-time pitch recommendations.

Tracks full game state including batter handedness, pitch sequencing,
and scores recommendations using RE24 run expectancy.
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from itertools import product

from utils import compute_re24_score, apply_location_penalties
from pitcher_pso import encode_categoricals, build_feature_matrix, PHYSICS_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(path="pitch_sequence_model.pkl"):
    """
    Load trained model artifact.

    Returns:
        (model, label_encoders, pitcher_pitch_avgs, batter_info, le_outcome,
         zone_location_avgs, pitcher_info, batter_stats, league_avg_batter_stats,
         batter_vs_pitch_type_splits, league_avg_batter_vs_pitch,
         pitcher_stats, league_avg_pitcher_stats, park_factors)
    """
    artifact = joblib.load(path)
    return (
        artifact["model"],
        artifact["label_encoders"],
        artifact["pitcher_pitch_avgs"],
        artifact.get("batter_info", {}),
        artifact.get("le_outcome"),
        artifact.get("zone_location_avgs", {}),
        artifact.get("pitcher_info", {}),
        artifact.get("batter_stats", {}),
        artifact.get("league_avg_batter_stats", {}),
        artifact.get("batter_vs_pitch_type_splits", {}),
        artifact.get("league_avg_batter_vs_pitch", {}),
        artifact.get("pitcher_stats", {}),
        artifact.get("league_avg_pitcher_stats", {}),
        artifact.get("park_factors", {}),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Game state
# ═══════════════════════════════════════════════════════════════════════════════

def new_game_state(batter_name="unknown", batter_stand="R",
                   pitcher_name="nathan eovaldi", home_team=""):
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
        "pitcher_name": pitcher_name,
        "prev_pitch_type": "none",
        "prev_zone_label": "none",
        "pitch_history": [],
        "batters_faced": 0,
        "times_through_order": 1,
        # new fields
        "pitch_number_in_ab": 1,
        "pitcher_pitches_today": 0,
        "score_differential": 0,
        "home_team": home_team,
    }


def update_state(state, outcome, pitch_type, zone_label):
    """
    Update game state after a pitch.

    Handles count changes, outs, and resets at-bat sequencing
    when a plate appearance ends.
    """
    # Track per-game and per-AB pitch counts
    state["pitcher_pitches_today"] = state.get("pitcher_pitches_today", 0) + 1
    state["pitch_number_in_ab"] = state.get("pitch_number_in_ab", 1)

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
    state["pitch_number_in_ab"] = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_next_pitch(
    state, model, label_encoders, pitcher_pitch_avgs, top_n=10,
    le_outcome=None, zone_location_avgs=None, pitcher_info=None,
    batter_stats=None, league_avg_batter_stats=None,
    batter_vs_pitch_type_splits=None, league_avg_batter_vs_pitch=None,
    pitcher_stats=None, league_avg_pitcher_stats=None,
    park_factors=None,
):
    """
    Generate all pitch x zone combos for the current state,
    predict outcome probabilities, score with RE24, return top N.

    Args:
        state:                      game state dict (must include 'pitcher_name').
        model:                      trained XGBClassifier.
        label_encoders:             dict of LabelEncoders for categorical features.
        pitcher_pitch_avgs:         {pitcher_name: {pitch_type: {feature: value}}}
        top_n:                      number of recommendations to return.
        le_outcome:                 LabelEncoder for outcome classes.
        zone_location_avgs:         {pitch_type: {zone_label: {plate_x, plate_z}}}
        pitcher_info:               {pitcher_name: {p_throws, pitch_types, ...}}
        batter_stats:               {batter_name: {stat: value}}
        league_avg_batter_stats:    {stat: value}  (fallback when batter not found)
        batter_vs_pitch_type_splits:{batter_name: {pitch_type: {stat: value}}}
        league_avg_batter_vs_pitch: {stat: value}  (fallback)
        pitcher_stats:              {pitcher_name: {stat: value}}
        league_avg_pitcher_stats:   {stat: value}  (fallback)
        park_factors:               {home_team: factor}
    """
    if zone_location_avgs is None:
        zone_location_avgs = {}
    if pitcher_info is None:
        pitcher_info = {}
    if batter_stats is None:
        batter_stats = {}
    if league_avg_batter_stats is None:
        league_avg_batter_stats = {}
    if batter_vs_pitch_type_splits is None:
        batter_vs_pitch_type_splits = {}
    if league_avg_batter_vs_pitch is None:
        league_avg_batter_vs_pitch = {}
    if pitcher_stats is None:
        pitcher_stats = {}
    if league_avg_pitcher_stats is None:
        league_avg_pitcher_stats = {}
    if park_factors is None:
        park_factors = {}

    pitcher_name = state.get("pitcher_name", "nathan eovaldi")

    # Look up this pitcher's pitch avgs; fall back to eovaldi or empty
    pitch_avgs = pitcher_pitch_avgs.get(
        pitcher_name,
        pitcher_pitch_avgs.get("nathan eovaldi", {}),
    )

    # Pitch types and handedness from pitcher_info
    p_info = pitcher_info.get(pitcher_name, {})
    p_throws = p_info.get("p_throws", "R")
    pitch_types = p_info.get("pitch_types", list(pitch_avgs.keys()))

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
    scenario["stand"]           = state.get("stand", "R")
    scenario["p_throws"]        = p_throws
    scenario["prev_pitch_type"] = state["prev_pitch_type"]
    scenario["prev_zone_label"] = state["prev_zone_label"]

    # Pitch physics (release_speed, release_spin_rate, pfx_x, pfx_z)
    for col in PHYSICS_FEATURES:
        scenario[col] = scenario["pitch_type"].apply(
            lambda pt: pitch_avgs.get(pt, {}).get(col, 0.0)
        )

    # plate_x / plate_z from zone_location_avgs
    scenario["plate_x"] = scenario.apply(
        lambda r: zone_location_avgs.get(r["pitch_type"], {})
                                    .get(r["zone_label"], {})
                                    .get("plate_x", 0.0),
        axis=1,
    )
    scenario["plate_z"] = scenario.apply(
        lambda r: zone_location_avgs.get(r["pitch_type"], {})
                                    .get(r["zone_label"], {})
                                    .get("plate_z", 0.0),
        axis=1,
    )

    # Batter aggregate stats — look up by name, fall back to league average
    batter_name = state.get("batter_name", "unknown")
    b_stats = batter_stats.get(batter_name, league_avg_batter_stats)
    for stat in ("batter_k_pct", "batter_bb_pct", "batter_whiff_pct", "batter_hard_hit_pct"):
        scenario[stat] = b_stats.get(stat, 0.0)

    # New situation features
    scenario["pitch_number"] = float(state.get("pitch_number_in_ab", 1))
    scenario["score_differential"] = float(state.get("score_differential", 0))
    scenario["pitcher_pitches_today"] = float(state.get("pitcher_pitches_today", 0))

    # Pitcher days rest from pitcher_info
    p_days_rest = p_info.get("pitcher_days_rest_avg", 5.0)
    scenario["pitcher_days_rest"] = p_days_rest

    # Pitcher season stats
    p_k_pct = p_info.get("pitcher_k_pct", league_avg_pitcher_stats.get("pitcher_k_pct", 0.0))
    p_bb_pct = p_info.get("pitcher_bb_pct", league_avg_pitcher_stats.get("pitcher_bb_pct", 0.0))
    scenario["pitcher_k_pct"] = p_k_pct
    scenario["pitcher_bb_pct"] = p_bb_pct

    # Batter vs pitch-type splits (per-row lookup by pitch_type)
    batter_pt = batter_vs_pitch_type_splits.get(batter_name, {})
    league_vs_pitch = league_avg_batter_vs_pitch
    scenario["batter_whiff_pct_vs_pitch"] = scenario["pitch_type"].apply(
        lambda pt: batter_pt.get(pt, {}).get(
            "batter_whiff_pct_vs_pitch",
            league_vs_pitch.get("batter_whiff_pct_vs_pitch", 0.0),
        )
    )
    scenario["batter_k_pct_vs_pitch"] = scenario["pitch_type"].apply(
        lambda pt: batter_pt.get(pt, {}).get(
            "batter_k_pct_vs_pitch",
            league_vs_pitch.get("batter_k_pct_vs_pitch", 0.0),
        )
    )
    scenario["batter_hard_hit_pct_vs_pitch"] = scenario["pitch_type"].apply(
        lambda pt: batter_pt.get(pt, {}).get(
            "batter_hard_hit_pct_vs_pitch",
            league_vs_pitch.get("batter_hard_hit_pct_vs_pitch", 0.0),
        )
    )

    # Park factor
    home_team = state.get("home_team", "")
    scenario["park_run_factor"] = park_factors.get(home_team, 1.0)

    # Encode & predict
    encoded, _ = encode_categoricals(scenario, label_encoders=label_encoders, fit=False)
    X = build_feature_matrix(encoded)
    probs = model.predict_proba(X)
    class_names = le_outcome.classes_ if le_outcome is not None else model.classes_
    prob_df = pd.DataFrame(probs, columns=class_names)

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

    # Penalize poor pitch-zone combos (breaking balls up, anything middle-middle)
    apply_location_penalties(scenario)

    return scenario.nlargest(top_n, "re24_score")


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_recommendations(recs, title):
    """Horizontal bar chart of pitch recommendations."""
    _, ax = plt.subplots(figsize=(10, 6))
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
        f"Pitcher: {state.get('pitcher_name', 'unknown')} | "
        f"Batter: {state['stand']}HB | "
        f"Prev: {state['prev_pitch_type']} @ {state['prev_zone_label']} | "
        f"TTO: {state['times_through_order']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    (model, encoders, pitcher_pitch_avgs, batter_info, le_outcome,
     zone_location_avgs, pitcher_info, batter_stats, league_avg_batter_stats,
     batter_vs_pitch_type_splits, league_avg_batter_vs_pitch,
     pitcher_stats, league_avg_pitcher_stats, park_factors) = load_model()

    # ── Scenario 1: Eovaldi vs RHB, 0-0 count ─────────────────────────────
    game = new_game_state(batter_stand="R", pitcher_name="nathan eovaldi")
    print("=== Initial state (Eovaldi vs RHB) ===")
    print_state(game)

    recs = recommend_next_pitch(
        game, model, encoders, pitcher_pitch_avgs,
        le_outcome=le_outcome,
        zone_location_avgs=zone_location_avgs,
        pitcher_info=pitcher_info,
        batter_stats=batter_stats,
        league_avg_batter_stats=league_avg_batter_stats,
        batter_vs_pitch_type_splits=batter_vs_pitch_type_splits,
        league_avg_batter_vs_pitch=league_avg_batter_vs_pitch,
        pitcher_stats=pitcher_stats,
        league_avg_pitcher_stats=league_avg_pitcher_stats,
        park_factors=park_factors,
    )
    print("\nTop 10 Recommendations:")
    print(recs[["combo", "re24_score", "out", "hit"]].to_string(index=False))
    plot_recommendations(recs, "Eovaldi vs RHB — 0-0 count")

    # ── Simulate: SL at middle_right → strike ─────────────────────────────
    game = update_state(game, "strike", "SL", "middle_right")
    print("\n=== After SL @ middle_right → strike ===")
    print_state(game)

    recs = recommend_next_pitch(
        game, model, encoders, pitcher_pitch_avgs,
        le_outcome=le_outcome,
        zone_location_avgs=zone_location_avgs,
        pitcher_info=pitcher_info,
        batter_stats=batter_stats,
        league_avg_batter_stats=league_avg_batter_stats,
        batter_vs_pitch_type_splits=batter_vs_pitch_type_splits,
        league_avg_batter_vs_pitch=league_avg_batter_vs_pitch,
        pitcher_stats=pitcher_stats,
        league_avg_pitcher_stats=league_avg_pitcher_stats,
        park_factors=park_factors,
    )
    print("\nTop 10 Recommendations:")
    print(recs[["combo", "re24_score", "out", "hit"]].to_string(index=False))
    plot_recommendations(
        recs,
        f"After SL → strike | {game['balls']}-{game['strikes']} count",
    )

    # ── Scenario 2: Eovaldi vs LHB ────────────────────────────────────────
    print("\n" + "=" * 60)
    game_lhb = new_game_state(batter_stand="L", pitcher_name="nathan eovaldi")
    print("=== Fresh AB vs LHB ===")
    print_state(game_lhb)

    recs_lhb = recommend_next_pitch(
        game_lhb, model, encoders, pitcher_pitch_avgs,
        le_outcome=le_outcome,
        zone_location_avgs=zone_location_avgs,
        pitcher_info=pitcher_info,
        batter_stats=batter_stats,
        league_avg_batter_stats=league_avg_batter_stats,
        batter_vs_pitch_type_splits=batter_vs_pitch_type_splits,
        league_avg_batter_vs_pitch=league_avg_batter_vs_pitch,
        pitcher_stats=pitcher_stats,
        league_avg_pitcher_stats=league_avg_pitcher_stats,
        park_factors=park_factors,
    )
    print("\nTop 10 Recommendations:")
    print(recs_lhb[["combo", "re24_score", "out", "hit"]].to_string(index=False))
    plot_recommendations(recs_lhb, "Eovaldi vs LHB — 0-0 count")
