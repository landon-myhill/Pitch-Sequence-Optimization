"""
Shared utilities for Pitch Sequence Optimization pipeline.
"""

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Zone Mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Statcast zone IDs → descriptive labels
#
#   ┌────────────────────────┐
#   │  11 (up_left)  │  12 (up_right)  │   ← outside strike zone (high)
#   ├────┬────┬────┤                   │
#   │  1 │  2 │  3 │  top row          │
#   │  4 │  5 │  6 │  middle row       │
#   │  7 │  8 │  9 │  bottom row       │
#   ├────┴────┴────┤                   │
#   │  13 (dn_left) │  14 (dn_right)  │   ← outside strike zone (low)
#   └────────────────────────┘
#
ZONE_LABELS = {
    1: "top_left",      2: "top_middle",      3: "top_right",
    4: "middle_left",   5: "middle_middle",   6: "middle_right",
    7: "bottom_left",   8: "bottom_middle",   9: "bottom_right",
    11: "up_left",      12: "up_right",
    13: "down_left",    14: "down_right",
}

# Zone label → vertical tier (for zone-tier hard-hit features)
ZONE_TIERS = {
    "top_left": "upper",    "top_middle": "upper",    "top_right": "upper",
    "up_left":  "upper",    "up_right":   "upper",
    "middle_left": "middle", "middle_middle": "middle", "middle_right": "middle",
    "bottom_left": "lower", "bottom_middle": "lower",  "bottom_right": "lower",
    "down_left":   "lower", "down_right":    "lower",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Location Penalties
# ═══════════════════════════════════════════════════════════════════════════════

# Breaking balls / offspeed up in the zone are easier to hit — penalize them.
_OFFSPEED_TYPES = {"CH", "CU", "KC", "CS", "SL", "ST", "SV"}
_UPPER_ZONES = {z for z, tier in ZONE_TIERS.items() if tier == "upper"}

# Penalty applied as a multiplier on RE24 score (lower = worse for pitcher).
OFFSPEED_UP_PENALTY = 0.15      # 85% reduction for breaking balls up
MIDDLE_MIDDLE_PENALTY = 0.10    # 90% reduction for anything middle-middle


def apply_location_penalties(df):
    """
    Adjust RE24 scores for poor pitch-zone combinations:
      - Breaking balls / offspeed in upper zones (hanging pitches)
      - Any pitch middle-middle (heart of the plate)

    Operates in-place on a DataFrame that has 'pitch_type', 'zone_label',
    and 're24_score' columns.
    """
    is_offspeed_up = (
        df["pitch_type"].isin(_OFFSPEED_TYPES)
        & df["zone_label"].isin(_UPPER_ZONES)
    )
    is_middle_middle = df["zone_label"] == "middle_middle"

    df.loc[is_offspeed_up, "re24_score"] *= OFFSPEED_UP_PENALTY
    df.loc[is_middle_middle, "re24_score"] *= MIDDLE_MIDDLE_PENALTY

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome Classification
# ═══════════════════════════════════════════════════════════════════════════════

HIT_EVENTS = {
    "single", "double", "triple", "home_run",
    "field_error", "catcher_interf",
}

OUT_EVENTS = {
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "double_play",
    "grounded_into_double_play",
    "sac_fly", "sac_bunt",
    "fielders_choice", "fielders_choice_out",
    "triple_play", "sac_fly_double_play",
}

WALK_HBP_EVENTS = {"walk", "hit_by_pitch"}


WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked",
    "missed_bunt", "swinging_pitchout",
}

FOUL_DESCRIPTIONS = {
    "foul", "foul_tip", "foul_bunt", "foul_pitchout",
}


def classify_outcome(row):
    """
    Classify a pitch into: hit, out, walk_hbp, ball,
    called_strike, whiff, foul, or NonTerminalPitch.

    Splitting strikes lets the model learn that corner pitches
    generate whiffs/called strikes while middle pitches get fouls.
    """
    event = row.get("events", "")
    pitch_result = row.get("type", "")
    desc = str(row.get("description", "")).lower().strip()

    if event in HIT_EVENTS:
        return "hit"
    if event in OUT_EVENTS:
        return "out"
    if event in WALK_HBP_EVENTS:
        return "walk_hbp"
    if pitch_result == "B":
        return "ball"
    if pitch_result == "S":
        if desc in WHIFF_DESCRIPTIONS:
            return "whiff"
        if desc in FOUL_DESCRIPTIONS:
            return "foul"
        return "called_strike"
    return "NonTerminalPitch"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Definitions
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORICAL_FEATURES = [
    # pitch info
    "pitch_type", "zone_label",
    # count & game state
    "balls", "strikes", "outs_when_up", "inning",
    # baserunners
    "on_1b_occupied", "on_2b_occupied", "on_3b_occupied",
    # matchup
    "stand", "p_throws",
    # sequencing (previous pitch in this at-bat)
    "prev_pitch_type", "prev_zone_label",
]

NUMERIC_FEATURES = [
    # pitch physics
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "spin_axis",
    "release_extension",
    # plate location
    "plate_x",
    "plate_z",
    # pitch context
    "pitch_number",
    # game situation
    "score_differential",
    "pitcher_pitches_today",
    "pitcher_days_rest",
    # pitcher season stats
    "pitcher_k_pct",
    "pitcher_bb_pct",
    # batter season stats
    "batter_k_pct",
    "batter_bb_pct",
    "batter_whiff_pct",
    "batter_hard_hit_pct",
    # batter vs this specific pitch type
    "batter_whiff_pct_vs_pitch",
    "batter_k_pct_vs_pitch",
    "batter_hard_hit_pct_vs_pitch",
    # park
    "park_run_factor",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ═══════════════════════════════════════════════════════════════════════════════
# RE24 Run Expectancy
# ═══════════════════════════════════════════════════════════════════════════════

# 2024 MLB averages — expected runs scored from this state to end of inning.
#
# Rows: base state (---/1--/-2-/12-/--3/1-3/-23/123)
# Cols: outs (0, 1, 2)
#
#                        0 out   1 out   2 out
_RE24_TABLE = {
    (False, False, False): (0.481, 0.254, 0.098),  # ---  bases empty
    (True,  False, False): (0.859, 0.509, 0.224),  # 1--  runner on 1st
    (False, True,  False): (1.100, 0.664, 0.319),  # -2-  runner on 2nd
    (True,  True,  False): (1.437, 0.884, 0.429),  # 12-  1st and 2nd
    (False, False, True):  (1.350, 0.950, 0.353),  # --3  runner on 3rd
    (True,  False, True):  (1.784, 1.130, 0.478),  # 1-3  1st and 3rd
    (False, True,  True):  (1.964, 1.376, 0.580),  # -23  2nd and 3rd
    (True,  True,  True):  (2.282, 1.541, 0.689),  # 123  bases loaded
}

# Expected run-value change when the count changes.
# Key: (balls, strikes) → batter's run-value advantage relative to 0-0.
# Positive = favors batter, negative = favors pitcher.
#
#             0 str   1 str   2 str
# 0 balls:   0.000  -0.038  -0.096
# 1 ball:    0.032  -0.011  -0.066
# 2 balls:   0.077   0.030  -0.029
# 3 balls:   0.152   0.098   0.044
#
_COUNT_VALUES = {
    (0, 0):  0.000,  (0, 1): -0.038,  (0, 2): -0.096,
    (1, 0):  0.032,  (1, 1): -0.011,  (1, 2): -0.066,
    (2, 0):  0.077,  (2, 1):  0.030,  (2, 2): -0.029,
    (3, 0):  0.152,  (3, 1):  0.098,  (3, 2):  0.044,
}


def _get_re24(on_1b, on_2b, on_3b, outs):
    """Look up expected runs. Returns 0 for 3-out states (inning over)."""
    if outs >= 3:
        return 0.0
    bases = (bool(on_1b), bool(on_2b), bool(on_3b))
    return _RE24_TABLE.get(bases, (0, 0, 0))[int(outs)]


def _re_after_walk(on_1b, on_2b, on_3b, outs):
    """RE after a walk — batter to 1st, forced runners advance."""
    if on_1b and on_2b and on_3b:
        return _get_re24(True, True, True, outs) + 1.0   # run scores
    if on_1b and on_2b:
        return _get_re24(True, True, True, outs)          # runner to 3rd
    if on_1b:
        return _get_re24(True, True, False, outs)         # runner to 2nd
    return _get_re24(True, on_2b, on_3b, outs)            # batter to 1st only


def _re_after_hit(on_1b, on_2b, on_3b, outs):
    """RE after a single (simplified) — runners advance ~1-2 bases."""
    if on_3b:
        return _get_re24(True, on_1b, on_2b, outs) + 1.0   # run scores from 3rd
    if on_2b:
        return _get_re24(True, False, True, outs) + 0.4     # ~40% score from 2nd
    if on_1b:
        return _get_re24(True, True, False, outs)            # runner to 2nd
    return _get_re24(True, False, False, outs)               # batter on 1st


def compute_re24_score(prob_dict, balls, strikes, outs, on_1b, on_2b, on_3b):
    """
    Score a pitch by expected run-value change. Higher = better for pitcher.

    Uses the RE24 matrix for terminal outcomes (out, hit, walk) and
    count-value tables for mid-AB outcomes (strike, ball).
    """
    re_before = _get_re24(on_1b, on_2b, on_3b, outs)

    # ── Terminal outcomes ──────────────────────────────────────────────────
    re_strikeout = _get_re24(on_1b, on_2b, on_3b, outs + 1)
    re_out       = _get_re24(on_1b, on_2b, on_3b, outs + 1)
    re_walk      = _re_after_walk(on_1b, on_2b, on_3b, outs)
    re_hit       = _re_after_hit(on_1b, on_2b, on_3b, outs)

    # ── Count-change outcomes ─────────────────────────────────────────────
    # Strike (non-terminal vs strikeout)
    if strikes >= 2:
        re_strike = re_strikeout
    else:
        count_delta = (
            _COUNT_VALUES.get((balls, strikes + 1), 0)
            - _COUNT_VALUES.get((balls, strikes), 0)
        )
        re_strike = re_before + count_delta

    # Ball (non-terminal vs walk)
    if balls >= 3:
        re_ball = re_walk
    else:
        count_delta = (
            _COUNT_VALUES.get((balls + 1, strikes), 0)
            - _COUNT_VALUES.get((balls, strikes), 0)
        )
        re_ball = re_before + count_delta

    # ── Foul ball RE ──────────────────────────────────────────────────────
    # Fouls add a strike only when strikes < 2; at 2 strikes they're dead pitches.
    if strikes >= 2:
        re_foul = re_before  # count doesn't change
    else:
        re_foul = re_strike  # same as adding a strike

    # ── Weighted expected RE after this pitch ─────────────────────────────
    p = prob_dict
    # Sum all strike-type probabilities for backwards compatibility
    p_called = p.get("called_strike", 0)
    p_whiff  = p.get("whiff", 0)
    p_foul   = p.get("foul", 0)
    # Also accept legacy "strike" key
    p_strike_legacy = p.get("strike", 0)

    expected_re = (
        (p_called + p_whiff + p_strike_legacy) * re_strike
        + p_foul                                 * re_foul
        + p.get("ball", 0)                       * re_ball
        + p.get("out", 0)                        * re_out
        + p.get("hit", 0)                        * re_hit
        + p.get("walk_hbp", 0)                   * re_walk
    )

    # Negative delta = good for pitcher → flip sign so higher = better
    return -(expected_re - re_before)


# ═══════════════════════════════════════════════════════════════════════════════
# Batter Stats
# ═══════════════════════════════════════════════════════════════════════════════

def compute_batter_stats(data):
    """
    Compute per-batter aggregate stats from pitch data.

    Returns a DataFrame indexed by batter_name with columns:
        batter_k_pct, batter_bb_pct, batter_whiff_pct, batter_hard_hit_pct
    """
    df = data.copy()

    # ── Plate appearance flag: only rows where events is non-null and non-empty ──
    df["_is_pa"] = df["events"].notna() & (df["events"].astype(str).str.strip() != "")

    # Terminal outcomes by batter
    df["_is_k"]    = df["events"].isin({"strikeout", "strikeout_double_play"})
    df["_is_bb"]   = df["events"].isin(WALK_HBP_EVENTS)
    df["_is_hit"]  = df["events"].isin(HIT_EVENTS)

    # Whiff / foul / called strike per pitch
    desc = df["description"].astype(str).str.lower().str.strip()
    df["_is_whiff"]        = desc.isin(WHIFF_DESCRIPTIONS)
    df["_is_foul"]         = desc.isin(FOUL_DESCRIPTIONS)
    df["_is_called_strike"] = (df["type"] == "S") & ~df["_is_whiff"] & ~df["_is_foul"]

    grouped = df.groupby("batter_name")

    pa         = grouped["_is_pa"].sum()
    k          = grouped["_is_k"].sum()
    bb         = grouped["_is_bb"].sum()
    hits       = grouped["_is_hit"].sum()
    whiffs     = grouped["_is_whiff"].sum()
    fouls      = grouped["_is_foul"].sum()
    c_strikes  = grouped["_is_called_strike"].sum()

    batter_stats = pd.DataFrame(index=pa.index)
    batter_stats["batter_k_pct"]        = (k / pa.clip(lower=1)).fillna(0.0)
    batter_stats["batter_bb_pct"]       = (bb / pa.clip(lower=1)).fillna(0.0)
    batter_stats["batter_hard_hit_pct"] = (hits / pa.clip(lower=1)).fillna(0.0)

    swing_opportunities = whiffs + fouls + c_strikes
    batter_stats["batter_whiff_pct"] = (
        whiffs / swing_opportunities.clip(lower=1)
    ).fillna(0.0)

    return batter_stats


def compute_batter_pitch_type_splits(data):
    """
    Compute per-batter, per-pitch-type stats.
    Returns DataFrame with: batter_name, pitch_type,
    batter_whiff_pct_vs_pitch, batter_k_pct_vs_pitch, batter_hard_hit_pct_vs_pitch
    """
    df = data.copy()
    desc = df["description"].astype(str).str.lower().str.strip()
    df["_is_whiff"] = desc.isin(WHIFF_DESCRIPTIONS)
    df["_is_foul"] = desc.isin(FOUL_DESCRIPTIONS)
    df["_is_called_strike"] = (df["type"] == "S") & ~df["_is_whiff"] & ~df["_is_foul"]
    df["_is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"})
    df["_is_pa"] = df["events"].notna() & (df["events"].astype(str).str.strip() != "")
    df["_is_hit"] = df["events"].isin(HIT_EVENTS)

    g = df.groupby(["batter_name", "pitch_type"])
    agg = g.agg(
        pa=("_is_pa", "sum"),
        k=("_is_k", "sum"),
        hits=("_is_hit", "sum"),
        whiffs=("_is_whiff", "sum"),
        fouls=("_is_foul", "sum"),
        c_strikes=("_is_called_strike", "sum"),
    ).reset_index()

    swing_opp = agg["whiffs"] + agg["fouls"] + agg["c_strikes"]
    agg["batter_whiff_pct_vs_pitch"] = (agg["whiffs"] / swing_opp.clip(lower=1)).fillna(0.0)
    agg["batter_k_pct_vs_pitch"] = (agg["k"] / agg["pa"].clip(lower=1)).fillna(0.0)
    agg["batter_hard_hit_pct_vs_pitch"] = (agg["hits"] / agg["pa"].clip(lower=1)).fillna(0.0)

    return agg[["batter_name", "pitch_type",
                "batter_whiff_pct_vs_pitch", "batter_k_pct_vs_pitch",
                "batter_hard_hit_pct_vs_pitch"]]


def compute_batter_pitch_zone_tier_splits(data):
    """
    Compute per-batter, per-pitch-type hard-hit% split by zone tier (upper/middle/lower).

    Returns DataFrame with: batter_name, pitch_type,
    batter_hard_hit_pct_vs_pitch_upper, batter_hard_hit_pct_vs_pitch_middle,
    batter_hard_hit_pct_vs_pitch_lower
    """
    df = data.copy()
    df["_zone_tier"] = df["zone_label"].map(ZONE_TIERS).fillna("middle")
    df["_is_hit"] = df["events"].isin(HIT_EVENTS)
    df["_is_pa"] = df["events"].notna() & (df["events"].astype(str).str.strip() != "")

    g = df.groupby(["batter_name", "pitch_type", "_zone_tier"])
    agg = g.agg(pa=("_is_pa", "sum"), hits=("_is_hit", "sum")).reset_index()
    agg["hard_hit_pct"] = (agg["hits"] / agg["pa"].clip(lower=1)).fillna(0.0)

    pivoted = agg.pivot_table(
        index=["batter_name", "pitch_type"],
        columns="_zone_tier",
        values="hard_hit_pct",
        fill_value=0.0,
    ).reset_index()

    for tier in ("upper", "middle", "lower"):
        if tier not in pivoted.columns:
            pivoted[tier] = 0.0

    pivoted = pivoted.rename(columns={
        "upper":  "batter_hard_hit_pct_vs_pitch_upper",
        "middle": "batter_hard_hit_pct_vs_pitch_middle",
        "lower":  "batter_hard_hit_pct_vs_pitch_lower",
    })

    cols = ["batter_name", "pitch_type",
            "batter_hard_hit_pct_vs_pitch_upper",
            "batter_hard_hit_pct_vs_pitch_middle",
            "batter_hard_hit_pct_vs_pitch_lower"]
    return pivoted[cols]


def compute_pitcher_stats(data):
    """
    Compute per-pitcher season K% and BB%.
    Returns DataFrame indexed by pitcher_name with pitcher_k_pct, pitcher_bb_pct.
    """
    df = data.copy()
    df["_is_pa"] = df["events"].notna() & (df["events"].astype(str).str.strip() != "")
    df["_is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"})
    df["_is_bb"] = df["events"].isin(WALK_HBP_EVENTS)

    g = df.groupby("pitcher_name")
    agg = g.agg(pa=("_is_pa", "sum"), k=("_is_k", "sum"), bb=("_is_bb", "sum"))
    agg["pitcher_k_pct"] = (agg["k"] / agg["pa"].clip(lower=1)).fillna(0.0)
    agg["pitcher_bb_pct"] = (agg["bb"] / agg["pa"].clip(lower=1)).fillna(0.0)
    return agg[["pitcher_k_pct", "pitcher_bb_pct"]]


def compute_park_factors(data):
    """
    Compute park run factors normalized to league average = 1.0.
    Returns {home_team: factor}
    """
    if "home_team" not in data.columns:
        return {}

    tmp = data[["game_pk", "home_team"]].copy()
    tmp["bat_score"] = pd.to_numeric(data.get("bat_score", 0), errors="coerce").fillna(0)
    tmp["fld_score"] = pd.to_numeric(data.get("fld_score", 0), errors="coerce").fillna(0)

    game_totals = (
        tmp.groupby(["game_pk", "home_team"])
        .agg(max_bat=("bat_score", "max"), max_fld=("fld_score", "max"))
        .reset_index()
    )
    game_totals["total_runs"] = game_totals["max_bat"] + game_totals["max_fld"]
    park_avg = game_totals.groupby("home_team")["total_runs"].mean()
    league_avg = park_avg.mean()
    if league_avg == 0:
        return {}
    return (park_avg / league_avg).to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(csv_path, pitcher_name=None, min_rows=500):
    """
    Load combined pitch CSV, apply zone labels, classify outcomes,
    compute batter aggregate stats, and optionally filter to a specific pitcher.

    Args:
        csv_path:     path to the combined CSV.
        pitcher_name: if None, return ALL pitchers (for pooled training).
                      If provided, filter to that pitcher only.
        min_rows:     minimum rows required when filtering to one pitcher.
    """
    data = pd.read_csv(csv_path, low_memory=False)

    # Remove truncated plate appearances
    data = data[data["events"] != "truncated_pa"].copy()

    # Map zone labels
    data["zone_label"] = data["zone"].map(ZONE_LABELS)
    data = data[data["zone_label"].notna()].copy()

    # Classify outcomes
    data["pitch_outcome"] = data.apply(classify_outcome, axis=1)

    # Fill missing sequencing features
    for col in ("prev_pitch_type", "prev_zone_label"):
        if col in data.columns:
            data[col] = data[col].fillna("none")

    # Fill missing pitch-physics numeric features with column median
    # (batter stat columns are added below via merge, so they won't be in
    #  the raw CSV and the loop below will skip them correctly)
    physics_cols = [
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "spin_axis", "release_extension",
        "plate_x", "plate_z",
        "pitch_number", "score_differential",
        "pitcher_pitches_today", "pitcher_days_rest",
    ]
    for col in physics_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())

    # Fill missing handedness (default right)
    for col in ("stand", "p_throws"):
        if col in data.columns:
            data[col] = data[col].fillna("R")

    # ── Batter aggregate stats ─────────────────────────────────────────────
    batter_stats = compute_batter_stats(data)

    # League-average fallback for batters not in training data
    league_avg = batter_stats.mean()

    # Merge stats into pitch rows
    data = data.merge(
        batter_stats.reset_index(),
        on="batter_name",
        how="left",
    )
    for stat_col in ["batter_k_pct", "batter_bb_pct", "batter_whiff_pct", "batter_hard_hit_pct"]:
        if stat_col in data.columns:
            data[stat_col] = data[stat_col].fillna(league_avg[stat_col])
        else:
            data[stat_col] = league_avg[stat_col]

    # Optionally filter to one pitcher
    if pitcher_name is not None:
        data = data[
            data["pitcher_name"].str.lower().str.strip() == pitcher_name.lower().strip()
        ].copy()

        if len(data) < min_rows:
            raise ValueError(
                f"Not enough data for {pitcher_name} "
                f"(only {len(data)} rows, need {min_rows})."
            )

    return data
