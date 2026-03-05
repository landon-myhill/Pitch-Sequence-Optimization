"""
Shared utilities for Pitch Sequence Optimization pipeline.
"""

import pandas as pd
import numpy as np

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


def classify_outcome(row):
    """Classify a pitch into: hit, out, walk_hbp, ball, strike, or NonTerminalPitch."""
    event = row.get("events", "")
    pitch_result = row.get("type", "")

    if event in HIT_EVENTS:
        return "hit"
    if event in OUT_EVENTS:
        return "out"
    if event in WALK_HBP_EVENTS:
        return "walk_hbp"
    if pitch_result == "B":
        return "ball"
    if pitch_result == "S":
        return "strike"
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
    "release_speed",      # velocity (mph)
    "release_spin_rate",  # spin (rpm)
    "pfx_x",              # horizontal movement (ft)
    "pfx_z",              # vertical movement (ft)
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

    # ── Weighted expected RE after this pitch ─────────────────────────────
    p = prob_dict
    expected_re = (
        p.get("strike", 0)   * re_strike
        + p.get("ball", 0)     * re_ball
        + p.get("out", 0)      * re_out
        + p.get("hit", 0)      * re_hit
        + p.get("walk_hbp", 0) * re_walk
    )

    # Negative delta = good for pitcher → flip sign so higher = better
    return -(expected_re - re_before)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(csv_path, pitcher_name=None, min_rows=500):
    """
    Load combined pitch CSV, apply zone labels, classify outcomes,
    and optionally filter to a specific pitcher.

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

    # Fill missing numeric features with column median
    for col in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())

    # Fill missing handedness (default right)
    for col in ("stand", "p_throws"):
        if col in data.columns:
            data[col] = data[col].fillna("R")

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
