"""
Combine and clean Statcast pitch data for the Texas Rangers.

Pipeline step 2: reads raw CSVs from games_by_team/, filters to Rangers,
cleans columns, engineers lag features, and saves a single CSV.
"""

import glob
import pandas as pd
from utils import ZONE_LABELS, NUMERIC_FEATURES


def main():
    # ── Load all game CSVs ────────────────────────────────────────────────
    csv_files = glob.glob("games_by_team/**/*.csv", recursive=True)
    print(f"Found {len(csv_files)} CSV files.")

    pitch_data = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in csv_files],
        ignore_index=True,
    )

    # Deduplicate — games appear under both home and away team folders
    if "game_pk" in pitch_data.columns and "pitch_number" in pitch_data.columns:
        pitch_data = pitch_data.drop_duplicates(
            subset=["game_pk", "at_bat_number", "pitch_number"], keep="first"
        )
    print(f"Total rows (deduplicated): {len(pitch_data)}")

    # ── Drop unwanted columns ─────────────────────────────────────────────
    # Kept: pfx_x, pfx_z, release_speed, release_spin_rate, stand, p_throws
    drop = [
        # deprecated / internal
        "spin_dir", "spin_rate_deprecated", "break_angle_deprecated",
        "break_length_deprecated", "tfs_deprecated", "tfs_zulu_deprecated",
        "umpire", "sv_id",
        # raw trajectory (redundant with pfx)
        "vx0", "vy0", "vz0", "ax", "ay", "az",
        # fielder IDs
        "fielder_2", "fielder_3", "fielder_4", "fielder_5",
        "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        # bat tracking
        "bat_speed", "swing_length",
        # hit coordinates (not used in model)
        "hc_x", "hc_y",
        # derived batting stats
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "babip_value", "iso_value",
        "launch_speed_angle", "estimated_slg_using_speedangle",
        "delta_pitcher_run_exp", "hyper_speed",
        "home_win_exp", "bat_win_exp",
    ]
    drop = [c for c in drop if c in pitch_data.columns]
    pitch_data.drop(columns=drop, inplace=True)

    # ── Baserunners → boolean flags ───────────────────────────────────────
    for base in ("on_1b", "on_2b", "on_3b"):
        if base in pitch_data.columns:
            pitch_data[f"{base}_occupied"] = pitch_data[base].notna()
            pitch_data.drop(columns=[base], inplace=True)

    # ── NA handling: categoricals ─────────────────────────────────────────
    _fill_na_str(pitch_data, "events",      "NonTerminalPitch")
    _fill_na_str(pitch_data, "description", "unknown")
    _fill_na_str(pitch_data, "hit_location", "NonTerminalLocation")
    _fill_na_str(pitch_data, "bb_type",     "NoBB")
    _fill_na_str(pitch_data, "pitcher_days_since_prev_game", "NoPriorGame")
    _fill_na_str(pitch_data, "batter_days_since_prev_game",  "NoPriorGame")

    # Batter names: normalize to lowercase, fill missing
    if "batter_name" in pitch_data.columns:
        pitch_data["batter_name"] = (
            pitch_data["batter_name"].fillna("unknown").str.lower().str.strip()
        )

    # Derive batter's team from inning_topbot (Top = away batting, Bot = home)
    if "inning_topbot" in pitch_data.columns:
        pitch_data["batter_team"] = pitch_data.apply(
            lambda r: r["away_team"] if r.get("inning_topbot") == "Top" else r["home_team"],
            axis=1,
        )

    # Ball-in-play metrics: not-applicable when no BIP
    for col in ("hit_distance_sc", "launch_speed", "launch_angle"):
        _fill_na_str(pitch_data, col, "NoBB")

    # ── NA handling: pitch physics (keep numeric, fill with median) ───────
    for col in NUMERIC_FEATURES:
        if col in pitch_data.columns:
            pitch_data[col] = pd.to_numeric(pitch_data[col], errors="coerce")
            median = pitch_data[col].median()
            pitch_data[col] = pitch_data[col].fillna(median)

    # ── Pitch sequencing: lag features within each at-bat ─────────────────
    _add_lag_features(pitch_data)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"Final shape: {pitch_data.shape}")
    pitch_data.to_csv("combined_pitch_data.csv", index=False)
    print("Saved combined_pitch_data.csv")


def _fill_na_str(df, col, fill_value):
    """Fill NAs in a column, converting to string."""
    if col in df.columns:
        df[col] = df[col].fillna(fill_value).astype(str)


def _add_lag_features(df):
    """
    Add prev_pitch_type and prev_zone_label by shifting within each at-bat.
    First pitch of each AB gets "none".
    """
    # Sort chronologically
    sort_cols = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)

    # Map zones for lag calc
    df["_zone_temp"] = df["zone"].map(ZONE_LABELS)

    if "game_pk" in df.columns and "at_bat_number" in df.columns:
        grp = df.groupby(["game_pk", "at_bat_number"])
        df["prev_pitch_type"]  = grp["pitch_type"].shift(1).fillna("none")
        df["prev_zone_label"]  = grp["_zone_temp"].shift(1).fillna("none")
    else:
        df["prev_pitch_type"]  = "none"
        df["prev_zone_label"]  = "none"

    df.drop(columns=["_zone_temp"], inplace=True)


if __name__ == "__main__":
    main()
