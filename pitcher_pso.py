"""
Train an XGBoost model to predict pitch outcomes.

Key design decisions:
  - Trains on ALL pitchers (pooled) for much larger sample size (~50k+ rows).
  - Uses batter handedness, pitch sequencing, and pitch physics as features.
  - Scores recommendations with RE24 run expectancy (context-aware).
  - Supports any pitcher via pitcher_pitch_avgs and pitcher_info dicts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from itertools import product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils import (
    load_and_prepare_data,
    compute_batter_stats,
    compute_batter_pitch_type_splits,
    compute_batter_pitch_zone_tier_splits,
    compute_pitcher_stats,
    compute_park_factors,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    compute_re24_score,
)

# Physics features stored per pitcher/pitch-type (excludes batter stats and plate location)
PHYSICS_FEATURES = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z", "spin_axis", "release_extension"]


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding helpers
# ═══════════════════════════════════════════════════════════════════════════════

def encode_categoricals(df, label_encoders=None, fit=False):
    """
    Encode categorical columns with LabelEncoder.

    If fit=True, creates new encoders and returns them.
    If fit=False, uses existing encoders (handles unseen labels by defaulting to 0).
    """
    if label_encoders is None:
        label_encoders = {}

    encoded = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in encoded.columns:
            encoded[col] = "unknown"

        encoded[col] = encoded[col].astype(str)

        if fit:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            known = set(le.classes_)
            encoded[col] = encoded[col].apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )

    return encoded, label_encoders


def build_feature_matrix(df):
    """Extract the feature matrix (categorical encoded + numeric) from a DataFrame."""
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df[ALL_FEATURES].values


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1) Load data ──────────────────────────────────────────────────────
    data = load_and_prepare_data("combined_pitch_data.csv", pitcher_name=None)
    print(f"Total rows (all pitchers): {len(data)}")

    # ── 2) Compute per-pitcher pitch physics averages (>= 200 pitches) ───
    pitcher_counts = data.groupby("pitcher_name").size()
    qualified_pitchers = pitcher_counts[pitcher_counts >= 200].index.tolist()

    pitcher_pitch_avgs = {}
    for pitcher in qualified_pitchers:
        pitcher_data = data[data["pitcher_name"] == pitcher]
        avgs = (
            pitcher_data.groupby("pitch_type")[PHYSICS_FEATURES]
            .mean()
            .to_dict(orient="index")
        )
        pitcher_pitch_avgs[pitcher] = avgs

    print(f"Computed pitch avgs for {len(pitcher_pitch_avgs)} pitchers")

    # ── 3) Compute pitcher_info (p_throws + pitch_types) ─────────────────
    pitcher_info = {}
    for pitcher in qualified_pitchers:
        pitcher_data = data[data["pitcher_name"] == pitcher]
        p_throws = pitcher_data["p_throws"].mode().iloc[0] if "p_throws" in pitcher_data.columns else "R"
        pitch_types = pitcher_data["pitch_type"].dropna().unique().tolist()
        pitcher_info[pitcher] = {
            "p_throws": p_throws,
            "pitch_types": pitch_types,
        }


    # ── 4) Eovaldi-specific averages (backwards-compat fallback key) ──────
    eovaldi_name = "nathan eovaldi"
    eovaldi_pitch_avgs = pitcher_pitch_avgs.get(eovaldi_name, {})
    print(f"Eovaldi pitch types: {list(eovaldi_pitch_avgs.keys())}")

    # ── 5) Compute zone_location_avgs: avg plate_x/plate_z per (pitch_type, zone_label)
    #       using ALL pitchers in training data
    zone_location_avgs = {}
    if "plate_x" in data.columns and "plate_z" in data.columns:
        zone_loc_df = (
            data.groupby(["pitch_type", "zone_label"])[["plate_x", "plate_z"]]
            .mean()
        )
        for (pt, zl), row in zone_loc_df.iterrows():
            if pt not in zone_location_avgs:
                zone_location_avgs[pt] = {}
            zone_location_avgs[pt][zl] = {
                "plate_x": float(row["plate_x"]),
                "plate_z": float(row["plate_z"]),
            }

    # ── 6) Compute batter stats dict and league averages ──────────────────
    batter_stats_df = compute_batter_stats(data)
    batter_stats_dict = batter_stats_df.to_dict(orient="index")
    league_avg_batter_stats = batter_stats_df.mean().to_dict()

    # ── 6b) Batter vs pitch-type splits ───────────────────────────────────
    batter_pt_df = compute_batter_pitch_type_splits(data)
    data = data.merge(batter_pt_df, on=["batter_name", "pitch_type"], how="left")
    for col in ["batter_whiff_pct_vs_pitch", "batter_k_pct_vs_pitch", "batter_hard_hit_pct_vs_pitch"]:
        data[col] = data[col].fillna(data[col].median())

    # Build nested dict for artifact: {batter_name: {pitch_type: {stat: val}}}
    batter_vs_pitch_type_splits = {}
    for _, row in batter_pt_df.iterrows():
        bn, pt = row["batter_name"], row["pitch_type"]
        if bn not in batter_vs_pitch_type_splits:
            batter_vs_pitch_type_splits[bn] = {}
        batter_vs_pitch_type_splits[bn][pt] = {
            "batter_whiff_pct_vs_pitch": float(row["batter_whiff_pct_vs_pitch"]),
            "batter_k_pct_vs_pitch": float(row["batter_k_pct_vs_pitch"]),
            "batter_hard_hit_pct_vs_pitch": float(row["batter_hard_hit_pct_vs_pitch"]),
        }

    league_avg_batter_vs_pitch = {
        "batter_whiff_pct_vs_pitch": float(batter_pt_df["batter_whiff_pct_vs_pitch"].mean()),
        "batter_k_pct_vs_pitch": float(batter_pt_df["batter_k_pct_vs_pitch"].mean()),
        "batter_hard_hit_pct_vs_pitch": float(batter_pt_df["batter_hard_hit_pct_vs_pitch"].mean()),
    }

    # ── 6b2) Batter vs pitch-type × zone-tier hard-hit splits ─────────────
    batter_zone_tier_df = compute_batter_pitch_zone_tier_splits(data)
    data = data.merge(batter_zone_tier_df, on=["batter_name", "pitch_type"], how="left")
    tier_cols = ["batter_hard_hit_pct_vs_pitch_upper",
                 "batter_hard_hit_pct_vs_pitch_middle",
                 "batter_hard_hit_pct_vs_pitch_lower"]
    for col in tier_cols:
        data[col] = data[col].fillna(data[col].median())

    # Merge zone-tier stats into batter_vs_pitch_type_splits dict
    for _, row in batter_zone_tier_df.iterrows():
        bn, pt = row["batter_name"], row["pitch_type"]
        if bn not in batter_vs_pitch_type_splits:
            batter_vs_pitch_type_splits[bn] = {}
        if pt not in batter_vs_pitch_type_splits[bn]:
            batter_vs_pitch_type_splits[bn][pt] = {}
        batter_vs_pitch_type_splits[bn][pt].update({
            "batter_hard_hit_pct_vs_pitch_upper":  float(row["batter_hard_hit_pct_vs_pitch_upper"]),
            "batter_hard_hit_pct_vs_pitch_middle": float(row["batter_hard_hit_pct_vs_pitch_middle"]),
            "batter_hard_hit_pct_vs_pitch_lower":  float(row["batter_hard_hit_pct_vs_pitch_lower"]),
        })

    for col in tier_cols:
        league_avg_batter_vs_pitch[col] = float(batter_zone_tier_df[col].mean())

    # ── 6c) Pitcher season stats ───────────────────────────────────────────
    pitcher_stats_df = compute_pitcher_stats(data)
    data = data.merge(pitcher_stats_df.reset_index(), on="pitcher_name", how="left")
    for col in ["pitcher_k_pct", "pitcher_bb_pct"]:
        data[col] = data[col].fillna(data[col].median())

    pitcher_stats_dict = pitcher_stats_df.to_dict(orient="index")
    league_avg_pitcher_stats = pitcher_stats_df.mean().to_dict()

    # ── 6d) Park factors ──────────────────────────────────────────────────
    park_factors = compute_park_factors(data)
    if park_factors and "home_team" in data.columns:
        data["park_run_factor"] = data["home_team"].map(park_factors).fillna(1.0)
    else:
        data["park_run_factor"] = 1.0

    # ── 6e) Enrich pitcher_info with stats and avg days rest ──────────────
    for pitcher in qualified_pitchers:
        p_stats = pitcher_stats_dict.get(pitcher, {})
        pitcher_info[pitcher]["pitcher_k_pct"] = p_stats.get(
            "pitcher_k_pct", league_avg_pitcher_stats.get("pitcher_k_pct", 0.0)
        )
        pitcher_info[pitcher]["pitcher_bb_pct"] = p_stats.get(
            "pitcher_bb_pct", league_avg_pitcher_stats.get("pitcher_bb_pct", 0.0)
        )
        pitcher_game_dates = data[data["pitcher_name"] == pitcher]["pitcher_days_rest"].dropna()
        pitcher_info[pitcher]["pitcher_days_rest_avg"] = (
            float(pitcher_game_dates.mean()) if len(pitcher_game_dates) > 0 else 5.0
        )

    # ── 7) Encode & split ─────────────────────────────────────────────────
    data_enc, label_encoders = encode_categoricals(data, fit=True)
    X = build_feature_matrix(data_enc)

    le_outcome = LabelEncoder()
    y = le_outcome.fit_transform(data_enc["pitch_outcome"].astype(str).values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y,
    )

    # ── 8) Train ──────────────────────────────────────────────────────────
    rf = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=123,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    rf.fit(X_train, y_train, sample_weight=sample_weights)

    # ── 9) Evaluate ───────────────────────────────────────────────────────
    preds = rf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds, labels=rf.classes_))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le_outcome.classes_))

    # ── 10) Save ──────────────────────────────────────────────────────────
    # Build batter list with handedness and team for the UI
    batter_df = data[["batter_name", "stand"]].copy()
    if "batter_team" in data.columns:
        batter_df["team"] = data["batter_team"]
    else:
        batter_df["team"] = "UNK"
    # Take the most common team/stand per batter (handles trades, etc.)
    batter_info = (
        batter_df.groupby("batter_name")
        .agg({"stand": lambda x: x.mode().iloc[0], "team": lambda x: x.mode().iloc[0]})
        .to_dict(orient="index")
    )

    artifact = {
        "model":                        rf,
        "label_encoders":               label_encoders,
        "le_outcome":                   le_outcome,
        "pitcher_pitch_avgs":           pitcher_pitch_avgs,
        "pitcher_info":                 pitcher_info,
        "zone_location_avgs":           zone_location_avgs,
        "batter_info":                  batter_info,
        "batter_stats":                 batter_stats_dict,
        "league_avg_batter_stats":      league_avg_batter_stats,
        "batter_vs_pitch_type_splits":  batter_vs_pitch_type_splits,
        "league_avg_batter_vs_pitch":   league_avg_batter_vs_pitch,
        "pitcher_stats":                pitcher_stats_dict,
        "league_avg_pitcher_stats":     league_avg_pitcher_stats,
        "park_factors":                 park_factors,
        # backwards-compat fallback
        "eovaldi_pitch_avgs":           eovaldi_pitch_avgs,
    }
    joblib.dump(artifact, "pitch_sequence_model.pkl")
    print("Saved pitch_sequence_model.pkl")

    # ── 11) Scenario predictions for Eovaldi ──────────────────────────────
    eovaldi_p_throws = pitcher_info.get(eovaldi_name, {}).get("p_throws", "R")
    eovaldi_k_pct = pitcher_info.get(eovaldi_name, {}).get(
        "pitcher_k_pct", league_avg_pitcher_stats.get("pitcher_k_pct", 0.0)
    )
    eovaldi_bb_pct = pitcher_info.get(eovaldi_name, {}).get(
        "pitcher_bb_pct", league_avg_pitcher_stats.get("pitcher_bb_pct", 0.0)
    )
    scenario = build_pitcher_scenario(
        eovaldi_pitch_avgs,
        label_encoders,
        p_throws=eovaldi_p_throws,
        zone_location_avgs=zone_location_avgs,
        batter_stats=league_avg_batter_stats,
        pitcher_k_pct=eovaldi_k_pct,
        pitcher_bb_pct=eovaldi_bb_pct,
    )
    results = predict_and_score(rf, scenario, label_encoders, le_outcome=le_outcome)

    # ── 12) Visualize ─────────────────────────────────────────────────────
    for outcome in ("out", "called_strike", "whiff", "hit"):
        plot_outcome_by_zone(results, outcome, f"P({outcome}) by Zone & Pitch Type (Eovaldi)")

    top10 = results.nlargest(10, "re24_score")
    print("\nTop 10 Pitch+Zone Combos (RE24 — higher = better for pitcher):")
    print(top10[["combo", "re24_score", "called_strike", "whiff", "foul", "out", "hit"]].to_string(index=False))
    plot_top_combos(top10)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario building & prediction
# ═══════════════════════════════════════════════════════════════════════════════

def build_pitcher_scenario(pitch_avgs, label_encoders, p_throws="R",
                           zone_location_avgs=None, batter_stats=None,
                           pitcher_k_pct=0.0, pitcher_bb_pct=0.0):
    """
    Build a scenario DataFrame with all pitch_type × zone combos
    at a default game state (0-0 count, 0 outs, bases empty, vs RHB).

    pitch_avgs: {pitch_type: {feature: value}} for PHYSICS_FEATURES only.
    zone_location_avgs: nested dict {pitch_type: {zone_label: {plate_x, plate_z}}}
    batter_stats: dict of {stat_name: value} (league avg or specific batter)
    pitcher_k_pct: pitcher season K%
    pitcher_bb_pct: pitcher season BB%
    """
    if zone_location_avgs is None:
        zone_location_avgs = {}
    if batter_stats is None:
        batter_stats = {}

    pitch_types = list(pitch_avgs.keys())
    zones = label_encoders["zone_label"].classes_.tolist()

    scenario = pd.DataFrame(
        list(product(pitch_types, zones)),
        columns=["pitch_type", "zone_label"],
    )

    # Default game state
    scenario["balls"]           = "0"
    scenario["strikes"]         = "0"
    scenario["outs_when_up"]    = "0"
    scenario["inning"]          = "1"
    scenario["on_1b_occupied"]  = "False"
    scenario["on_2b_occupied"]  = "False"
    scenario["on_3b_occupied"]  = "False"
    scenario["stand"]           = "R"
    scenario["p_throws"]        = p_throws
    scenario["prev_pitch_type"] = "none"
    scenario["prev_zone_label"] = "none"

    # Pitch physics (release_speed, release_spin_rate, pfx_x, pfx_z, spin_axis, release_extension)
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

    # Batter aggregate stats
    for stat in ("batter_k_pct", "batter_bb_pct", "batter_whiff_pct", "batter_hard_hit_pct"):
        scenario[stat] = batter_stats.get(stat, 0.0)

    # New situation / context features
    scenario["pitch_number"]              = 1.0
    scenario["score_differential"]        = 0.0
    scenario["pitcher_pitches_today"]     = 1.0
    scenario["pitcher_days_rest"]         = 5.0
    scenario["pitcher_k_pct"]             = pitcher_k_pct
    scenario["pitcher_bb_pct"]            = pitcher_bb_pct
    scenario["batter_whiff_pct_vs_pitch"] = batter_stats.get("batter_whiff_pct_vs_pitch", 0.0)
    scenario["batter_k_pct_vs_pitch"]     = batter_stats.get("batter_k_pct_vs_pitch", 0.0)
    scenario["batter_hard_hit_pct_vs_pitch"] = batter_stats.get("batter_hard_hit_pct_vs_pitch", 0.0)
    scenario["batter_hard_hit_pct_vs_pitch_upper"]  = 0.0
    scenario["batter_hard_hit_pct_vs_pitch_middle"] = 0.0
    scenario["batter_hard_hit_pct_vs_pitch_lower"]  = 0.0
    scenario["park_run_factor"]           = 1.0

    return scenario


# Keep old name as alias for backwards compatibility with visualisation calls
def build_eovaldi_scenario(pitch_avgs, label_encoders):
    return build_pitcher_scenario(pitch_avgs, label_encoders, p_throws="R")


def predict_and_score(model, scenario_df, label_encoders, le_outcome=None):
    """Encode scenario, predict probs, score with RE24, return results."""
    encoded, _ = encode_categoricals(scenario_df, label_encoders=label_encoders, fit=False)
    X = build_feature_matrix(encoded)

    probs = model.predict_proba(X)
    class_names = le_outcome.classes_ if le_outcome is not None else model.classes_
    prob_df = pd.DataFrame(probs, columns=class_names)

    results = pd.concat(
        [scenario_df[["pitch_type", "zone_label"]].reset_index(drop=True), prob_df],
        axis=1,
    )

    # Ensure all outcome columns exist
    for col in ("called_strike", "whiff", "foul", "out", "hit", "ball", "walk_hbp"):
        if col not in results.columns:
            results[col] = 0.0

    # RE24 score at default state (0-0, 0 outs, bases empty)
    results["re24_score"] = results.apply(
        lambda r: compute_re24_score(
            {"called_strike": r["called_strike"], "whiff": r["whiff"],
             "foul": r["foul"], "ball": r["ball"], "out": r["out"],
             "hit": r["hit"], "walk_hbp": r["walk_hbp"]},
            balls=0, strikes=0, outs=0, on_1b=False, on_2b=False, on_3b=False,
        ),
        axis=1,
    )
    results["combo"] = results["pitch_type"] + "_" + results["zone_label"]

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_outcome_by_zone(results_df, outcome, title):
    """Grouped bar chart: one bar per pitch type within each zone."""
    pivot = results_df.pivot_table(index="zone_label", columns="pitch_type", values=outcome)
    ax = pivot.plot(kind="bar", figsize=(12, 5), width=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Probability")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Pitch Type")
    plt.tight_layout()
    plt.show()


def plot_top_combos(top_df):
    """Horizontal bar chart of top pitch+zone combos by RE24 score."""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["combo"], top_df["re24_score"], color="steelblue")
    ax.set_xlabel("RE24 Score (Run Prevention Value)")
    ax.set_title("Top 10 Pitch+Zone Combos by RE24 (Eovaldi vs RHB, 0-0)", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
