"""
Train a Random Forest model to predict pitch outcomes.

Key design decisions:
  - Trains on ALL pitchers (pooled) for much larger sample size (~50k+ rows).
  - Uses batter handedness, pitch sequencing, and pitch physics as features.
  - Scores recommendations with RE24 run expectancy (context-aware).
  - Generates Eovaldi-specific predictions using his pitch physics averages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from itertools import product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from utils import (
    load_and_prepare_data,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    compute_re24_score,
)


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

    eovaldi = load_and_prepare_data("combined_pitch_data.csv", pitcher_name="nathan eovaldi")
    print(f"Eovaldi rows: {len(eovaldi)}")

    # Eovaldi's average pitch physics per pitch type
    eovaldi_pitch_avgs = (
        eovaldi.groupby("pitch_type")[NUMERIC_FEATURES].mean().to_dict(orient="index")
    )

    # ── 2) Encode & split ─────────────────────────────────────────────────
    data_enc, label_encoders = encode_categoricals(data, fit=True)
    X = build_feature_matrix(data_enc)
    y = data_enc["pitch_outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y,
    )

    # ── 3) Train ──────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    rf.fit(X_train, y_train)

    # ── 4) Evaluate ───────────────────────────────────────────────────────
    preds = rf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds, labels=rf.classes_))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # ── 5) Save ───────────────────────────────────────────────────────────
    artifact = {
        "model": rf,
        "label_encoders": label_encoders,
        "eovaldi_pitch_avgs": eovaldi_pitch_avgs,
    }
    joblib.dump(artifact, "model_for_pitcher_nathan_eovaldi.pkl")
    print("Saved model_for_pitcher_nathan_eovaldi.pkl")

    # ── 6) Scenario predictions for Eovaldi ───────────────────────────────
    scenario = build_eovaldi_scenario(eovaldi_pitch_avgs, label_encoders)
    results = predict_and_score(rf, scenario, label_encoders)

    # ── 7) Visualize ──────────────────────────────────────────────────────
    for outcome in ("out", "strike", "hit"):
        plot_outcome_by_zone(results, outcome, f"P({outcome}) by Zone & Pitch Type (Eovaldi)")

    top10 = results.nlargest(10, "re24_score")
    print("\nTop 10 Pitch+Zone Combos (RE24 — higher = better for pitcher):")
    print(top10[["combo", "re24_score", "strike", "out", "hit"]].to_string(index=False))
    plot_top_combos(top10)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario building & prediction
# ═══════════════════════════════════════════════════════════════════════════════

def build_eovaldi_scenario(pitch_avgs, label_encoders):
    """
    Build a scenario DataFrame with all pitch_type × zone combos
    at a default game state (0-0 count, 0 outs, bases empty, vs RHB).
    """
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
    scenario["p_throws"]        = "R"       # Eovaldi is RHP
    scenario["prev_pitch_type"] = "none"
    scenario["prev_zone_label"] = "none"

    # Eovaldi's average pitch physics per pitch type
    for col in NUMERIC_FEATURES:
        scenario[col] = scenario["pitch_type"].apply(
            lambda pt: pitch_avgs.get(pt, {}).get(col, 0.0)
        )

    return scenario


def predict_and_score(model, scenario_df, label_encoders):
    """Encode scenario, predict probs, score with RE24, return results."""
    encoded, _ = encode_categoricals(scenario_df, label_encoders=label_encoders, fit=False)
    X = build_feature_matrix(encoded)

    probs = model.predict_proba(X)
    prob_df = pd.DataFrame(probs, columns=model.classes_)

    results = pd.concat(
        [scenario_df[["pitch_type", "zone_label"]].reset_index(drop=True), prob_df],
        axis=1,
    )

    # Ensure all outcome columns exist
    for col in ("strike", "out", "hit", "ball", "walk_hbp"):
        if col not in results.columns:
            results[col] = 0.0

    # RE24 score at default state (0-0, 0 outs, bases empty)
    results["re24_score"] = results.apply(
        lambda r: compute_re24_score(
            {"strike": r["strike"], "ball": r["ball"], "out": r["out"],
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_df["combo"], top_df["re24_score"], color="steelblue")
    ax.set_xlabel("RE24 Score (Run Prevention Value)")
    ax.set_title("Top 10 Pitch+Zone Combos by RE24 (Eovaldi vs RHB, 0-0)", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
