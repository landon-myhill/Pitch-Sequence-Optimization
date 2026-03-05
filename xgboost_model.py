"""
Train an XGBoost model to predict pitch outcomes.

Same methodology as pitcher_pso.py (pooled training, RE24 scoring)
but uses gradient-boosted trees instead of Random Forest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from utils import (
    load_and_prepare_data,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, ALL_FEATURES,
    compute_re24_score,
)
from pitcher_pso import (
    encode_categoricals, build_feature_matrix,
    build_eovaldi_scenario, plot_outcome_by_zone, plot_top_combos,
)


def main():
    # ── 1) Load data ──────────────────────────────────────────────────────
    data = load_and_prepare_data("combined_pitch_data.csv", pitcher_name=None)
    print(f"Total rows (all pitchers): {len(data)}")

    eovaldi = load_and_prepare_data("combined_pitch_data.csv", pitcher_name="nathan eovaldi")
    eovaldi_pitch_avgs = (
        eovaldi.groupby("pitch_type")[NUMERIC_FEATURES].mean().to_dict(orient="index")
    )

    # ── 2) Encode & split ─────────────────────────────────────────────────
    data_enc, label_encoders = encode_categoricals(data, fit=True)

    # XGBoost needs numeric target labels
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(data_enc["pitch_outcome"])

    X = build_feature_matrix(data_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=123, stratify=y_encoded,
    )

    # ── 3) Train ──────────────────────────────────────────────────────────
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": len(target_le.classes_),
        "eval_metric": "merror",
        "max_depth": 6,
        "eta": 0.3,
        "nthread": 2,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=10,
    )

    # ── 4) Evaluate ───────────────────────────────────────────────────────
    pred_probs = model.predict(dtest)
    preds = target_le.inverse_transform(pred_probs.argmax(axis=1))
    y_test_labels = target_le.inverse_transform(y_test)

    print(f"\nXGBoost Accuracy: {accuracy_score(y_test_labels, preds):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, preds, labels=target_le.classes_))
    print("\nClassification Report:")
    print(classification_report(y_test_labels, preds))

    # ── 5) Save ───────────────────────────────────────────────────────────
    artifact = {
        "model": model,
        "label_encoders": label_encoders,
        "target_le": target_le,
        "eovaldi_pitch_avgs": eovaldi_pitch_avgs,
    }
    joblib.dump(artifact, "model_for_pitcher_nathan_eovaldi_xgboost.pkl")
    print("Saved model_for_pitcher_nathan_eovaldi_xgboost.pkl")

    # ── 6) Scenario predictions ───────────────────────────────────────────
    scenario = build_eovaldi_scenario(eovaldi_pitch_avgs, label_encoders)
    results = predict_and_score_xgb(model, target_le, scenario, label_encoders)

    # ── 7) Visualize ──────────────────────────────────────────────────────
    for outcome in ("out", "strike", "hit"):
        plot_outcome_by_zone(results, outcome, f"XGBoost: P({outcome}) by Zone & Pitch (Eovaldi)")

    top10 = results.nlargest(10, "re24_score")
    print("\nTop 10 Pitch+Zone Combos (XGBoost, RE24):")
    print(top10[["combo", "re24_score", "strike", "out", "hit"]].to_string(index=False))
    plot_top_combos(top10)


def predict_and_score_xgb(model, target_le, scenario_df, label_encoders):
    """XGBoost-specific prediction: uses DMatrix and target_le for class names."""
    encoded, _ = encode_categoricals(scenario_df, label_encoders=label_encoders, fit=False)
    X = build_feature_matrix(encoded)

    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)
    prob_df = pd.DataFrame(probs, columns=target_le.classes_)

    results = pd.concat(
        [scenario_df[["pitch_type", "zone_label"]].reset_index(drop=True), prob_df],
        axis=1,
    )

    for col in ("strike", "out", "hit", "ball", "walk_hbp"):
        if col not in results.columns:
            results[col] = 0.0

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


if __name__ == "__main__":
    main()
