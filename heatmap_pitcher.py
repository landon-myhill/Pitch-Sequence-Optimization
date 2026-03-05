"""
Exploratory Data Analysis: pitch location heatmaps and distributions.

Generates visualizations for Nathan Eovaldi's pitch tendencies:
  - Pitch type distribution
  - Release speed distribution
  - Pitch location heatmaps (overall + per pitch type)
  - Combined 2×3 subplot grid saved as PNG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Strike zone boundaries (feet from center of plate)
STRIKE_ZONE = {"x": (-0.85, 0.85), "z": (1.4, 3.6)}


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    data = pd.read_csv("combined_pitch_data.csv", low_memory=False)
    eovaldi = data[
        data["pitcher_name"].str.contains("nathan eovaldi", case=False, na=False)
    ].copy()
    print(f"Eovaldi pitches: {len(eovaldi)}")

    pitch_types = sorted(eovaldi["pitch_type"].dropna().unique())

    # ── Pitch type distribution ───────────────────────────────────────────
    counts = eovaldi["pitch_type"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color="steelblue")
    ax.set_title("Pitch Type Distribution — Nathan Eovaldi", fontsize=14)
    ax.set_xlabel("Pitch Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ── Release speed distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eovaldi["release_speed"].dropna(), bins=30, color="steelblue", edgecolor="black")
    ax.set_title("Release Speed Distribution — Nathan Eovaldi", fontsize=14)
    ax.set_xlabel("Release Speed (mph)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ── Overall pitch location heatmap ────────────────────────────────────
    _plot_heatmap(eovaldi, "Pitch Location Heatmap — Nathan Eovaldi")

    # ── Plate X / Plate Z histograms ─────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(eovaldi["plate_x"].dropna(), bins=60, color="steelblue", edgecolor="black")
    ax1.set_title("Pitch Location Distribution (X)")
    ax1.set_xlabel("Plate X Position")
    ax1.set_ylabel("Count")

    ax2.hist(eovaldi["plate_z"].dropna(), bins=60, color="steelblue", edgecolor="black")
    ax2.set_title("Pitch Location Distribution (Z)")
    ax2.set_xlabel("Plate Z Position")
    ax2.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ── Individual pitch type heatmaps ────────────────────────────────────
    for pt in pitch_types:
        subset = eovaldi[eovaldi["pitch_type"] == pt]
        _plot_heatmap(subset, f"Pitch Location — Eovaldi — {pt}")

    # ── Combined grid (overall + each pitch type) ─────────────────────────
    labels = ["Overall"] + pitch_types
    ncols = 3
    nrows = (len(labels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, label in enumerate(labels):
        subset = eovaldi if label == "Overall" else eovaldi[eovaldi["pitch_type"] == label]
        ax = axes[i]
        ax.hist2d(
            subset["plate_x"].dropna(), subset["plate_z"].dropna(),
            bins=30, cmap="coolwarm", alpha=0.7,
        )
        _add_strike_zone(ax)
        ax.set_title(f"Eovaldi — {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Plate X")
        ax.set_ylabel("Plate Z")

    # Hide unused subplots
    for j in range(len(labels), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("combined_eovaldi_heatmaps.png", dpi=300, bbox_inches="tight")
    print("Saved combined_eovaldi_heatmaps.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_heatmap(df, title):
    """Single pitch location heatmap with strike zone overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hist2d(
        df["plate_x"].dropna(), df["plate_z"].dropna(),
        bins=30, cmap="coolwarm", alpha=0.7,
    )
    plt.colorbar(hb[3], ax=ax, label="Count")
    _add_strike_zone(ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Plate X Position")
    ax.set_ylabel("Plate Z Position")
    plt.tight_layout()
    plt.show()


def _add_strike_zone(ax):
    """Draw the strike zone rectangle."""
    sz = STRIKE_ZONE
    rect = patches.Rectangle(
        (sz["x"][0], sz["z"][0]),
        sz["x"][1] - sz["x"][0],
        sz["z"][1] - sz["z"][0],
        linewidth=1.5, edgecolor="black", facecolor="none",
    )
    ax.add_patch(rect)


if __name__ == "__main__":
    main()
