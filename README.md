# Pitch Sequence Optimization

A machine learning system that recommends optimal pitch selections for MLB pitchers in real time. Given a game situation — count, outs, baserunners, batter, inning — it predicts pitch outcome probabilities and scores each pitch+zone combination using **RE24 run expectancy** to find the option that best minimizes expected runs.

Built with the full **2024 MLB Statcast** season (~700K pitches across all 30 teams). Supports any pitcher/batter matchup in the dataset through an interactive Flask web app.

---

## Key Features

- **Context-aware recommendations** — factors in count, outs, baserunners, inning, score differential, pitcher fatigue, and batter/pitcher splits
- **RE24 scoring** — weights outcome probabilities by run expectancy tables calibrated to the 2024 MLB season, so recommendations reflect actual game leverage
- **40+ engineered features** — pitch physics, sequencing (previous pitch type/zone), fatigue (cumulative pitch count), days rest, park factors, and per-batter pitch-type splits
- **Interactive web app** — simulate full plate appearances, advance game state pitch-by-pitch, and view ranked recommendations with probability breakdowns

---

## Architecture

```
                    Statcast API
                        |
            +-----------+-----------+
            |                       |
   pitch_by_pitch_collection.py     |
   (scrape 2024 season day-by-day)  |
            |                       |
      games_by_team/                |
      (raw CSVs by team)            |
            |                       |
      combine_data.py               |
      (dedupe, clean, engineer      |
       lag/fatigue/rest features)   |
            |                       |
   combined_pitch_data.csv          |
            |                       |
      pitcher_pso.py                |
      (train XGBoost classifier,    |
       compute stats, save model)   |
            |                       |
   pitch_sequence_model.pkl         |
            |                       |
      dynamic_pitcher.py            |
      (game state engine +          |
       recommendation logic)        |
            |                       |
          app.py  <-----------------+
          (Flask web server)
            |
      templates/index.html
      static/style.css
```

---

## How It Works

1. **Data Collection** — Pulls every 2024 MLB pitch from Baseball Savant via `pybaseball`
2. **Feature Engineering** — Cleans and encodes count, base state, pitch sequencing, fatigue, park factors, and batter/pitcher splits
3. **Model Training** — XGBoost multi-class classifier predicts pitch outcome probabilities across 7 classes: called strike, whiff, foul, ball, out, hit, walk/HBP
4. **RE24 Scoring** — Each pitch+zone combo is scored by expected run-value change given the current game state (count, outs, runners). Higher score = better for the pitcher
5. **Web App** — Flask interface lets you set up any game situation, simulate pitches, and see real-time ranked recommendations

### Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (multi-class, `mlogloss`) |
| Training data | All 2024 MLB pitches (pooled across pitchers) |
| Features | 40+ (14 categorical + 26 numeric) |
| Outcome classes | `called_strike`, `whiff`, `foul`, `ball`, `out`, `hit`, `walk_hbp` |
| Class balancing | Sample weights via `compute_sample_weight("balanced")` |
| Train/test split | 70/30 stratified |

---

## Project Structure

```
pitcher_pso.py                # Train XGBoost model, save artifact
dynamic_pitcher.py            # Game state engine + recommendation logic
utils.py                      # RE24 tables, feature definitions, stat computations
app.py                        # Flask web server
combine_data.py               # Clean, deduplicate, engineer features
pitch_by_pitch_collection.py  # Scrape Statcast data for 2024 season
heatmap_pitcher.py            # EDA visualizations (pitch location heatmaps)
templates/index.html          # Interactive web UI
static/style.css              # Styling
```

---

## Demo

https://github.com/user-attachments/assets/demo.mp4

<video src="demo.mp4" controls width="100%"></video>

Watch a full at-bat simulation: Eovaldi vs Jose Altuve — 4-pitch strikeout with real-time recommendation updates after each pitch.

---

## Quick Start

The trained model is included via Git LFS, so the app works right after cloning:

```bash
git clone https://github.com/landon-myhill/Pitch-Sequence-Optimization.git
cd Pitch-Sequence-Optimization
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

> If `git clone` didn't pull the model (e.g. Git LFS wasn't installed), run `git lfs pull` first.

The app lets you:
- Select any pitcher/batter combination from the 2024 dataset
- Set count, outs, inning, baserunners, and score differential
- Simulate pitch outcomes to advance the game state
- View the top 10 pitch+zone recommendations ranked by RE24 score, with full outcome probability breakdowns

---

## Retraining the Model (Optional)

The raw Statcast data (~1.2 GB) is not included. To retrain from scratch:

```bash
# 1. Collect Statcast data for all 2024 games (~6 hours, rate-limited)
python pitch_by_pitch_collection.py

# 2. Clean, deduplicate, and engineer features
python combine_data.py

# 3. Train the XGBoost model (overwrites pitch_sequence_model.pkl)
python pitcher_pso.py
```

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost, matplotlib, Flask
- **Data** — MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball)
- **Scoring** — RE24 run expectancy tables calibrated to 2024 MLB season averages

---

## License

MIT

---

Landon Myhill · [github.com/landon-myhill](https://github.com/landon-myhill)
