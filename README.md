# Pitch Sequence Optimization

A dynamic pitch recommendation system for MLB pitchers using 2024 Statcast data. Given a game situation (count, outs, baserunners, batter, inning), the model recommends the optimal pitch type and zone to minimize run expectancy.

Built for pitcher **Nathan Eovaldi** using the full 2024 MLB season as training context, with support for any pitcher/batter combination in the dataset.

---

## How It Works

1. **Data Collection** — Pulls every 2024 MLB pitch from Baseball Savant via `pybaseball`
2. **Feature Engineering** — Cleans and encodes count, base state, pitch history, fatigue, park factors, and batter/pitcher splits
3. **Model Training** — XGBoost classifier predicts pitch outcome probabilities (called strike, whiff, ball, hit, etc.)
4. **RE24 Scoring** — Outcomes are weighted by run expectancy tables to compute an expected run-value score per pitch-zone combo
5. **Web App** — Flask interface lets you simulate live plate appearances and see real-time recommendations

---

## Project Structure

```
├── pitch_by_pitch_collection.py  # Scrape Statcast data for every 2024 game
├── combine_data_rangers.py       # Clean, deduplicate, and engineer features
├── pitcher_pso.py                # Train XGBoost model, save pitch_sequence_model.pkl
├── dynamic_pitcher.py            # Game state engine and recommendation logic
├── heatmap_pitcher.py            # EDA visualizations (pitch heatmaps by type/zone)
├── utils.py                      # Shared utilities: RE24 tables, feature definitions, encoders
├── app.py                        # Flask web app
├── templates/index.html          # Frontend UI
├── static/style.css              # Styling
└── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

> **Note:** The trained model (`pitch_sequence_model.pkl`) and raw data files are not tracked in git due to size (~58MB and ~1.2GB respectively). To reproduce from scratch, follow the pipeline steps below.

---

## Running the Pipeline (from scratch)

```bash
# 1. Collect Statcast data for all 2024 games
python pitch_by_pitch_collection.py

# 2. Clean, deduplicate, and engineer features
python combine_data_rangers.py

# 3. Train the XGBoost model (outputs pitch_sequence_model.pkl)
python pitcher_pso.py
```

---

## Running the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

The app lets you:
- Select any pitcher/batter combination from the 2024 dataset
- Set count, outs, inning, baserunners, and score differential
- Simulate pitch outcomes to advance game state
- View ranked pitch+zone recommendations with RE24 scores and outcome probability breakdowns

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, XGBoost, matplotlib, Flask
- **Data Source** — MLB Statcast via [pybaseball](https://github.com/jldbc/pybaseball)
- **Model** — XGBoost classifier with stratified train/test split and sample weighting for class balance
- **Scoring** — RE24 run expectancy tables calibrated to 2024 MLB season averages

---

Landon Myhill · [github.com/landon-myhill](https://github.com/landon-myhill)
