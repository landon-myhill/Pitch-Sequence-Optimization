"""
Collect pitch-by-pitch Statcast data for the full 2024 MLB season.

Pipeline step 1: queries Baseball Savant day-by-day via pybaseball,
resolves player IDs to names, and saves per-team CSVs to games_by_team/.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast, playerid_reverse_lookup

# Avoid repeated lookups
_player_cache = {}


def get_player_name(player_id):
    """Resolve a Statcast player ID to a full name, with caching."""
    if pd.isna(player_id):
        return None

    try:
        player_id_int = int(player_id)
    except (ValueError, TypeError):
        return str(player_id)

    if player_id_int in _player_cache:
        return _player_cache[player_id_int]

    try:
        result = playerid_reverse_lookup([player_id_int])
        if not result.empty:
            full_name = f"{result.iloc[0]['name_first']} {result.iloc[0]['name_last']}"
        else:
            full_name = str(player_id_int)
    except Exception:
        full_name = str(player_id_int)

    _player_cache[player_id_int] = full_name
    return full_name


def main():
    season_start = datetime(2024, 3, 28)
    season_end = datetime(2024, 9, 30)
    base_folder = "games_by_team"
    os.makedirs(base_folder, exist_ok=True)

    current_date = season_start
    while current_date <= season_end:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date: {date_str}")

        try:
            day_data = statcast(start_dt=date_str, end_dt=date_str)
            if day_data.empty:
                print(f"No games found on {date_str}.")
            else:
                for game_id, game_data in day_data.groupby("game_pk"):
                    home_team = game_data["home_team"].iloc[0]
                    away_team = game_data["away_team"].iloc[0]

                    if "batter" in game_data.columns:
                        game_data["batter_name"] = game_data["batter"].apply(get_player_name)
                    if "pitcher" in game_data.columns:
                        game_data["pitcher_name"] = game_data["pitcher"].apply(get_player_name)

                    for team in [home_team, away_team]:
                        team_folder = os.path.join(base_folder, team)
                        os.makedirs(team_folder, exist_ok=True)

                        file_name = f"{date_str}_{game_id}.csv"
                        file_path = os.path.join(team_folder, file_name)
                        game_data.to_csv(file_path, index=False)
                        print(f"Saved game {game_id} for {team} to {file_path}")
        except Exception as e:
            print(f"Error processing {date_str}: {e}")

        current_date += timedelta(days=1)


if __name__ == "__main__":
    main()
