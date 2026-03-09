import os
import pandas as pd
from datetime import datetime, timedelta
from pybaseball import statcast, playerid_reverse_lookup

# Avoid repeated lookups
player_cache = {}

def get_player_name(player_id):
    """
    Given a player ID, returns the player's full name using pybaseball's
    reverse lookup. The player ID is passed as a list to the lookup function.
    Caches results to avoid repeated lookups.
    """
    if pd.isna(player_id):
        return None

    try:
        player_id_int = int(player_id)
    except Exception as e:
        print(f"Error converting player_id {player_id}: {e}")
        return str(player_id)
    
    if player_id_int in player_cache:
        return player_cache[player_id_int]
    
    try:
        # Pass the player_id as a list
        result = playerid_reverse_lookup([player_id_int])
        if not result.empty:
            full_name = f"{result.iloc[0]['name_first']} {result.iloc[0]['name_last']}"
            print(f"Found name for {player_id_int}: {full_name}")
        else:
            print(f"Lookup returned empty for {player_id_int}")
            full_name = str(player_id_int)
    except Exception as e:
        print(f"Error looking up player_id {player_id_int}: {e}")
        full_name = str(player_id_int)
    
    player_cache[player_id_int] = full_name
    return full_name

# Define season start and end dates 
season_start = datetime(2024, 3, 28)  
season_end   = datetime(2024, 9, 30)   

# Define the base folder 
base_folder = 'games_by_team'
os.makedirs(base_folder, exist_ok=True)

# Loop through each day of the season
current_date = season_start
while current_date <= season_end:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\nProcessing date: {date_str}")
    
    try:
        # Query Statcast data for the current date
        day_data = statcast(start_dt=date_str, end_dt=date_str)
        if day_data.empty:
            print(f"No games found on {date_str}.")
        else:
            # Only keep games involving these teams
            TARGET_TEAMS = {"TEX", "HOU"}

            # Process each game individually (grouped by 'game_pk')
            for game_id, game_data in day_data.groupby('game_pk'):
                # Extract team information (adjust column names if needed)
                home_team = game_data['home_team'].iloc[0]
                away_team = game_data['away_team'].iloc[0]

                # Skip games that don't involve a target team
                if home_team not in TARGET_TEAMS and away_team not in TARGET_TEAMS:
                    continue

                # Add actual player names
                if 'batter' in game_data.columns:
                    game_data['batter_name'] = game_data['batter'].apply(get_player_name)
                if 'pitcher' in game_data.columns:
                    game_data['pitcher_name'] = game_data['pitcher'].apply(get_player_name)

                # Save the updated game data
                for team in [home_team, away_team]:
                    team_folder = os.path.join(base_folder, team)
                    os.makedirs(team_folder, exist_ok=True)

                    file_name = f"{date_str}_{game_id}.csv"
                    file_path = os.path.join(team_folder, file_name)

                    game_data.to_csv(file_path, index=False)
                    print(f"Saved game {game_id} for team {team} to {file_path}")
    except Exception as e:
        print(f"Error processing {date_str}: {e}")
    
    # Move to the next day
    current_date += timedelta(days=1)
