# 1) Setup
setwd("~/Desktop/PSO/Pitcher")  

library(tidyverse)
library(caret)
library(randomForest)
library(forcats)

# 2) Read All CSVs
data_dir <- "~/Desktop/PSO/games_by_team"
csv_files <- list.files(
  path = data_dir, 
  pattern = "\\.csv$", 
  recursive = TRUE, 
  full.names = TRUE
)
cat("Found", length(csv_files), "CSV files.\n")

# Combine 
pitch_data <- map_df(csv_files, read_csv) %>%
  filter(home_team == "TEX" | away_team == "TEX")

# 3) Remove Unwanted Columns
pitch_data_cleaned <- pitch_data %>%
  select(
    -c(
      spin_dir, 
      spin_rate_deprecated, 
      break_angle_deprecated, 
      break_length_deprecated,
      
      tfs_deprecated,
      tfs_zulu_deprecated,
      umpire,
      sv_id,
      
      vx0, vy0, vz0,
      ax, ay, az,
      
      fielder_2, fielder_3, fielder_4, 
      fielder_5, fielder_6, fielder_7,
      fielder_8, fielder_9,
      
      bat_speed,
      swing_length,
      
      pfx_x,
      pfx_z,
      hc_x,
      hc_y,
      
      estimated_ba_using_speedangle,
      estimated_woba_using_speedangle,
      woba_value,
      woba_denom,
      babip_value,
      iso_value,
      launch_speed_angle,
      estimated_slg_using_speedangle,
      delta_pitcher_run_exp,
      hyper_speed,
      home_win_exp,
      bat_win_exp
    )
  )

# 4) Cleanup & NA Handling

# (A) 'events' to "NonTerminalPitch"
pitch_data_cleaned$events <- as.factor(pitch_data_cleaned$events)
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(events = fct_explicit_na(events, na_level = "NonTerminalPitch"))

# (B) 'hit_location' to "NonTerminalLocation"
pitch_data_cleaned$hit_location <- as.factor(pitch_data_cleaned$hit_location)
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(hit_location = fct_explicit_na(hit_location, na_level = "NonTerminalLocation"))

# (C) 'bb_type' to "NoBB"
pitch_data_cleaned$bb_type <- as.factor(pitch_data_cleaned$bb_type)
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(bb_type = fct_explicit_na(bb_type, na_level = "NoBB"))

# 5) Convert on_1b, on_2b, on_3b to Boolean 
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(
    on_1b_occupied = !is.na(on_1b),
    on_2b_occupied = !is.na(on_2b),
    on_3b_occupied = !is.na(on_3b)
  ) %>%
  select(-on_1b, -on_2b, -on_3b)

# 6) Replace NA with "NoBB" for Certain Numeric Columns

# (A) hit_distance_sc
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(
    hit_distance_sc = as.character(hit_distance_sc),
    hit_distance_sc = if_else(
      is.na(hit_distance_sc),
      "NoBB",
      hit_distance_sc
    ),
    hit_distance_sc = as.factor(hit_distance_sc)
  )

# (B) launch_speed
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(
    launch_speed = as.character(launch_speed),
    launch_speed = if_else(
      is.na(launch_speed),
      "NoBB",
      launch_speed
    ),
    launch_speed = as.factor(launch_speed)
  )

# (C) launch_angle
pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(
    launch_angle = as.character(launch_angle),
    launch_angle = if_else(
      is.na(launch_angle),
      "NoBB",
      launch_angle
    ),
    launch_angle = as.factor(launch_angle)
  )

# 7) NA -> "NoPriorGame"

pitch_data_cleaned$pitcher_days_since_prev_game <- 
  as.factor(pitch_data_cleaned$pitcher_days_since_prev_game)
pitch_data_cleaned$batter_days_since_prev_game <-
  as.factor(pitch_data_cleaned$batter_days_since_prev_game)

pitch_data_cleaned <- pitch_data_cleaned %>%
  mutate(
    pitcher_days_since_prev_game = fct_explicit_na(
      pitcher_days_since_prev_game, 
      na_level = "NoPriorGame"
    ),
    batter_days_since_prev_game  = fct_explicit_na(
      batter_days_since_prev_game,  
      na_level = "NoPriorGame"
    )
  )

# 8) Final Check & Write to CSV
glimpse(pitch_data_cleaned)
write_csv(pitch_data_cleaned, "combined_pitch_data.csv")
