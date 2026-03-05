setwd("~/Desktop/PSO/Pitcher")

library(tidyverse)
library(caret)
library(ggplot2)

# Load the trained model for Nathan Eovaldi
model_eovaldi <- readRDS("model_for_pitcher_nathan_eovaldi.rds")

# Define the initial game state
game_state <- list(
  balls = 0,
  strikes = 0,
  outs = 0,
  inning = 1,
  on_1b = FALSE,
  on_2b = FALSE,
  on_3b = FALSE,
  pitch_history = data.frame(pitch_type = character(), zone_label = character(), outcome = character()),
  batters_faced = 0,  # Track the number of batters faced
  times_through_order = 1  # Start at 1 (first time through the order)
)

# Pitch types and zones learned from the base code
pitch_types <- c("FF", "FS", "CU", "SL", "FC")  # Learned pitch types: Fastball (FF), Slider (SL), Curveball (CU), Cutter (FC), Splitter (FS)
zones <- c("top_left", "top_middle", "top_right", "middle_left", "middle_middle", 
           "middle_right", "bottom_left", "bottom_middle", "bottom_right", 
           "up_left", "up_right", "down_left", "down_right")  # Zones learned from base code

# Function to recommend the next pitch based on current game state
recommend_next_pitch <- function(state, model, pitch_types, zones) {
  # Ensure all factors have consistent levels as in the training data
  input <- expand.grid(
    pitch_type = pitch_types,
    zone_label = zones,
    balls = factor(state$balls, levels = c(0, 1, 2, 3)),
    strikes = factor(state$strikes, levels = c(0, 1, 2)),
    outs_when_up = factor(state$outs, levels = c(0, 1, 2)),
    inning = factor(state$inning, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9)),  # Treat inning as a factor
    on_1b_occupied = factor(state$on_1b, levels = c(TRUE, FALSE)),
    on_2b_occupied = factor(state$on_2b, levels = c(TRUE, FALSE)),
    on_3b_occupied = factor(state$on_3b, levels = c(TRUE, FALSE)),
    times_through_order = factor(state$times_through_order, levels = c(1, 2, 3, 4))  # Include times_through_order as a factor
  )
  
  # Predict outcomes using the model
  probs <- predict(model, newdata = input, type = "prob")
  
  # Add probabilities and calculate combined score
  input <- input %>%
    mutate(
      strike = probs$strike,
      out = probs$out,
      hit = probs$hit,
      ball = probs$ball,
      combined = (0.6 * strike) + (0.4 * out),
      combo = paste(pitch_type, zone_label, sep = "_")
    )
  
  best_recommendations <- input %>% arrange(desc(combined)) %>% head(10)  # Top 10 recommendations
  
  return(best_recommendations)
}

# Function to update the game state after each pitch outcome
update_state <- function(state, outcome, pitch_type, zone_label) {
  state$pitch_history <- rbind(state$pitch_history, data.frame(
    pitch_type = pitch_type,
    zone_label = zone_label,
    outcome = outcome
  ))
  
  # Update state based on outcome
  if (outcome == "strike") {
    state$strikes <- state$strikes + 1
    if (state$strikes == 3) {
      state$outs <- min(state$outs + 1, 3)  # A maximum of 3 outs per batter
      state$strikes <- 0  # Reset strikes after 3
    }
  } else if (outcome == "ball") {
    state$balls <- state$balls + 1
    if (state$balls == 4) {
      state$balls <- 0  # Reset balls after 4
      state$strikes <- 0  # Reset strikes if walked
    }
  } else if (outcome == "out") {
    state$outs <- min(state$outs + 1, 3)  # A maximum of 3 outs per batter
    state$balls <- 0
    state$strikes <- 0
  } else if (outcome == "hit") {
    state$balls <- 0
    state$strikes <- 0
  }
  
  # Update the batter count after each pitch outcome (out, hit, or walk)
  state$batters_faced <- state$batters_faced + 1
  
  # Check if we've faced 9 batters (full turn through the order)
  if (state$batters_faced == 9) {
    state$times_through_order <- min(state$times_through_order + 1, 4)  # Limit to 4 times through the order
    state$batters_faced <- 0  # Reset batters faced after 9 batters
  }
  
  return(state)
}

# Initial prediction and game state update (Example)
best_recommendations <- recommend_next_pitch(game_state, model_eovaldi, pitch_types, zones)

# Plot the initial (original) graph with a bold, larger title
ggplot(best_recommendations, aes(x = reorder(combo, combined), y = combined, fill = combo)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Initial Top 10 Recommended Pitches for Nathan Eovaldi", 
       x = "Pitch Type and Zone Combination", y = "Combined Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Adjust angle for better readability
        plot.title = element_text(size = 18, face = "bold"))  # Increase size and make title bold






# Simulate the first pitch outcome (e.g., "strike")
outcome <- "strike"
pitch_type <- "SL"
zone_label <- "middle_right"

# Update game state after the pitch
game_state <- update_state(game_state, outcome, pitch_type, zone_label)

# After updating the game state, get the next top 10 pitch recommendations
best_recommendations <- recommend_next_pitch(game_state, model_eovaldi, pitch_types, zones)

# Plot the updated graph with dynamic title and bold, larger text
ggplot(best_recommendations, aes(x = reorder(combo, combined), y = combined, fill = combo)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = paste("Previous Pitch Type: ", pitch_type, " Location: ", zone_label, 
                     " Count: ", game_state$balls, "-", game_state$strikes, 
                     " Times Through Order: ", game_state$times_through_order), 
       x = "Pitch Type and Zone Combination", y = "Combined Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Adjust angle for better readability
        plot.title = element_text(size = 18, face = "bold"))  # Make title bigger and bold
