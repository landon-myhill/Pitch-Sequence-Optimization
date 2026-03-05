# Full Model Creation Block for Nathan Eovaldi

setwd("~/Desktop/PSO/Pitcher")

# Load required libraries
library(tidyverse)
library(caret)
library(forcats)
library(purrr)
library(ggplot2)
library(knitr)
library(dplyr)

# Load the dataset
data <- read_csv("combined_pitch_data_Rangers.csv")

# Zone mapping
zone_labels <- c(
  "1"  = "top_left", "2"  = "top_middle", "3"  = "top_right",
  "4"  = "middle_left", "5"  = "middle_middle", "6"  = "middle_right",
  "7"  = "bottom_left", "8"  = "bottom_middle", "9"  = "bottom_right",
  "11" = "up_left", "12" = "up_right", "13" = "down_left", "14" = "down_right"
)

# Clean and label data
data <- data %>%
  filter(events != "truncated_pa") %>%
  mutate(
    next_outcome = case_when(
      events %in% c("single","double","triple","home_run","field_error","catcher_interf") ~ "hit",
      events %in% c("strikeout","strikeout_double_play","field_out","force_out","double_play",
                    "grounded_into_double_play","sac_fly","sac_bunt","fielders_choice",
                    "fielders_choice_out","triple_play","sac_fly_double_play") ~ "out",
      events %in% c("walk","hit_by_pitch") ~ "walk_hbp",
      type == "B" ~ "ball",
      type == "S" ~ "strike",
      TRUE ~ "NonTerminalPitch"
    ),
    next_outcome = as.factor(next_outcome),
    zone_label = zone_labels[as.character(zone)],
    zone_label = if_else(is.na(zone_label), "unmapped_zone", zone_label)
  ) %>%
  filter(zone_label != "unmapped_zone")

# Focus on Eovaldi only
eovaldi_data <- data %>% filter(pitcher_name == "nathan eovaldi")

if (nrow(eovaldi_data) < 500) {
  stop("Not enough data for Nathan Eovaldi (only ", nrow(eovaldi_data), " rows).")
}

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(eovaldi_data$next_outcome, p = 0.7, list = FALSE)
train_data <- eovaldi_data[trainIndex, ]
test_data  <- eovaldi_data[-trainIndex, ]

# Prepare factor columns
factor_cols <- c("pitch_type", "zone_label", "balls", "strikes", "outs_when_up", "inning",
                 "on_1b_occupied", "on_2b_occupied", "on_3b_occupied")

for (col in factor_cols) {
  if (!col %in% names(train_data)) {
    message("Column '", col, "' not found in train_data. Removing it from formula.")
    factor_cols <- setdiff(factor_cols, col)
    next
  }
  if (!is.factor(train_data[[col]])) {
    train_data[[col]] <- as.factor(train_data[[col]])
  }
  if (n_distinct(train_data[[col]]) < 2) {
    message("Column '", col, "' has <2 levels. Removing from formula.")
    factor_cols <- setdiff(factor_cols, col)
  }
}

# Align factor levels in test set
for (col in factor_cols) {
  test_data[[col]] <- factor(test_data[[col]], levels = levels(train_data[[col]]))
}

# Build model formula
rhs <- paste(factor_cols, collapse = " + ")
model_formula <- as.formula(paste("next_outcome ~", rhs))

# Train Random Forest Model
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = TRUE)

model_eovaldi <- train(
  model_formula,
  data = train_data,
  method = "rf",
  trControl = fitControl,
  tuneLength = 3
)

# Evaluate and Save Model
preds <- predict(model_eovaldi, newdata = test_data)
conf_mat <- confusionMatrix(preds, test_data$next_outcome)
message("Nathan Eovaldi | Accuracy: ", round(conf_mat$overall["Accuracy"], 3))

saveRDS(model_eovaldi, "model_for_pitcher_nathan_eovaldi.rds")


























# 10) Weight Pitch Type/Zone Combinations by Frequency

# Count the occurrences of each pitch_type and zone_label combination
pitch_zone_counts <- data %>%
  count(pitch_type, zone_label)

# Calculate the frequency (higher frequency combinations will get higher weight)
pitch_zone_counts <- pitch_zone_counts %>%
  mutate(weight = 1 / n)  # Rare combinations get higher weight

# Merge the weight back to the main data
data <- data %>%
  left_join(pitch_zone_counts, by = c("pitch_type", "zone_label"))

# Normalize the weights so that they sum to 1
data$weight <- data$weight / sum(data$weight, na.rm = TRUE)

# 11) Scenario Setup
scenario_eovaldi <- data.frame(
  pitch_type = NA,
  zone_label = NA,  
  balls = 0,
  strikes = 0,
  on_1b_occupied = factor(FALSE, levels = levels(train_data$on_1b_occupied)),
  on_2b_occupied = factor(FALSE, levels = levels(train_data$on_2b_occupied)),
  on_3b_occupied = factor(FALSE, levels = levels(train_data$on_3b_occupied)),
  inning = 1,
  outs_when_up = 0
)

# Get pitch type and zone options from training data
pitch_types <- if ("pitch_type" %in% factor_cols) levels(train_data$pitch_type) else c()
zones       <- if ("zone_label" %in% factor_cols) levels(train_data$zone_label) else c()

# Create all pitch_type/zone_label combos with scenario info
scenario_rows <- expand.grid(
  pitch_type = pitch_types,
  zone_label = zones,
  stringsAsFactors = FALSE
) %>%
  mutate(
    balls = 0,
    strikes = 0,
    on_1b_occupied = factor(FALSE, levels = levels(train_data$on_1b_occupied)),
    on_2b_occupied = factor(FALSE, levels = levels(train_data$on_2b_occupied)),
    on_3b_occupied = factor(FALSE, levels = levels(train_data$on_3b_occupied)),
    inning = 1,
    outs_when_up = 0
  )

# Align factor levels before predicting
for (col in factor_cols) {
  if (col %in% names(scenario_rows)) {
    scenario_rows[[col]] <- factor(scenario_rows[[col]], levels = levels(train_data[[col]]))
  }
}

# 12) Predict Probabilities
pred_probs <- predict(model_eovaldi, newdata = scenario_rows, type = "prob")

# Combine predictions with pitch/zone combos
results_df <- cbind(scenario_rows[, c("pitch_type", "zone_label")], pred_probs)

# ---- 1) Add Combined Weighted Probability to Results ----
results_df <- results_df %>%
  left_join(pitch_zone_counts, by = c("pitch_type", "zone_label")) %>%
  mutate(
    # Weighted score: prioritize Strike > Out > Avoiding Hit
    combined_prob = (0.4 * strike) + (0.4 * out) + (0.2 * (1 - hit)),
    
    # Apply weight to the combined probability based on frequency
    weighted_combined_prob = combined_prob * weight,
    
    combo = paste(pitch_type, zone_label, sep = "_")
  )






# ---------------------------------------------
# 12) Probability by Zone and Pitch Type – Separate Plots for Out, Strike, Hit

# Melt the data into long format for plotting
results_long <- results_df %>%
  select(pitch_type, zone_label, out, strike, hit) %>%
  pivot_longer(cols = c(out, strike, hit), names_to = "outcome", values_to = "probability")

# Create filtered datasets
results_out <- results_long %>% filter(outcome == "out")
results_strike <- results_long %>% filter(outcome == "strike")
results_hit <- results_long %>% filter(outcome == "hit")

# --------------------------
# Plot 1: Probability of Out
ggplot(results_out, aes(x = zone_label, y = probability, fill = pitch_type)) +
  geom_col(position = position_dodge(width = 0.8)) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Probability of Out by Zone and Pitch Type",
    x = "Zone",
    y = "Probability",
    fill = "Pitch Type"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# --------------------------
# Plot 2: Probability of Strike
ggplot(results_strike, aes(x = zone_label, y = probability, fill = pitch_type)) +
  geom_col(position = position_dodge(width = 0.8)) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Probability of Strike by Zone and Pitch Type",
    x = "Zone",
    y = "Probability",
    fill = "Pitch Type"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# --------------------------
# Plot 3: Probability of Hit
ggplot(results_hit, aes(x = zone_label, y = probability, fill = pitch_type)) +
  geom_col(position = position_dodge(width = 0.8)) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Probability of Hit by Zone and Pitch Type",
    x = "Zone",
    y = "Probability",
    fill = "Pitch Type"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )







# ---- 1) Add Combined Weighted Probability to Results ----
results_df <- results_df %>%
  mutate(
    # Weighted score: prioritize Strike > Out > Avoiding Hit
    combined_prob = (0.6 * strike) + (0.4 * out),
    combo = paste(pitch_type, zone_label, sep = "_")
  )

# ---- 2) Print Results with Combined Probability ----
print(results_df %>% arrange(desc(combined_prob)) %>% head(10))

# ---- 3) Plot Top 10 Pitch+Zone Combos by Combined Score ----
top_combined <- results_df %>%
  arrange(desc(combined_prob)) %>%
  head(10)

ggplot(top_combined, aes(
  x = reorder(combo, combined_prob),
  y = combined_prob,
  fill = pitch_type
)) +
  geom_col() +
  coord_flip() +
  labs(
    x = "Pitch Type & Zone",
    y = "Combined Success Probability",
    title = "Top 10 Pitch+Zone Combos (Weighted: Strike > Out > Not a Hit)"
  ) +
  theme_minimal(base_size = 14)










# ---------------------------------------------
# 13) Confusion Matrix and Metrics (Model Eval)
conf_mat <- confusionMatrix(preds, test_data$next_outcome)

# Print confusion matrix and accuracy
print(conf_mat)

# Overall Metrics Table
overall_stats <- data.frame(
  Metric = c("Accuracy", "Kappa", "95% CI", "No Information Rate", "P-Value (Acc > NIR)"),
  Value = c(
    round(conf_mat$overall["Accuracy"], 4),
    round(conf_mat$overall["Kappa"], 3),
    paste0("[", round(conf_mat$overall["AccuracyLower"], 4), ", ", round(conf_mat$overall["AccuracyUpper"], 4), "]"),
    round(conf_mat$overall["AccuracyNull"], 4),
    format.pval(conf_mat$overall["AccuracyPValue"], digits = 3, eps = 1e-4)
  )
)

cat("### Model Performance Metrics – Nathan Eovaldi Pitch Outcome Classifier\n\n")
kable(overall_stats)

# Class-Specific Metrics Table
by_class <- as.data.frame(conf_mat$byClass)
class_names <- rownames(by_class)

class_metrics <- by_class %>%
  select(Sensitivity, `Pos Pred Value`, `Balanced Accuracy`) %>%
  mutate(
    Outcome = gsub("Class: ", "", class_names),
    Sensitivity = round(Sensitivity, 3),
    Precision = round(`Pos Pred Value`, 3),
    `Balanced Accuracy` = round(`Balanced Accuracy`, 3)
  ) %>%
  select(Outcome, Sensitivity, Precision, `Balanced Accuracy`)

cat("\n### Class-Specific Performance\n\n")
kable(class_metrics)







# 1. Evaluate the Random Forest model
preds <- predict(model_eovaldi, newdata = test_data)
conf_mat <- confusionMatrix(preds, test_data$next_outcome)
message("Nathan Eovaldi | Accuracy: ", round(conf_mat$overall["Accuracy"], 3))

# Print the confusion matrix
print(conf_mat)

# Overall metrics table
overall_stats <- data.frame(
  Metric = c("Accuracy", "Kappa", "95% CI", "No Information Rate", "P-Value (Acc > NIR)"),
  Value = c(
    round(conf_mat$overall["Accuracy"], 4),
    round(conf_mat$overall["Kappa"], 3),
    paste0("[", round(conf_mat$overall["AccuracyLower"], 4), ", ", round(conf_mat$overall["AccuracyUpper"], 4), "]"),
    round(conf_mat$overall["AccuracyNull"], 4),
    format.pval(conf_mat$overall["AccuracyPValue"], digits = 3, eps = 1e-4)
  )
)

cat("### Model Performance Metrics – Nathan Eovaldi Pitch Outcome Classifier\n\n")
kable(overall_stats)

# Class-Specific Performance Metrics
by_class <- as.data.frame(conf_mat$byClass)
class_names <- rownames(by_class)

class_metrics <- by_class %>%
  select(Sensitivity, `Pos Pred Value`, `Balanced Accuracy`) %>%
  mutate(
    Outcome = gsub("Class: ", "", class_names),
    Sensitivity = round(Sensitivity, 3),
    Precision = round(`Pos Pred Value`, 3),
    `Balanced Accuracy` = round(`Balanced Accuracy`, 3)
  ) %>%
  select(Outcome, Sensitivity, Precision, `Balanced Accuracy`)

cat("\n### Class-Specific Performance\n\n")
kable(class_metrics)












