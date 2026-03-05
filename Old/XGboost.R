# Full Model Creation Block for Nathan Eovaldi using XGBoost

setwd("~/Desktop/PSO/Pitcher")

# Load required libraries
library(tidyverse)
library(caret)
library(forcats)
library(purrr)
library(ggplot2)
library(knitr)
library(dplyr)
library(xgboost)

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

# Convert factors to dummy variables for XGBoost
train_data_dummies <- model.matrix(~ . - 1, data = train_data[, factor_cols])
test_data_dummies <- model.matrix(~ . - 1, data = test_data[, factor_cols])

# Prepare target variable
train_labels <- as.numeric(train_data$next_outcome) - 1
test_labels <- as.numeric(test_data$next_outcome) - 1

# Train XGBoost Model
dtrain <- xgb.DMatrix(data = train_data_dummies, label = train_labels)
dtest <- xgb.DMatrix(data = test_data_dummies, label = test_labels)

# Set parameters for XGBoost
params <- list(
  objective = "multi:softmax",  # Multi-class classification
  num_class = length(levels(train_data$next_outcome)), # Number of classes
  eval_metric = "merror",  # Multi-class error rate
  max_depth = 6,
  eta = 0.3,
  nthread = 2
)

# Train the model
model_eovaldi <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  print.every.n = 10
)

# Predict with XGBoost
preds <- predict(model_eovaldi, newdata = dtest)
preds <- factor(preds, levels = 0:(length(levels(train_data$next_outcome)) - 1),
                labels = levels(train_data$next_outcome))

# Confusion Matrix
conf_mat <- confusionMatrix(preds, test_data$next_outcome)
message("Nathan Eovaldi | Accuracy: ", round(conf_mat$overall["Accuracy"], 3))

saveRDS(model_eovaldi, "model_for_pitcher_nathan_eovaldi_xgboost.rds")

# ----------------------------

# 10) Weight Pitch Type/Zone Combinations by Frequency
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

# ----------------------------

# 12) Scenario Setup
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

# 13) Predict Probabilities
scenario_dummies <- model.matrix(~ . - 1, data = scenario_rows[, factor_cols])
pred_probs <- predict(model_eovaldi, newdata = scenario_dummies)

# Combine predictions with pitch/zone combos
results_df <- cbind(scenario_rows[, c("pitch_type", "zone_label")], pred_probs)

# ----------------------------

# Confusion Matrix and Metrics (Model Eval)
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

