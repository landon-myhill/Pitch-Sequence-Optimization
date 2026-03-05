# Setup
setwd("~/Desktop/PSO/Pitcher")

library(dplyr)
library(ggplot2)
library(patchwork)  
data <- read.csv("combined_pitch_data_rangers.csv")

# Filter for Nathan Eovaldi
eovaldi_data <- data[grep("nathan eovaldi", data$pitcher_name, ignore.case = TRUE), ]

# Check distribution  pitch types
pitch_type_dist <- eovaldi_data %>%
  group_by(pitch_type) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Plot Pitch Type Distribution (Histogram)
pitch_type_histogram <- ggplot(pitch_type_dist, aes(x = reorder(pitch_type, count), y = count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Pitch Type Distribution for Nathan Eovaldi", x = "Pitch Type", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot Release Speed (Histogram)
release_speed_histogram <- ggplot(eovaldi_data, aes(x = release_speed)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Release Speed for Nathan Eovaldi", x = "Release Speed (mph)", y = "Count")

# Plot Pitch Location Heatmap 
pitch_location_heatmap <- ggplot(eovaldi_data, aes(x = plate_x, y = plate_z)) +
  geom_bin2d(bins = 30, alpha = 0.7) + 
  scale_fill_gradient(low = "blue", high = "red") + 
  geom_rect(aes(xmin = -0.85, xmax = 0.85, ymin = 1.4, ymax = 3.6), 
            color = "black", size = 1.5, fill = NA) + 
  labs(title = "Pitch Location Heatmap for Nathan Eovaldi", 
       x = "Plate X Position", 
       y = "Plate Z Position") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, size = 14)
  )

# Plot Pitch Zone Histogram (X-axis)
pitch_zone_histogram <- ggplot(eovaldi_data, aes(x = plate_x)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "black") +
  labs(title = "Pitch Location Distribution (X)", x = "Plate X Position", y = "Count") +
  theme_minimal()

# Plot Pitch Zone Histogram (Z-axis)
pitch_zone_histogram_z <- ggplot(eovaldi_data, aes(x = plate_z)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "black") +
  labs(title = "Pitch Location Distribution (Z)", x = "Plate Z Position", y = "Count") +
  theme_minimal()

# Print each plot individually
print(pitch_type_histogram)
print(release_speed_histogram)
print(pitch_location_heatmap)
print(pitch_zone_histogram)
print(pitch_zone_histogram_z)


# Plot pitch location 
ggplot(eovaldi_data, aes(x = plate_x, y = plate_z)) +
  # Heatmap for all pitches
  geom_bin2d(bins = 30, alpha = 0.7) + 
  scale_fill_gradient(low = "blue", high = "red") + 
  # Strike zone 
  geom_rect(aes(xmin = -0.85, xmax = 0.85, ymin = 1.4, ymax = 3.6), 
            color = "black", size = 1.5, fill = NA) + 
  labs(title = "Pitch Location Heatmap for Nathan Eovaldi", 
       x = "Plate X Position", 
       y = "Plate Z Position") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, size = 14)
  )








# Plot for each pitch type 
library(ggplot2)

for(pitch in unique(eovaldi_data$pitch_type)) {
    pitch_data <- eovaldi_data[eovaldi_data$pitch_type == pitch, ]
  
  # Plot
  p <- ggplot(pitch_data, aes(x = plate_x, y = plate_z)) +
    # Heatmap for each pitch type
    geom_bin2d(bins = 30, alpha = 0.7) + 
    scale_fill_gradient(low = "blue", high = "red") + 
    # Strike zone 
    geom_rect(aes(xmin = -0.85, xmax = 0.85, ymin = 1.4, ymax = 3.6), 
              color = "black", size = 1.5, fill = NA) + 
    labs(title = paste("Pitch Location Heatmap for Nathan Eovaldi -", pitch), 
         x = "Plate X Position", 
         y = "Plate Z Position") +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      plot.title = element_text(hjust = 0.5, size = 14)
    )
  
  print(p)
  
  
  
  
# Pitch type count 
pitch_outcome_analysis <- eovaldi_data %>%
  group_by(pitch_type, events) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

head(pitch_outcome_analysis)



# Combine Plots
library(ggplot2)
library(patchwork)  

pitch_types <- unique(eovaldi_data$pitch_type)
plots <- list()

for (pitch in pitch_types) {
  pitch_data <- eovaldi_data[eovaldi_data$pitch_type == pitch, ]
  
  p <- ggplot(pitch_data, aes(x = plate_x, y = plate_z)) +
    geom_bin2d(bins = 30, alpha = 0.7) + 
    scale_fill_gradient(low = "blue", high = "red") + 
    geom_rect(aes(xmin = -0.85, xmax = 0.85, ymin = 1.4, ymax = 3.6), 
              color = "black", size = 1.2, fill = NA) + 
    labs(title = paste("Nathan Eovaldi -", pitch),
         x = "X-axis", 
         y = "Y-axis") +
    theme_minimal() +
    theme(
      axis.title = element_text(size = 10),
      plot.title = element_text(hjust = 0.5, size = 12)
    )

  plots[[pitch]] <- p
}

combined_plot <- wrap_plots(plots, ncol = 2)
print(combined_plot)












# Setup
setwd("~/Desktop/PSO/Pitcher")

library(dplyr)
library(ggplot2)
library(patchwork)  
data <- read.csv("combined_pitch_data_rangers.csv")

# Filter for Nathan Eovaldi
eovaldi_data <- data[grep("nathan eovaldi", data$pitcher_name, ignore.case = TRUE), ]

# Overall heatmap for all pitches
overall_heatmap <- ggplot(eovaldi_data, aes(x = plate_x, y = plate_z)) +
  geom_bin2d(bins = 30, alpha = 0.7) + 
  scale_fill_gradient(low = "blue", high = "red") + 
  geom_rect(aes(xmin = -0.85, xmax = 0.85, ymin = 1.4, ymax = 3.6), 
            color = "black", size = 1.5, fill = NA) + 
  labs(title = "Overall Pitch Location Heatmap for Nathan Eovaldi", 
       x = "Plate X Position", 
       y = "Plate Z Position") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  )

# Plot for each pitch type and combine into one image
pitch_types <- unique(eovaldi_data$pitch_type)
plots <- list()

# Add the overall heatmap to the list of plots
plots[["Overall"]] <- overall_heatmap

for (pitch in pitch_types) {
  pitch_data <- eovaldi_data[eovaldi_data$pitch_type == pitch, ]
  
  p <- ggplot(pitch_data, aes(x = plate_x, y = plate_z)) +
    geom_bin2d(bins = 30, alpha = 0.7) + 
    scale_fill_gradient(low = "blue", high = "red") + 
    geom_rect(aes(xmin = -0.85, xmax = 0.85, ymin = 1.4, ymax = 3.6), 
              color = "black", size = 1.5, fill = NA) + 
    labs(title = paste("Pitch Location Heatmap for Nathan Eovaldi -", pitch), 
         x = "Plate X Position", 
         y = "Plate Z Position") +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    )
  
  plots[[pitch]] <- p
}

# Combine all heatmaps into one image (2 rows, 3 columns layout)
combined_heatmap <- wrap_plots(plots, ncol = 3, nrow = 2)  # 2 rows, 3 columns

# Save the combined heatmap as a PNG file
ggsave("combined_eovaldi_heatmaps_2x3.png", combined_heatmap, width = 15, height = 10, dpi = 300)
