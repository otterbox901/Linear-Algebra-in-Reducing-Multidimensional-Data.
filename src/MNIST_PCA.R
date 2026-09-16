# EE-PCA-vs-LDA
# Load required libraries
library(ggplot2)
library(dplyr)
library(keras)

# Load MNIST dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x

# Prepare the data
# Take first 1000 samples and flatten the images
set.seed(123)  # for reproducibility
sample_indices <- sample(1:nrow(x_train), 1000)
mnist_sample <- x_train[sample_indices, , ]

# Flatten the 28x28 images to 784-dimensional vectors
mnist_flat <- apply(mnist_sample, 1, function(x) as.vector(x))
mnist_flat <- t(mnist_flat)  # transpose to get samples as rows

cat("Original data dimensions:", dim(mnist_flat), "\n")

# Remove constant columns (columns with zero variance)
col_vars <- apply(mnist_flat, 2, var)
non_constant_cols <- which(col_vars > 1e-10)  # Keep columns with variance > small threshold
mnist_filtered <- mnist_flat[, non_constant_cols]

cat("After removing constant columns:", dim(mnist_filtered), "\n")
cat("Removed", ncol(mnist_flat) - ncol(mnist_filtered), "constant columns\n")

# Standardize the filtered data
mnist_scaled <- scale(mnist_filtered)

# Check for any remaining issues
if(any(is.na(mnist_scaled))) {
  cat("Warning: NA values found after scaling. Removing...\n")
  na_cols <- which(apply(is.na(mnist_scaled), 2, any))
  mnist_scaled <- mnist_scaled[, -na_cols]
  cat("Final dimensions after removing NA columns:", dim(mnist_scaled), "\n")
}

# Perform PCA
cat("Performing PCA on", nrow(mnist_scaled), "samples with", ncol(mnist_scaled), "features...\n")
pca_result <- prcomp(mnist_scaled, center = FALSE, scale.  = FALSE)

# Calculate variance explained by each component
variance_explained <- pca_result$sdev^2
total_variance <- sum(variance_explained)
prop_variance <- variance_explained / total_variance

# Calculate cumulative variance explained for up to 800 components
max_components <- min(800, length(prop_variance), ncol(mnist_scaled))
cumulative_variance <- cumsum(prop_variance[1:max_components])

# Find where 95% variance is captured
variance_95_index <- which(cumulative_variance >= 0.95)[1]
if(is.na(variance_95_index)) {
  variance_95_index <- length(cumulative_variance)
  cat("Warning: 95% variance not reached within available components\n")
}
variance_95_value <- cumulative_variance[variance_95_index]

cat(sprintf("95%% of variance is captured by the first %d components\n", variance_95_index))
cat(sprintf("Actual variance captured: %.3f%%\n", variance_95_value * 100))

# Create data frame for plotting
pca_data <- data.frame(
  Component = 1:max_components,
  Cumulative_Variance = cumulative_variance,
  Individual_Variance = prop_variance[1:max_components]
)

# Create the main plot
p1 <- ggplot(pca_data, aes(x = Component, y = Cumulative_Variance)) +
  geom_line(color = "blue", size = 1) +
  geom_hline(yintercept = 0.95, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = variance_95_index, color = "red", linetype = "dashed", size = 1) +
  geom_point(x = variance_95_index, y = variance_95_value, 
             color = "red", size = 3) +
  annotate("text", x = variance_95_index + 30, y = 0.92, 
           label = sprintf("95%% variance\n(%d components)", variance_95_index),
           color = "red", vjust = 0, hjust = 0, size = 3.5) +
  labs(
    title = "PCA Variance Explained - MNIST Dataset (1000 samples)",
    subtitle = sprintf("95%% of variance captured by first %d components out of %d total", 
                      variance_95_index, max_components),
    x = "Principal Component",
    y = "Cumulative Proportion of Variance Explained"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  scale_y_continuous(labels = scales::percent_format(), 
                     limits = c(0, 1)) +
  scale_x_continuous(breaks = seq(0, max_components, 50))

# Create a zoomed-in plot for the first part
zoom_limit <- min(200, variance_95_index + 50)
p2 <- ggplot(pca_data[1:zoom_limit, ], 
             aes(x = Component, y = Cumulative_Variance)) +
  geom_line(color = "blue", size = 1) +
  geom_hline(yintercept = 0.95, color = "red", linetype = "dashed", size = 1) +
  geom_vline(xintercept = variance_95_index, color = "red", linetype = "dashed", size = 1) +
  geom_point(x = variance_95_index, y = variance_95_value, 
             color = "red", size = 3) +
  annotate("text", x = variance_95_index + 10, y = variance_95_value - 0.05, 
           label = sprintf("%d components\nfor 95%% variance", variance_95_index),
           color = "red", vjust = 1, hjust = 0, size = 3.5) +
  labs(
    title = "PCA Variance Explained - Detailed View",
    x = "Principal Component",
    y = "Cumulative Proportion of Variance Explained"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_continuous(breaks = seq(0, zoom_limit, 25))

# Display plots
print(p1)
print(p2)

# Print summary statistics
cat("\n=== PCA Summary Statistics ===\n")
cat(sprintf("Total samples:  1000\n"))
cat(sprintf("Original features: 784 (28x28 pixels)\n"))
cat(sprintf("Features after removing constants: %d\n", ncol(mnist_scaled)))
cat(sprintf("Principal components calculated: %d\n", max_components))
cat(sprintf("Components for 95%% variance: %d (%.1f%% of total)\n", 
            variance_95_index, (variance_95_index/max_components)*100))

# Calculate other variance thresholds
variance_90_index <- which(cumulative_variance >= 0.90)[1]
variance_99_index <- which(cumulative_variance >= 0.99)[1]

if(!is.na(variance_90_index)) {
  cat(sprintf("Components for 90%% variance:  %d\n", variance_90_index))
}
if(!is. na(variance_99_index)) {
  cat(sprintf("Components for 99%% variance: %d\n", variance_99_index))
}

# Show variance explained by first 10 components
cat("\nVariance explained by first 10 components:\n")
first_10 <- data.frame(
  Component = 1:10,
  Individual_Pct = round(prop_variance[1:10] * 100, 2),
  Cumulative_Pct = round(cumulative_variance[1:10] * 100, 2)
)
print(first_10)

# Create a bar plot for individual variance of first 20 components
p3 <- ggplot(pca_data[1:20, ], aes(x = Component, y = Individual_Variance)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7, color = "darkblue") +
  labs(
    title = "Individual Variance Explained by First 20 Principal Components",
    x = "Principal Component",
    y = "Proportion of Variance Explained"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_continuous(breaks = 1:20) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)

# Create an elbow plot to show the "elbow" effect
p4 <- ggplot(pca_data[1:50, ], aes(x = Component, y = Individual_Variance)) +
  geom_line(color = "darkgreen", size = 1) +
  geom_point(color = "darkgreen", size = 2) +
  labs(
    title = "Scree Plot - Individual Variance by Component (First 50)",
    subtitle = "Shows the 'elbow' where variance drops off",
    x = "Principal Component",
    y = "Proportion of Variance Explained"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format())

print(p4)

# Save the results
write.csv(pca_data, "mnist_pca_results.csv", row. names = FALSE)
ggsave("mnist_pca_cumulative_variance.png", p1, width = 12, height = 8, dpi = 300)
ggsave("mnist_pca_zoomed.png", p2, width = 10, height = 6, dpi = 300)
ggsave("mnist_pca_individual_variance. png", p3, width = 10, height = 6, dpi = 300)
ggsave("mnist_pca_scree_plot.png", p4, width = 10, height = 6, dpi = 300)

# Show dimensionality reduction efficiency
cat(sprintf("\n=== Dimensionality Reduction Efficiency ===\n"))
cat(sprintf("Original dimensions: 784\n"))
cat(sprintf("Effective dimensions (non-constant): %d\n", ncol(mnist_scaled)))
cat(sprintf("Dimensions for 95%% variance: %d\n", variance_95_index))
cat(sprintf("Compression ratio: %.1fx\n", ncol(mnist_scaled) / variance_95_index))
cat(sprintf("Data retention: %.1f%%\n", variance_95_value * 100))

cat("\nAnalysis complete! All plots saved as PNG files and data saved as CSV.\n")
