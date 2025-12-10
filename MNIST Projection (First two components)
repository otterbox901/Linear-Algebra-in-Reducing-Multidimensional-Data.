# Load required libraries
library(ggplot2)
library(dplyr)
library(keras)
library(MASS)  # For LDA
library(tidyr)

# ============================================
# LOAD AND PREPARE MNIST DATA
# ============================================

# Load MNIST dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y

# Take a sample for computational efficiency
set.seed(123)
n_samples <- 2000  # Use 2000 samples
sample_indices <- sample(1:nrow(x_train), n_samples)
mnist_sample <- x_train[sample_indices, , ]
labels <- y_train[sample_indices]

# Flatten the 28x28 images to 784-dimensional vectors
mnist_flat <- apply(mnist_sample, 1, function(x) as.vector(x))
mnist_flat <- t(mnist_flat)

cat("Original data dimensions:", dim(mnist_flat), "\n")
cat("Number of classes:", length(unique(labels)), "\n")
cat("Class distribution:\n")
print(table(labels))

# Remove constant columns (columns with zero variance)
col_vars <- apply(mnist_flat, 2, var)
non_constant_cols <- which(col_vars > 1e-10)
mnist_filtered <- mnist_flat[, non_constant_cols]

cat("\nAfter removing constant columns:", dim(mnist_filtered), "\n")

# Standardize the data
mnist_scaled <- scale(mnist_filtered)

# Remove any NA columns after scaling
if(any(is.na(mnist_scaled))) {
  na_cols <- which(apply(is.na(mnist_scaled), 2, any))
  mnist_scaled <- mnist_scaled[, -na_cols]
}

cat("Final data dimensions:", dim(mnist_scaled), "\n")

# ============================================
# FUNCTION DEFINITIONS FOR METRICS
# ============================================

# Function to calculate class means
calculate_class_means <- function(data, labels) {
  classes <- unique(labels)
  class_means <- t(sapply(classes, function(c) {
    colMeans(data[labels == c, , drop = FALSE])
  }))
  rownames(class_means) <- classes
  return(class_means)
}

# Function to calculate Within-Class Scatter Matrix (Sw)
calculate_within_class_scatter <- function(data, labels) {
  classes <- unique(labels)
  n_features <- ncol(data)
  Sw <- matrix(0, n_features, n_features)
  
  for(c in classes) {
    class_data <- data[labels == c, , drop = FALSE]
    class_mean <- colMeans(class_data)
    centered <- sweep(class_data, 2, class_mean)
    Sw <- Sw + t(centered) %*% centered
  }
  return(Sw)
}

# Function to calculate Between-Class Scatter Matrix (Sb)
calculate_between_class_scatter <- function(data, labels) {
  classes <- unique(labels)
  n_features <- ncol(data)
  overall_mean <- colMeans(data)
  Sb <- matrix(0, n_features, n_features)
  
  for(c in classes) {
    class_data <- data[labels == c, , drop = FALSE]
    n_c <- nrow(class_data)
    class_mean <- colMeans(class_data)
    mean_diff <- class_mean - overall_mean
    Sb <- Sb + n_c * (mean_diff %*% t(mean_diff))
  }
  return(Sb)
}

# Function to calculate metrics from scatter matrices
calculate_metrics <- function(Sw, Sb) {
  # Within-class scatter (trace of Sw)
  within_scatter <- sum(diag(Sw))
  
  # Between-class distance (trace of Sb)
  between_distance <- sum(diag(Sb))
  
  # Separation ratio (trace(Sb) / trace(Sw))
  # Higher is better - means classes are more separated relative to their spread
  separation_ratio <- between_distance / within_scatter
  
  # Alternative:  J = trace(Sw^-1 * Sb) - Fisher criterion
  # Using pseudo-inverse for stability
  tryCatch({
    Sw_inv <- MASS::ginv(Sw)
    fisher_criterion <- sum(diag(Sw_inv %*% Sb))
  }, error = function(e) {
    fisher_criterion <- NA
  })
  
  return(list(
    within_scatter = within_scatter,
    between_distance = between_distance,
    separation_ratio = separation_ratio,
    fisher_criterion = fisher_criterion
  ))
}

# Function to calculate pairwise class distances
calculate_pairwise_distances <- function(class_means) {
  n_classes <- nrow(class_means)
  distances <- matrix(0, n_classes, n_classes)
  
  for(i in 1:n_classes) {
    for(j in 1:n_classes) {
      distances[i, j] <- sqrt(sum((class_means[i, ] - class_means[j, ])^2))
    }
  }
  rownames(distances) <- rownames(class_means)
  colnames(distances) <- rownames(class_means)
  return(distances)
}

# ============================================
# PERFORM PCA
# ============================================

cat("\n========== PERFORMING PCA ==========\n")

# Perform PCA
pca_result <- prcomp(mnist_scaled, center = FALSE, scale.  = FALSE)

# Choose number of components (e.g., 50 for comparison)
n_pca_components <- 50
pca_data <- pca_result$x[, 1:n_pca_components]

cat("PCA reduced dimensions:", dim(pca_data), "\n")

# Calculate metrics for PCA
cat("Calculating PCA metrics...\n")

pca_class_means <- calculate_class_means(pca_data, labels)
pca_Sw <- calculate_within_class_scatter(pca_data, labels)
pca_Sb <- calculate_between_class_scatter(pca_data, labels)
pca_metrics <- calculate_metrics(pca_Sw, pca_Sb)

cat("\n----- PCA METRICS (", n_pca_components, "components ) -----\n")
cat(sprintf("Within-Class Scatter (trace Sw): %.4f\n", pca_metrics$within_scatter))
cat(sprintf("Between-Class Distance (trace Sb): %.4f\n", pca_metrics$between_distance))
cat(sprintf("Separation Ratio (Sb/Sw): %.6f\n", pca_metrics$separation_ratio))
cat(sprintf("Fisher Criterion (trace Sw^-1 Sb): %.4f\n", pca_metrics$fisher_criterion))

# ============================================
# PERFORM LDA
# ============================================

cat("\n========== PERFORMING LDA ==========\n")

# LDA requires fewer features than samples per class
# First reduce with PCA, then apply LDA
n_pre_pca <- min(100, ncol(mnist_scaled) - 1)
pca_pre <- pca_result$x[, 1:n_pre_pca]

# Prepare data for LDA
lda_input <- as.data.frame(pca_pre)
lda_input$label <- as.factor(labels)

# Perform LDA (max components = n_classes - 1 = 9 for MNIST)
lda_result <- lda(label ~ ., data = lda_input)

# Transform data using LDA
lda_data <- as.matrix(pca_pre) %*% lda_result$scaling
n_lda_components <- ncol(lda_data)

cat("LDA reduced dimensions:", dim(lda_data), "\n")

# Calculate metrics for LDA
cat("Calculating LDA metrics...\n")

lda_class_means <- calculate_class_means(lda_data, labels)
lda_Sw <- calculate_within_class_scatter(lda_data, labels)
lda_Sb <- calculate_between_class_scatter(lda_data, labels)
lda_metrics <- calculate_metrics(lda_Sw, lda_Sb)

cat("\n----- LDA METRICS (", n_lda_components, "components ) -----\n")
cat(sprintf("Within-Class Scatter (trace Sw): %.4f\n", lda_metrics$within_scatter))
cat(sprintf("Between-Class Distance (trace Sb): %.4f\n", lda_metrics$between_distance))
cat(sprintf("Separation Ratio (Sb/Sw): %.6f\n", lda_metrics$separation_ratio))
cat(sprintf("Fisher Criterion (trace Sw^-1 Sb): %.4f\n", lda_metrics$fisher_criterion))

# ============================================
# COMPARISON ACROSS DIFFERENT DIMENSIONS
# ============================================

cat("\n========== METRICS ACROSS DIMENSIONS ==========\n")

# Calculate metrics for various PCA dimensions
pca_dims <- c(2, 5, 10, 20, 30, 50, 100, 150, 200)
pca_comparison <- data.frame()

for(d in pca_dims) {
  if(d <= ncol(pca_result$x)) {
    pca_d <- pca_result$x[, 1:d]
    Sw <- calculate_within_class_scatter(pca_d, labels)
    Sb <- calculate_between_class_scatter(pca_d, labels)
    metrics <- calculate_metrics(Sw, Sb)
    
    pca_comparison <- rbind(pca_comparison, data.frame(
      Method = "PCA",
      Dimensions = d,
      Within_Scatter = metrics$within_scatter,
      Between_Distance = metrics$between_distance,
      Separation_Ratio = metrics$separation_ratio,
      Fisher_Criterion = metrics$fisher_criterion
    ))
  }
}

# Add LDA results (fixed at 9 components for 10 classes)
lda_comparison <- data.frame(
  Method = "LDA",
  Dimensions = n_lda_components,
  Within_Scatter = lda_metrics$within_scatter,
  Between_Distance = lda_metrics$between_distance,
  Separation_Ratio = lda_metrics$separation_ratio,
  Fisher_Criterion = lda_metrics$fisher_criterion
)

# Combine results
all_comparison <- rbind(pca_comparison, lda_comparison)

cat("\nComparison Table:\n")
print(all_comparison, digits = 4)

# ============================================
# PAIRWISE CLASS DISTANCES
# ============================================

cat("\n========== PAIRWISE CLASS DISTANCES ==========\n")

# PCA pairwise distances (using first 2 components for visualization)
pca_2d <- pca_result$x[, 1:2]
pca_2d_means <- calculate_class_means(pca_2d, labels)
pca_pairwise <- calculate_pairwise_distances(pca_2d_means)

cat("\nPCA (2D) Pairwise Class Distances:\n")
print(round(pca_pairwise, 2))

# LDA pairwise distances (using first 2 components)
lda_2d <- lda_data[, 1:min(2, ncol(lda_data))]
lda_2d_means <- calculate_class_means(lda_2d, labels)
lda_pairwise <- calculate_pairwise_distances(lda_2d_means)

cat("\nLDA (2D) Pairwise Class Distances:\n")
print(round(lda_pairwise, 2))

# ============================================
# VISUALIZATIONS
# ============================================

cat("\n========== CREATING VISUALIZATIONS ==========\n")

# 1. Comparison bar plot
comparison_long <- all_comparison %>%
  pivot_longer(cols = c(Within_Scatter, Between_Distance, Separation_Ratio),
               names_to = "Metric", values_to = "Value")

# Separation Ratio plot
p1 <- ggplot(all_comparison, aes(x = factor(Dimensions), y = Separation_Ratio, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(
    title = "Separation Ratio Comparison:  PCA vs LDA",
    subtitle = "Higher separation ratio = better class discrimination",
    x = "Number of Dimensions",
    y = "Separation Ratio (Between/Within)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  scale_fill_manual(values = c("PCA" = "steelblue", "LDA" = "coral"))

print(p1)

# 2. 2D Scatter plot - PCA
pca_plot_data <- data.frame(
  PC1 = pca_2d[, 1],
  PC2 = pca_2d[, 2],
  Label = as.factor(labels)
)

p2 <- ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = Label)) +
  geom_point(alpha = 0.5, size = 1) +
  stat_ellipse(level = 0.95, size = 1) +
  labs(
    title = "PCA Projection of MNIST (First 2 Components)",
    subtitle = sprintf("Separation Ratio: %.4f", 
                      pca_comparison$Separation_Ratio[pca_comparison$Dimensions == 2]),
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Paired")

print(p2)

# 3. 2D Scatter plot - LDA
lda_plot_data <- data.frame(
  LD1 = lda_data[, 1],
  LD2 = lda_data[, 2],
  Label = as.factor(labels)
)

p3 <- ggplot(lda_plot_data, aes(x = LD1, y = LD2, color = Label)) +
  geom_point(alpha = 0.5, size = 1) +
  stat_ellipse(level = 0.95, size = 1) +
  labs(
    title = "LDA Projection of MNIST (First 2 Components)",
    subtitle = sprintf("Separation Ratio: %.4f", lda_metrics$separation_ratio),
    x = "Linear Discriminant 1",
    y = "Linear Discriminant 2"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Paired")

print(p3)

# 4. Metrics comparison across dimensions
p4 <- ggplot(pca_comparison, aes(x = Dimensions)) +
  geom_line(aes(y = Separation_Ratio, color = "Separation Ratio"), size = 1) +
  geom_point(aes(y = Separation_Ratio, color = "Separation Ratio"), size = 2) +
  geom_hline(yintercept = lda_metrics$separation_ratio, 
             color = "red", linetype = "dashed", size = 1) +
  annotate("text", x = max(pca_comparison$Dimensions) * 0.8, 
           y = lda_metrics$separation_ratio * 1.1,
           label = sprintf("LDA (9D): %.4f", lda_metrics$separation_ratio),
           color = "red") +
  labs(
    title = "PCA Separation Ratio vs Number of Components",
    subtitle = "Red dashed line shows LDA performance (9 components)",
    x = "Number of PCA Components",
    y = "Separation Ratio"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p4)

# 5. Heatmap of pairwise distances
library(reshape2)

# LDA pairwise distance heatmap
lda_pairwise_df <- melt(lda_pairwise)
colnames(lda_pairwise_df) <- c("Class1", "Class2", "Distance")

p5 <- ggplot(lda_pairwise_df, aes(x = Class1, y = Class2, fill = Distance)) +
  geom_tile() +
  geom_text(aes(label = round(Distance, 1)), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "LDA Pairwise Class Distances (2D Projection)",
    x = "Class",
    y = "Class"
  ) +
  theme_minimal() +
  coord_fixed()

print(p5)

# ============================================
# FINAL SUMMARY
# ============================================

cat("\n========================================\n")
cat("         FINAL SUMMARY COMPARISON        \n")
cat("========================================\n\n")

cat("METHOD        | DIMS | WITHIN SCATTER | BETWEEN DIST | SEP.  RATIO | FISHER CRIT.\n")
cat("--------------|------|----------------|--------------|------------|-------------\n")

# PCA at 9 dimensions (same as LDA)
pca_9d <- pca_result$x[, 1:9]
Sw_9 <- calculate_within_class_scatter(pca_9d, labels)
Sb_9 <- calculate_between_class_scatter(pca_9d, labels)
pca_9_metrics <- calculate_metrics(Sw_9, Sb_9)

cat(sprintf("PCA           |  9   | %14.2f | %12.2f | %10.4f | %11.2f\n",
            pca_9_metrics$within_scatter, pca_9_metrics$between_distance,
            pca_9_metrics$separation_ratio, pca_9_metrics$fisher_criterion))

cat(sprintf("LDA           |  9   | %14.2f | %12.2f | %10.4f | %11.2f\n",
            lda_metrics$within_scatter, lda_metrics$between_distance,
            lda_metrics$separation_ratio, lda_metrics$fisher_criterion))

cat("\n")
cat("KEY OBSERVATIONS:\n")
cat("-----------------\n")

if(lda_metrics$separation_ratio > pca_9_metrics$separation_ratio) {
  improvement <- ((lda_metrics$separation_ratio / pca_9_metrics$separation_ratio) - 1) * 100
  cat(sprintf("✓ LDA achieves %. 1f%% higher separation ratio than PCA (at same dimensions)\n", improvement))
} else {
  cat("✓ PCA achieves higher separation ratio than LDA at same dimensions\n")
}

cat("✓ LDA maximizes between-class variance while minimizing within-class variance\n")
cat("✓ PCA maximizes total variance without considering class labels\n")
cat("✓ For classification tasks, LDA typically provides better class separation\n")

# Save plots
ggsave("separation_ratio_comparison. png", p1, width = 10, height = 6, dpi = 300)
ggsave("pca_2d_projection.png", p2, width = 10, height = 8, dpi = 300)
ggsave("lda_2d_projection. png", p3, width = 10, height = 8, dpi = 300)
ggsave("pca_separation_vs_dims.png", p4, width = 10, height = 6, dpi = 300)
ggsave("lda_pairwise_distances. png", p5, width = 8, height = 8, dpi = 300)

# Save results to CSV
write.csv(all_comparison, "pca_lda_metrics_comparison.csv", row.names = FALSE)

cat("\nAll plots saved as PNG files and results saved to CSV.\n")
