auto_data <- read.csv("D:/Scarlett/资料/Mcgill/Courses/Fall/Multi-stats/Final project/Dataset_5_Automobile data.csv")
attach(auto_data)
summary(auto_data)


# Replace '?' with NA in the dataset
auto_data[auto_data == "?"] <- NA
auto_data[auto_data == "l"] <- NA

str(auto_data)

unique(auto_data$horsepower)
unique(auto_data$peak.rpm)

unique(auto_data$price)

auto_data$horsepower <- as.numeric(auto_data$horsepower)
auto_data$peak.rpm <- as.numeric(auto_data$peak.rpm)
auto_data$price <- as.numeric(auto_data$price)
auto_data$bore <- as.numeric(auto_data$bore)
auto_data$stroke <- as.numeric(auto_data$stroke)

# Missing value 
missing_values <- sapply(auto_data, function(x) sum(is.na(x)))
print(missing_values)

# 1. Remove rows with missing values in the target variable 'price'
auto_data <- auto_data[!is.na(auto_data$price), ]

# 2. Drop the 'normalized.losses' variable
auto_data <- auto_data[, !names(auto_data) %in% c('normalized.losses')]

# 3. Impute missing values for numerical variables with the mean

auto_data[] <- lapply(auto_data, function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
  }
  return(x)
})


str(auto_data)
missing_values <- sapply(auto_data, function(x) sum(is.na(x)))
print(missing_values)

# 4. Impute missing values in categorical columns with the mode
auto_data[] <- lapply(auto_data, function(x) {
  if (is.character(x)) {
    if (any(is.na(x))) {
      # Replace NA with mode
      x[is.na(x)] <- names(sort(table(x), decreasing = TRUE))[1]
    }
  }
  return(x)
})

# Verify if there are any missing values left
missing_values <- sapply(auto_data, function(x) sum(is.na(x)))
print(missing_values)


#Outlier
# Visualize numeric columns using boxplots

boxplot(auto_data[, numeric_cols], 
        main = "Boxplots of Numeric Columns", 
        las = 2, 
        col = "lightblue")


detect_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  return(which(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)))
}

numeric_cols <- sapply(auto_data, is.numeric)
outlier_indices <- lapply(auto_data[, numeric_cols], detect_outliers)
print(outlier_indices)

# Apply logarithmic transformation
auto_data$compression.ratio <- log(auto_data$compression.ratio + 1)
auto_data$stroke <- log(auto_data$stroke + 1)

# Verify the changes
summary(auto_data[, c("compression.ratio", "stroke")])

# EDA
numeric_cols <- sapply(auto_data, is.numeric)
# Subset numeric columns
numeric_data <- auto_data[, numeric_cols]

# Set layout for multiple plots
par(mfrow = c(2, 3))
dev.off()
# Plot histograms for each numeric column
for (col in names(numeric_data)) {
  hist(numeric_data[[col]], 
       main = paste("Histogram of", col), 
       xlab = col, 
       col = "skyblue", 
       border = "white")
}

library(reshape2)
library(ggplot2)
# Compute the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

print(correlation_matrix)
# Melt the correlation matrix into long format
correlation_data <- melt(correlation_matrix)

# Plot the heatmap
ggplot(correlation_data, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(x = "", y = "")


#Categorical plots
# Box plot for fuel.type vs price
ggplot(auto_data, aes(x = fuel.type, y = price, fill = fuel.type)) +
  geom_boxplot() +
  labs(title = "Fuel Type vs Price", x = "Fuel Type", y = "Price") +
  theme_minimal()

# Box plot for make vs price 
ggplot(auto_data, aes(x = make, y = price, fill = make)) +
  geom_boxplot() +
  labs(title = "Make vs Price", x = "Make", y = "Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) # Rotate x-axis labels for better visibility

library(dplyr)

# List of categorical variables
categorical_vars <- c("fuel.type", "aspiration", "body.style", "drive.wheels", "num.of.doors")

# Frequency Distribution
for (var in categorical_vars) {
  cat("\n\nFrequency Distribution for", var, ":\n")
  print(table(auto_data[[var]]))
  cat("\n")
}


library(tidyr)

# 1. Bar plot for fuel.type
ggplot(auto_data, aes(x = fuel.type)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Distribution of Fuel Type", x = "Fuel Type", y = "Count") +
  theme_minimal()

# 3. Grouped bar plot for body.style split by fuel.type
ggplot(auto_data, aes(x = body.style, fill = fuel.type)) +
  geom_bar(position = "dodge") +
  labs(title = "Body Style Split by Fuel Type", x = "Body Style", y = "Count") +
  theme_minimal()

# Box plot for engine.location and price
ggplot(auto_data, aes(x = engine.location, y = price, fill = engine.location)) +
  geom_boxplot() +
  labs(title = "Engine Location vs Price", x = "Engine Location", y = "Price") +
  theme_minimal() +
  scale_fill_manual(values = c("skyblue", "lightgreen"))


#transform categorical data to numer
auto_data_encoded <- auto_data
# Apply one-hot encoding for categorical variables
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ make - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ fuel.type - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ aspiration - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ num.of.doors - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ body.style - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ drive.wheels - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ engine.location - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ engine.type - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ num.of.cylinders - 1, data = auto_data))
auto_data_encoded <- cbind(auto_data_encoded, model.matrix(~ fuel.system - 1, data = auto_data))

# Remove original categorical columns that are now encoded
auto_data_encoded <- auto_data_encoded[, !(names(auto_data) %in% c("make", "fuel.type", "aspiration", 
                                                   "num.of.doors", "body.style", 
                                                   "drive.wheels", "engine.location", 
                                                   "engine.type", "num.of.cylinders", "fuel.system"))]



###################################################################
#PCA
# Standardize the data (center and scale)
standardized_data <- scale(numeric_data)

# Step 2: Perform PCA
pca_result <- prcomp(standardized_data, center = TRUE, scale. = TRUE)

# Step 3: Check the PCA summary
pca_result

# Extract PCA summary components
pca_summary <- summary(pca_result)


# Step 4: Visualize the explained variance (Scree plot)
# A scree plot shows the proportion of variance explained by each principal component

# Extract the proportion of variance from PCA
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Plot the scree plot
plot(explained_variance, type = "b", pch = 16, col = "blue", 
     main = "Scree Plot", xlab = "Principal Components", 
     ylab = "Proportion of Variance Explained")

# Add labels to each point
text(x = 1:length(explained_variance), 
     y = explained_variance, 
     labels = paste("PC", 1:length(explained_variance), sep=""), 
     pos = 3, cex = 0.8, col = "black")
install.packages("ggfortify")
library(ggfortify)

autoplot(pca_result, data = standardized_data, loadings = TRUE, loadings.label = TRUE )

# Add a price range category to auto_data_encoded
auto_data_encoded$price_range <- cut(
  auto_data_encoded$price,
  breaks = c(0, 10000, 25000, 50000),
  labels = c("0-10000", "10001-25000", "25001-50000"),
  include.lowest = TRUE
)

# Plot the PCA biplot
# Add a price range category to auto_data_encoded
auto_data_encoded$price_range <- cut(
  auto_data_encoded$price,
  breaks = c(0, 10000, 25000, 50000),
  labels = c("0-10000", "10001-25000", "25001-50000"),
  include.lowest = TRUE
)

# Load necessary library
library(ggfortify)

# Plot the PCA biplot
autoplot(pca_result, data = auto_data_encoded, 
         colour = 'price_range',  # Color by price range
         loadings = TRUE, 
         loadings.label = TRUE) +
  scale_colour_manual(values = c("blue", "green", "red")) +  # Custom colors
  ggtitle("PCA Biplot: Car Characteristics by Price Range") +
  theme_minimal()



###Tree model
install.packages("rpart")
install.packages("rpart.plot")
# Load the necessary libraries
library(rpart)
library(rpart.plot)

set.seed(123)  # For reproducibility

# Assuming your dataset is named `auto_data_encoded`
# Set the proportion for training data (e.g., 70% training, 30% test)
train_index <- sample(1:nrow(auto_data_encoded), size = 0.7 * nrow(auto_data_encoded))

# Split the data into training and test sets
train_data <- auto_data_encoded[train_index, ]
test_data <- auto_data_encoded[-train_index, ]

# Build the regression tree model
tree_model <- rpart(price ~ ., data = train_data, method = "anova")

# Visualize the tree
rpart.plot(tree_model, main = "Regression Tree for Car Prices", cex = 0.8)

# Print variable importance
print(tree_model$variable.importance)

# Print the complexity parameter table
printcp(tree_model)

# Prune the tree using the optimal cp
optimal_tree <- prune(tree_model, cp = 0.010770)

# Plot the pruned tree
rpart.plot(
  optimal_tree,                   # Your rpart model object
  type = 4,                       # Uniform display of splits
  digits = 0,                     # Round to integers
  fallen.leaves = TRUE,           # Align terminal nodes
  box.palette = "auto",           # Automatic node color
  main = "Prune Regression Tree for Car Prices", # Title of the tree
  tweak = 1,                    # Adjust text size
  extra = 101,                    # Add percentages and node predictions
  roundint = TRUE                 # Ensure integers are displayed
)

# Make predictions on the test data
pred_test <- predict(optimal_tree, newdata = test_data)

# Evaluate the model (e.g., MSE or R-squared)
mse <- mean((pred_test - test_data$price)^2)
rss <- sum((pred_test - test_data$price)^2)
tss <- sum((test_data$price - mean(test_data$price))^2)
r_squared <- 1 - (rss / tss)

# Print performance metrics
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")

##Using Final data
# Set the proportion for training data (e.g., 70% training, 30% test)
train_index <- sample(1:nrow(final_data), size = 0.7 * nrow(final_data))

# Split the data into training and test sets
train_data <- final_data[train_index, ]
test_data <- final_data[-train_index, ]


# Build the regression tree model
tree_model <- rpart(target ~ ., data = train_data, method = "anova")

# Visualize the tree
rpart.plot(tree_model, main = "Regression Tree for Car Prices", cex = 0.8)

# Print variable importance
print(tree_model$variable.importance)

# Print the complexity parameter table
printcp(tree_model)

# Prune the tree using the optimal cp
optimal_tree <- prune(tree_model, cp = 0.010000)

# Plot the pruned tree
rpart.plot(
  optimal_tree,                   # Your rpart model object
  type = 4,                       # Uniform display of splits
  digits = 0,                     # Round to integers
  fallen.leaves = TRUE,           # Align terminal nodes
  box.palette = "auto",           # Automatic node color
  main = "Prune Regression Tree for Car Prices", # Title of the tree
  tweak = 1,                      # Adjust text size
  extra = 101,                    # Add percentages and node predictions
  roundint = TRUE                 # Ensure integers are displayed
)

# Make predictions on the test data
pred_test <- predict(optimal_tree, newdata = test_data)

# Evaluate the model (e.g., MSE or R-squared)
mse <- mean((pred_test - test_data$target)^2)
rss <- sum((pred_test - test_data$target)^2)
tss <- sum((test_data$price - mean(test_data$target))^2)
r_squared <- 1 - (rss / tss)

# Print performance metrics
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")

### RF
# Install and load the required packages
install.packages("randomForest")
library(randomForest)
library(caret)

colnames(auto_data_encoded) <- make.names(colnames(auto_data_encoded))

# Split into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(auto_data_encoded$price, p = 0.7, list = FALSE)
trainData <- auto_data_encoded[trainIndex, ]
testData <- auto_data_encoded[-trainIndex, ]


# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(
  price ~ ., 
  data = trainData, 
  importance = TRUE, 
  ntree = 500
)

# View model summary
print(rf_model)

# Predict on test data
predictions <- predict(rf_model, newdata = testData)

# Calculate RMSE
rmse <- sqrt(mean((testData$price - predictions)^2))
cat("RMSE:", rmse, "\n")

# Calculate R-squared
rsq <- 1 - sum((testData$price - predictions)^2) / sum((testData$price - mean(testData$price))^2)
cat("R-squared:", rsq, "\n")

# Plot variable importance
importance <- importance(rf_model)
varImpPlot(rf_model)

##GBM
# Install and load the gbm package if not already installed
install.packages("gbm")
library(gbm)

# Fit the Gradient Boosting Model
gbm_model <- gbm(
  formula = price ~ .,           # Predicting 'price' using all other variables
  data = trainData,              # Training data
  distribution = "gaussian",     # Regression problem
  n.trees = 1000,                 # Number of boosting iterations (trees)
  interaction.depth = 6,         # Maximum depth of trees
  shrinkage = 0.01,              # Learning rate
  cv.folds = 5,                  # Cross-validation folds
  n.cores = 2                    # Number of cores for parallel processing
)

# Summary of the GBM model
summary(gbm_model)

# Predict on the test data
predictions_gbm <- predict(gbm_model, newdata = testData, n.trees = gbm_model$n.trees)

# Calculate RMSE
rmse_gbm <- sqrt(mean((testData$price - predictions_gbm)^2))
cat("RMSE for GBM:", rmse_gbm, "\n")

# Calculate R-squared
rsq_gbm <- 1 - sum((testData$price - predictions_gbm)^2) / sum((testData$price - mean(testData$price))^2)
cat("R-squared for GBM:", rsq_gbm, "\n")

# Plot the learning curve (performance over iterations)
gbm.perf(gbm_model, method = "cv")

# Get the summary of the GBM model
summary_gbm <- summary(gbm_model)

# Extract the relative importance and feature names
relative_influence <- summary_gbm$rel.inf
feature_names <- summary_gbm$var

# Sort the relative importance in descending order
sorted_indices <- order(relative_influence, decreasing = TRUE)
relative_influence_sorted <- relative_influence[sorted_indices]
feature_names_sorted <- feature_names[sorted_indices]

# Select the top 17 features
top_n <- 17
relative_influence_top17 <- relative_influence_sorted[1:top_n]
feature_names_top17 <- feature_names_sorted[1:top_n]

par(mar = c(5, 10, 4, 2))  # Increase left margin (2nd value) for better label space

# Create the barplot for the top 17 features
barplot(relative_influence_top17, 
        names.arg = feature_names_top17,  # Feature names on the Y-axis
        col = "blue", 
        las = 2,  # Rotate axis labels for readability (90 degrees)
        cex.names = 0.7,  # Adjust size of feature names (make them smaller)
        main = "Top Relative Influence of Features in GBM Model", 
        xlab = "Relative Influence", 
        horiz = TRUE,  # Horizontal bars
        xlim = c(0, max(relative_influence_top17) + 5),  # Adjust xlim for spacing
        cex.main = 1.2,  # Adjust title size for better readability
        cex.lab = 1)  # Adjust axis label size

# Add the relative importance value as text on the bars
text(x = relative_influence_top17 + 1,  # Position text slightly to the right of the bars
     y = seq_along(relative_influence_top17),  # Place text next to each feature
     labels = round(relative_influence_top17, 2),  # Display relative importance rounded to 2 decimal places
     pos = 4,  # Place text to the right of the bar
     cex = 0.7)  # Adjust text size for clarity

#####################################################################


# Load necessary libraries
library(cluster)     # For clustering
library(ggplot2)     # For visualization
library(factoextra)  # For advanced visualizations

# Split the data (assuming the data is already loaded as 'auto_data_encoded')
set.seed(123)
trainIndex <- sample(1:nrow(auto_data_encoded), 0.7 * nrow(auto_data_encoded))
trainData <- auto_data_encoded[trainIndex, ]
testData <- auto_data_encoded[-trainIndex, ]

# Check for columns with all 0s or all 1s in the training set
binary_check_train <- apply(trainData, 2, function(x) all(x == 0) | all(x == 1))
cols_to_remove_train <- names(binary_check_train[binary_check_train])

# Check for columns with all 0s or all 1s in the test set
binary_check_test <- apply(testData, 2, function(x) all(x == 0) | all(x == 1))
cols_to_remove_test <- names(binary_check_test[binary_check_test])

# Remove those columns from both train and test datasets
trainData <- trainData[, !colnames(trainData) %in% c(cols_to_remove_train, cols_to_remove_test)]
testData <- testData[, !colnames(testData) %in% c(cols_to_remove_train, cols_to_remove_test)]

# Scale the data (excluding the target variable 'price' if applicable)
scaled_features_train <- scale(trainData[, -which(names(trainData) == "price")])
scaled_features_test <- scale(testData[, -which(names(testData) == "price")])

#Find the optimal number of clustering
library(cluster)
sil_width <- numeric(10)  # Store silhouette width for each k
sil_results <- list()     # Store detailed silhouette results

# Run K-means clustering for different values of k
for (k in 2:10) {
  # Perform K-means clustering
  kmeans_result <- kmeans(scaled_features_train, centers = k, nstart = 25)
  
  # Calculate silhouette width
  sil <- silhouette(kmeans_result$cluster, dist(scaled_features_train))
  sil_width[k] <- mean(sil[, 3])  # Average silhouette width
  
  # Store the detailed silhouette results
  sil_results[[k]] <- sil
}

# Print silhouette analysis for each k
for (k in 2:10) {
  cat("\nSilhouette Analysis for k =", k, "clusters:\n")
  print(sil_results[[k]])
}

# Plot the silhouette scores for each k
library(ggplot2)
silhouette_data <- data.frame(k = 2:10, sil_width = sil_width[2:10])
ggplot(silhouette_data, aes(x = k, y = sil_width)) +
  geom_line() + 
  geom_point() +
  labs(title = "Silhouette Analysis", x = "Number of Clusters (k)", y = "Average Silhouette Width")

# Set the number of clusters to 5 based on the silhouette analysis
k_optimal <- 3

# Perform K-means clustering
kmeans_result <- kmeans(scaled_features_train, centers = k_optimal, nstart = 25)

# Add the cluster assignments to the train data
trainData$cluster <- as.factor(kmeans_result$cluster)

# Visualize the clustering result using PCA projection
library(ggplot2)
pca_result <- prcomp(scaled_features_train)
pca_data <- data.frame(PC1 = pca_result$x[,1], PC2 = pca_result$x[,2], Cluster = trainData$cluster)

ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  labs(title = "K-means Clustering (PCA Projection)", x = "PC1", y = "PC2") +
  theme_minimal()

# Analyze the average price by cluster
average_price_by_cluster <- aggregate(price ~ cluster, data = trainData, FUN = mean)
print(average_price_by_cluster)

# Visualize the price distribution by cluster using boxplot
ggplot(trainData, aes(x = cluster, y = price, fill = cluster)) +
  geom_boxplot() +
  labs(title = "Price Distribution by K-means Cluster", x = "Cluster", y = "Price") +
  theme_minimal()

