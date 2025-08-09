# Automobile-Price-Prediction

This repository includes the data, R code, and report for a machine learning project focused on predicting automobile prices using regression trees, Random Forests, and Gradient Boosting. The project provides insights into key factors influencing automobile prices. It serves as the final project for MGSC 661 - Multivariate Statistical Analysis (Data Analytics and AI for Business).


## Overview
This project aims to predict car prices using machine learning techniques, including regression trees, Random Forests, Gradient Boosting Machine (GBM), and Principal Component Analysis (PCA). By analyzing various car features, such as engine size, curb weight, fuel type, and more, the goal is to provide valuable insights for car manufacturers to optimize their pricing and design strategies.

## Data Description
The dataset includes numerical and categorical features such as:
- **Numerical Variables**: Engine size, curb weight, horsepower, price, and fuel efficiency metrics (city and highway mpg).
- **Categorical Variables**: Fuel type, make, body style, and engine location.

The data was cleaned and preprocessed, including:
- Imputation of missing values using mean (for numeric) and mode (for categorical).
- Handling of outliers by applying transformations for specific features.
- Correlation analysis to identify key relationships between features and price.

## Methodology
- **Principal Component Analysis (PCA)** was used to reduce dimensionality and address multicollinearity among numeric variables as it is a high-dimensional dataset with many features.
- **Modeling**: Tree-based models for non-linear relationship
  - **Regression Tree**: Quick benchmark and easy explanation of rules. The initial model explained 84.1% of price variation.
  - **Random Forest**: Ensemble of many decision trees, reduces variance, more robust to noise, handles mixed data types. Achieved 90.78% R-squared, significantly improving accuracy.
  - **Gradient Boosting Machine (GBM)**: Builds trees sequentially, each correcting previous errors; excellent predictive power; can handle complex patterns well. Achieved 87.75% R-squared, slightly outperforming the regression tree but lagging behind the Random Forest.
- **Validation Metrics**:
  - **Mean Absolute Error**: Stakeholders can easily grasp “average error in real units,” making it easy to communicate.
  - **R square**: measures how much variance in the target variable is explained by the model compared to a simple mean-based prediction.

## Results & Insights
- Key factors influencing car prices include **engine size**, **curb weight**, and **fuel efficiency**.
- Premium brands and vehicle size also play significant roles in determining price.
- The Random Forest model showed the best performance, explaining 90.78% of the variance in car prices.
- Car manufacturers should prioritize enhancing engine performance and vehicle dimensions to cater to higher price segments, while maintaining a balance in fuel efficiency for budget-conscious consumers.

