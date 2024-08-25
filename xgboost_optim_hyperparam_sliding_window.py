#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:34:11 2024

@author: G7
# credit: Google Gemini, ChatGPT
"""

import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import optuna

# Generate random vectors
x = np.random.uniform(5, 10, 1000)
y = np.random.uniform(5, 10, 1000)
t = np.random.uniform(5, 10, 1000)
R = np.random.uniform(5, 10, 1000)

# Create a DataFrame with the input and output columns
data = pd.DataFrame({'x': x, 'y': y, 't': t, 'R': R})

# Function to create sliding window batches
def create_batches(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size + 1):
        X.append(data.iloc[i:i+window_size, :3].values.flatten())  # Flatten the window
        y.append(data.iloc[i+window_size-1, 3])
    return np.array(X), np.array(y)

# Create batches of 50 rows
window_size = 50
X_train, y_train = create_batches(data, window_size)

# Objective function for Optuna
def objective(trial):
    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    return rmse

# Optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the model with best hyperparameters
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_train)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_train, best_predictions))
r2 = r2_score(y_train, best_predictions)

print(f"Best RMSE: {rmse:.4f}")
print(f"Best R-squared: {r2:.4f}")

# Plotting y_train vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_train, best_predictions, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.xlabel('True Values (y_train)')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.grid(True)
plt.show()

# Plotting tornado chart (sensitivity analysis)
def plot_tornado_chart(study):
    import pandas as pd
    import seaborn as sns

    # Extracting hyperparameter importance from study
    trials_df = study.trials_dataframe()
    params = ['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma']
    
    # Calculate the mean and standard deviation of RMSE for different values of each hyperparameter
    importance_df = pd.DataFrame()
    for param in params:
        param_values = trials_df[param].unique()
        mean_rmse = []
        for value in param_values:
            subset = trials_df[trials_df[param] == value]
            mean_rmse.append(subset['value'].mean())
        importance_df[param] = mean_rmse
    
    importance_df.set_index(params, inplace=True)
    importance_df = importance_df.reset_index()
    importance_df = pd.melt(importance_df, id_vars=['index'], var_name='Hyperparameter', value_name='Mean RMSE')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='Mean RMSE', y='Hyperparameter', hue='index', dodge=True)
    plt.title('Tornado Chart - Sensitivity Analysis of Hyperparameters')
    plt.xlabel('Mean RMSE')
    plt.ylabel('Hyperparameter')
    plt.grid(True)
    plt.show()

plot_tornado_chart(study)
