#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:34:11 2024

@author: G7
"""

import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
window_size = 20
X_train, y_train = create_batches(data, window_size)

# Check shapes for debugging
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Train the XGBoost model
model = xgb.XGBRegressor(
    booster='gbtree',
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    objective='reg:squarederror',
    eval_metric='rmse'
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_train, predictions))
r2 = r2_score(y_train, predictions)

print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plotting y_train vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_train, predictions, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.xlabel('True Values (y_train)')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.grid(True)
plt.show()
