#!/usr/bin/env python
# coding: utf-8

# # Principal Component Regression (PCR)

# Import required libraries
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Read data
path = '/Users/mouyasushi/Desktop/Machine Learning /skl_cs/crypto/Alpha-Research/ML/agg_data.parquet'
df = pq.read_table(path).to_pandas()
print(f"Dataset shape: {df.shape}")

# Data preprocessing (same as HW1.ipynb)
df = df.dropna(subset=['daily_return'])
df['open_time'] = pd.to_datetime(df['open_time'])

feature_cols = ['open', 'high', 'low', 'close', 'volume',
       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
       'taker_buy_quote_asset_volume',
       'volatility_7d', 'volatility_14d', 'volatility_30d', 'typical_price',
       'tp_vol', 'vwap_7d', 'vwap_14d', 'vwap_30d', 'vwap_7d_diff',
       'vwap_14d_diff', 'vwap_30d_diff', 'vwap_7d_trend',
       'vwap_7d_trend_direction', 'vwap_14d_trend', 'vwap_14d_trend_direction',
       'vwap_30d_trend', 'vwap_30d_trend_direction', 'rsi_7', 'rsi_14',
       'rsi_21', 'ema_7', 'ema_14', 'ema_30', 'ema_90', 'atr_7', 'atr_14',
       'atr_21', 'macd_line', 'macd_signal', 'macd_histogram', 'alpha1']

target_col = 'daily_return'

# Handle missing values
missing_values = df[feature_cols + [target_col]].isnull().sum()
if missing_values.sum() > 0:
    print("\nMissing values:")
    print(missing_values[missing_values > 0])
    df = df.dropna(subset=feature_cols + [target_col])

print(f"Dataset size after handling missing values: {len(df)}")

# Handle categorical variables
trend_mapping = {
    'Up': 1,
    'Down': 0,
}

# Map 'Up'/'Down' columns to numerical values
df['vwap_7d_trend_direction'] = df['vwap_7d_trend_direction'].map(trend_mapping)
df['vwap_14d_trend_direction'] = df['vwap_14d_trend_direction'].map(trend_mapping)
df['vwap_30d_trend_direction'] = df['vwap_30d_trend_direction'].map(trend_mapping)

# Split training and testing sets
cutoff_date = df['open_time'].sort_values().iloc[-int(len(df)/10)]  # Use the last 10% of data as test set
train_data = df[df['open_time'] <= cutoff_date]
test_data = df[df['open_time'] > cutoff_date]

print(f"\nTraining set shape: {train_data.shape}")
print(f"Testing set shape: {test_data.shape}")
print(f"Training set date range: {train_data['open_time'].min()} to {train_data['open_time'].max()}")
print(f"Testing set date range: {test_data['open_time'].min()} to {test_data['open_time'].max()}")

# Extract features and target variables
X_train = train_data[feature_cols].values
y_train = train_data[target_col].values
X_test = test_data[feature_cols].values
y_test = test_data[target_col].values

# Feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get number of features
p = X_train.shape[1]
print(f"Number of features: {p}")

# Store RMSE for each K value for training and testing sets
train_rmse = []
test_rmse = []

# Perform PCR for each K value from 1 to P
for k in range(1, p + 1):
    # 1. Use PCA for dimensionality reduction
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 2. Train linear regression model with reduced features
    pcr_model = LinearRegression()
    pcr_model.fit(X_train_pca, y_train)
    
    # 3. Predict on training set and calculate RMSE
    y_train_pred = pcr_model.predict(X_train_pca)
    rmse_train_k = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_rmse.append(rmse_train_k)
    
    # 4. Predict on testing set and calculate RMSE
    y_test_pred = pcr_model.predict(X_test_pca)
    rmse_test_k = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_rmse.append(rmse_test_k)
    
    print(f"K = {k}, Training RMSE = {rmse_train_k:.6f}, Testing RMSE = {rmse_test_k:.6f}")

# Find K values with minimum RMSE for training and testing sets
min_train_rmse_idx = np.argmin(train_rmse)
min_test_rmse_idx = np.argmin(test_rmse)

print(f"\nK value with minimum training RMSE: {min_train_rmse_idx + 1}, RMSE = {train_rmse[min_train_rmse_idx]:.6f}")
print(f"K value with minimum testing RMSE: {min_test_rmse_idx + 1}, RMSE = {test_rmse[min_test_rmse_idx]:.6f}")

# Visualize RMSE for different K values
plt.figure(figsize=(12, 6))
plt.plot(range(1, p + 1), train_rmse, 'b-', label='Training RMSE')
plt.plot(range(1, p + 1), test_rmse, 'r-', label='Testing RMSE')
plt.axvline(x=min_train_rmse_idx + 1, color='b', linestyle='--', label=f'Best Training K={min_train_rmse_idx + 1}')
plt.axvline(x=min_test_rmse_idx + 1, color='r', linestyle='--', label=f'Best Testing K={min_test_rmse_idx + 1}')
plt.xlabel('Number of Principal Components (K)')
plt.ylabel('RMSE')
plt.title('RMSE for PCR Models with Different K Values')
plt.legend()
plt.grid(True)
plt.savefig('PCR_all_components.png')
plt.show()

# Visualize RMSE for first 20 K values (if P is large, this provides a clearer view of the trend)
if p > 20:
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 21), train_rmse[:20], 'b-', label='Training RMSE')
    plt.plot(range(1, 21), test_rmse[:20], 'r-', label='Testing RMSE')
    if min_train_rmse_idx < 20:
        plt.axvline(x=min_train_rmse_idx + 1, color='b', linestyle='--', label=f'Best Training K={min_train_rmse_idx + 1}')
    if min_test_rmse_idx < 20:
        plt.axvline(x=min_test_rmse_idx + 1, color='r', linestyle='--', label=f'Best Testing K={min_test_rmse_idx + 1}')
    plt.xlabel('Number of Principal Components (K)')
    plt.ylabel('RMSE')
    plt.title('RMSE for PCR Models with First 20 K Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('PCR_first_20_components.png')
    plt.show()

# Calculate explained variance ratio
pca_full = PCA().fit(X_train_scaled)
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Visualize explained variance ratio
plt.figure(figsize=(12, 6))
plt.bar(range(1, p + 1), explained_variance_ratio, alpha=0.5, label='Individual Explained Variance Ratio')
plt.step(range(1, p + 1), cumulative_variance_ratio, where='mid', label='Cumulative Explained Variance Ratio')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Explained Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance Ratio')
plt.legend()
plt.grid(True)
plt.savefig('PCA_explained_variance.png')
plt.show()

# Find number of principal components needed to explain 90% and 95% variance
k_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
k_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of principal components needed to explain 90% variance: {k_90}")
print(f"Number of principal components needed to explain 95% variance: {k_95}")

# Save results to CSV file
results_df = pd.DataFrame({
    'K': range(1, p + 1),
    'Train_RMSE': train_rmse,
    'Test_RMSE': test_rmse,
    'Explained_Variance_Ratio': explained_variance_ratio,
    'Cumulative_Variance_Ratio': cumulative_variance_ratio
})
results_df.to_csv('PCR_results.csv', index=False)

print("Analysis complete. Results saved to CSV file and image files.") 