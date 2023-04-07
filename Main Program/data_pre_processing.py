#
# Data Pre-Processing:  This code checks for any outliers, applies a moving average filter to reduce noise in the data,
#                       normalizes the data to make it suitable for logistic regression.
#

# Import Statements
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# Check for any NaNs or Nulls
def check_nan_null_values(data, data_name):
    print(f"\nData file: {data_name}")

    nan_count = data.isna().sum()
    print("\nThe total number of NaNs are:\n", nan_count)

    # Check for null values
    null_count = data.isnull().sum()
    print("\nThe total number of Nulls are:\n", null_count)

    return nan_count, null_count


# Handle missing values if there are any
def handle_missing_values(data, method='drop'):
    if method == 'drop':
        cleaned_data = data.dropna()
    elif method == 'fillna':
        cleaned_data = data.fillna(data.mean())

    return cleaned_data


# Applying the moving average filter
def moving_average_filter(df):
    window_sizes = [5, 50, 100]
    filtered_data = {}

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    acceleration_axes = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

    for i, col in enumerate(acceleration_axes):
        for window_size in window_sizes:
            filtered_df = df.copy()
            filtered_df[col] = filtered_df[col].rolling(window=window_size).mean()
            filtered_data[f'window_size_{window_size}'] = filtered_df.dropna()

            axes[i].plot(filtered_df['Time (s)'], filtered_df[col], label=f"Window size {window_size}")

        # Plotting the original data
        axes[i].plot(df['Time (s)'], df[col], label="Original data", alpha=0.3)
        axes[i].set_title(col)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Acceleration (m/s^2)")
        axes[i].legend()

    plt.tight_layout()
    #plt.show()

    return filtered_data


# Removing any potential outliers in the data
def remove_outliers(data, threshold=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data_filtered = data[~((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)]

    return data_filtered


# Normalizing the data
def normalize_data(data):
    scaler = StandardScaler()
    columns_to_normalize = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
    scaler.fit(data[columns_to_normalize])
    data_normalized = data.copy()
    data_normalized[columns_to_normalize] = scaler.transform(data[columns_to_normalize])

    # Plotting the normalized data
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    acceleration_axes = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

    for i, col in enumerate(acceleration_axes):
        axes[i].plot(data_normalized['Time (s)'], data_normalized[col], label=f"Normalized {col}")
        axes[i].set_title(f"Normalized {col}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Acceleration (m/s^2)")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    return data_normalized
