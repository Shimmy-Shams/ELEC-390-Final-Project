#
# Data Pre-Processing:  This code checks for any outliers, applies a moving average filter to reduce noise in the data,
#                       normalizes the data to make it suitable for logistic regression.
#

# Import Statements
from sklearn import preprocessing
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
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

    acceleration_axes = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

    for i, col in enumerate(acceleration_axes):
        for window_size in window_sizes:
            filtered_df = df.copy()
            filtered_df[col] = filtered_df[col].rolling(window=window_size).mean()
            filtered_data[f'window_size_{window_size}'] = filtered_df.dropna()

    return filtered_data


# Plotting the filtered data
def plot_filtered_data(df, filtered_data):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    acceleration_axes = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

    for i, col in enumerate(acceleration_axes):
        for window_size, filtered_df in filtered_data.items():
            axes[i].scatter(filtered_df['Time (s)'], filtered_df[col], label=f"{window_size}", s=10)

        # Plotting the original data
        axes[i].scatter(df['Time (s)'], df[col], label="Original data", alpha=0.3, s=10)
        axes[i].set_title(col)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Acceleration (m/s^2)")
        axes[i].legend()

    #plt.tight_layout()
    #plt.show()


# Applying an exponential moving average to further reduce noise
def exponential_moving_average_filter(df, alpha=0.1):
    filtered_df = df.copy()
    acceleration_axes = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

    for col in acceleration_axes:
        filtered_df[col] = df[col].ewm(alpha=alpha).mean()

    return filtered_df


# Removing any potential outliers in the data
def remove_outliers(data, threshold=2):
    Q1 = data.quantile(0.25, numeric_only=True)
    Q3 = data.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1

    left, right = data.align(Q1 - threshold * IQR, axis=1, copy=False)
    data_filtered = data[~((left < right) | (data > (Q3 + threshold * IQR))).any(axis=1)]

    return data_filtered


# Handling imbalance of data
def handle_imbalance(X, y, k_neighbors=4):
    smote = SMOTE(sampling_strategy='minority', k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# Normalizing the data
def normalize_data(data):
    scaler = StandardScaler()
    columns_to_normalize = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
    scaler.fit(data[columns_to_normalize])
    data_normalized = data.copy()
    data_normalized[columns_to_normalize] = scaler.transform(data[columns_to_normalize])

    return data_normalized


def plot_normalized_data_histograms(data_normalized):
    # Plotting the normalized data
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    acceleration_axes = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']

    for i, col in enumerate(acceleration_axes):
        axes[i].hist(data_normalized[col], bins=50, alpha=0.75, label=f"Normalized {col}")
        axes[i].set_title(f"Normalized {col}")
        axes[i].set_xlabel("Acceleration (m/s^2)")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    #plt.tight_layout()
   # plt.show()

