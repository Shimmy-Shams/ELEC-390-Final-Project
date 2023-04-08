#
# Data Extracting Features: This code extracts a minimum of 10 features.

# Importing all the necessary modules
import numpy as np
import pandas as pd
import scipy.stats as stats


# Code to convert the data files into train and test files
def convert_to_segments(data, window_size, sample_rate):
    num_samples = window_size * sample_rate
    segments = []

    for i in range(0, len(data) - num_samples + 1, num_samples):
        segment = data.iloc[i:i + num_samples]
        segments.append(segment)

    return segments


# Code to extract the features
def extract_features(segments):
    feature_names = ['max', 'min', 'range', 'mean', 'median', 'variance', 'std',
                     'skewness', 'kurtosis', 'rms', 'mean_abs_change', 'mean_crossing_rate']
    features = []

    for segment in segments:
        segment_features = []
        for axis in ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']:
            data = segment[axis]

            max_value = np.max(data)
            min_value = np.min(data)
            range_value = max_value - min_value
            mean_value = np.mean(data)
            median_value = np.median(data)
            variance_value = np.var(data)
            std_value = np.std(data)
            skewness_value = stats.skew(data)
            kurtosis_value = stats.kurtosis(data)
            rms_value = np.sqrt(np.mean(np.square(data)))
            mean_abs_change = np.mean(np.abs(np.diff(data)))
            mean_crossing_rate = np.mean(np.diff(data > np.mean(data)))

            segment_features.extend([max_value, min_value, range_value, mean_value, median_value,
                                     variance_value, std_value, skewness_value, kurtosis_value,
                                     rms_value, mean_abs_change, mean_crossing_rate])

        features.append(segment_features)

    feature_df = pd.DataFrame(features,
                              columns=[f'{axis}_{feature}' for axis in ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z'] for feature in feature_names])

    return feature_df


# Saving the features into a csv file.
def save_features_to_csv(data_features, data_filename):
    """
    Save the train and test DataFrames containing extracted features to CSV files.

    Args:
        data_features (pd.DataFrame): DataFrame containing features from the training dataset
        data_filename (str): File name for the CSV file of the training dataset features
    """

    data_features.to_csv(data_filename, index=False)
