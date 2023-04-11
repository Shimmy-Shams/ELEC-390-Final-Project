# Final Project
# Group: Shams, Harshil and Jasmine
# Course: ELEC 390

# Main.py
# JACKPOT!
# Code that works by normalizing the data after feature extraction

# Importing all the necessary libraries
import data_extracting_features as de
import data_pre_processing as dpp
import data_visualization as dv
import data_processing as dp
import data_classifier as dc
import pandas as pd
import joblib


# --------------------------------------------------------------------------------
# -------------------------- Step 1: Data Collection -----------------------------

# Importing the csv files from the directory
csv_file_paths = [
    'Data/Raw_Data_Harshil.csv',
    'Data/Raw_Data_Shams.csv',
    'Data/Raw_Data_Jasmine.csv'
]

# Importing the meta data
meta_data_time = [
    'Data/meta_Harshil/time.csv',
    'Data/meta_Jasmine/time.csv',
    'Data/meta_Shams/time.csv'
]


# --------------------------------------------------------------------------------
# --------------- Step 2: Data Storage and Processing ----------------------------

window_size = 5  # 5 seconds
sample_rate = 100  # Assuming 100 samples per second, adjust if necessary

# Load data
data = dp.load_data(csv_file_paths)

# Segment data into 5-second windows
segments = dp.segment_data(data, window_size, sample_rate)

# Shuffle and create train and test splits
train_data, test_data = dp.create_splits(segments)
# Saving the Test data to test the classifier in the app
normalized_test_data_filename = "Data/Test_data.csv"
de.save_features_to_csv(test_data, normalized_test_data_filename)

# Save data to HDF5 file
dp.write_to_hdf5(data, csv_file_paths, train_data, test_data)
original_data, train_data, test_data = dp.reading_from_hdf5()
dp.display_data(original_data, train_data, test_data)


# --------------------------------------------------------------------------------
# -------------------- Step 3: Data Visualization --------------------------------

# Extract the original data from the dp file and visualize it with the dv file
for dataset_name in original_data.keys():
    print(f"Visualizing data for {dataset_name}")
    time, accelerationX, accelerationY, accelerationZ = dp.extract_time_and_acceleration(original_data, dataset_name)
    dv.plot_general_acceleration_vs_time(time, accelerationX, accelerationY, accelerationZ)


# Visualization of the meta data
df_md_time_list = [dp.process_meta_data_time(time_file) for time_file in meta_data_time]
dv.plot_md_time_gantt(df_md_time_list)
dv.plot_md_time_bar(df_md_time_list)


train_data_visualization = dv.Activity(train_data, data_source="Train")
test_data_visualization = dv.Activity(test_data, data_source="Test")

# Scatter plots
train_data_visualization.plot_scatter("Walking")
train_data_visualization.plot_scatter("Jumping")
test_data_visualization.plot_scatter("Walking")
test_data_visualization.plot_scatter("Jumping")

# Line plots
train_data_visualization.plot_line("Walking")
train_data_visualization.plot_line("Jumping")
test_data_visualization.plot_line("Walking")
test_data_visualization.plot_line("Jumping")

# Histograms
train_data_visualization.plot_histogram("Walking")
train_data_visualization.plot_histogram("Jumping")
test_data_visualization.plot_histogram("Walking")
test_data_visualization.plot_histogram("Jumping")

# Bar Plot of mean
train_data_visualization.plot_mean_bar("Walking")
train_data_visualization.plot_mean_bar("Jumping")
test_data_visualization.plot_mean_bar("Walking")
test_data_visualization.plot_mean_bar("Jumping")

# Box Plot
train_data_visualization.plot_box("Walking")
train_data_visualization.plot_box("Jumping")
test_data_visualization.plot_box("Walking")
test_data_visualization.plot_box("Jumping")

# Spectrogram
train_data_visualization.plot_spectrogram("Walking")
train_data_visualization.plot_spectrogram("Jumping")
test_data_visualization.plot_spectrogram("Walking")
test_data_visualization.plot_spectrogram("Jumping")


# --------------------------------------------------------------------------------
# -------------------- Step 4: Data Preprocessing --------------------------------
# Checking for any missing values
train_data_file_name = "Training Data"
test_data_file_name = "Test Data"

train_data_chk_missing_values = dpp.check_nan_null_values(train_data, train_data_file_name)
test_data_chk_missing_values = dpp.check_nan_null_values(test_data, test_data_file_name)


train_data_handle_missing_values = dpp.handle_missing_values(train_data, method='drop')
test_data_handle_missing_values = dpp.handle_missing_values(test_data, method='drop')


# Applying moving average
train_data_filtered = dpp.moving_average_filter(train_data)
test_data_filtered = dpp.moving_average_filter(test_data)
# Plot the moving average filter
dpp.plot_filtered_data(train_data, train_data_filtered)
dpp.plot_filtered_data(test_data, test_data_filtered)

# Select the filtered data with window size 50
selected_window_size = 100
train_data_filtered_100 = train_data_filtered[f'window_size_{selected_window_size}']
test_data_filtered_100 = test_data_filtered[f'window_size_{selected_window_size}']

# Applying exponential moving average
train_data_ema_filtered = dpp.exponential_moving_average_filter(train_data_filtered_100, alpha=0.15)
test_data_ema_filtered = dpp.exponential_moving_average_filter(test_data_filtered_100, alpha=0.15)

# Removing outliers
train_data_no_outliers = dpp.remove_outliers(train_data_ema_filtered, threshold=2)
test_data_no_outliers = dpp.remove_outliers(test_data_ema_filtered, threshold=2)

# Apply normalization to the raw training and test data sets
train_data_normalized = dpp.normalize_data(train_data_no_outliers)
test_data_normalized = dpp.normalize_data(test_data_no_outliers)
dpp.plot_normalized_data_histograms(train_data_normalized)
dpp.plot_normalized_data_histograms(test_data_normalized)


# Convert the normalized data back into segments
train_data_normalized_segments = de.convert_to_segments(train_data_normalized, window_size, sample_rate)
test_data_normalized_segments = de.convert_to_segments(test_data_normalized, window_size, sample_rate)

# --------------------------------------------------------------------------------
# -------------------- Step 5: Feature Extraction --------------------------------
# Convert the raw data into segments
train_data_segments = de.convert_to_segments(train_data_normalized, window_size, sample_rate)
test_data_segments = de.convert_to_segments(test_data_normalized, window_size, sample_rate)

# Extract features from the segments
train_data_features = de.extract_features(train_data_segments)
test_data_features = de.extract_features(test_data_segments)

# Saving the extracted features into a csv file.
train_data_filename = "Data/Features/train_data_features.csv"
test_data_filename = "Data/Features/test_data_features.csv"

de.save_features_to_csv(train_data_features, train_data_filename)
de.save_features_to_csv(test_data_features, test_data_filename)


# --------------------------------------------------------------------------------
# -------------------- Step 6: Train and Evaluate Classifier ---------------------
# Extract target variable for each segment
train_data_y = [1 if segment['Activity_Jumping'].iloc[0] == 1 else 0 for segment in train_data_segments]
test_data_y = [1 if segment['Activity_Jumping'].iloc[0] == 1 else 0 for segment in test_data_segments]


print("train_data_normalized_y length:", len(train_data_y))
print("test_data_normalized_y length:", len(test_data_y))

# Before SMOTE
print("Original class distribution:")
print(pd.Series(train_data_y).value_counts())


# Handle class imbalance
train_data_features, train_data_y = dpp.handle_imbalance(train_data_features, train_data_y)


# After SMOTE
print("Resampled class distribution:")
print(pd.Series(train_data_y).value_counts())

# Train and evaluate the classifier
log_reg_model = dc.train_and_evaluate_logistic_regression(train_data_y, test_data_y, train_data_features, test_data_features)


# --------------------------------------------------------------------------------
# -------------------- Step 7: Deploying Classifier to An App ---------------------
# Saving the logistic regression model to a file for the app
joblib.dump(log_reg_model, 'log_reg_model.pkl')
