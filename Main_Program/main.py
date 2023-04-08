# Final Project
# Group: Shams, Harshil and Jasmine
# Course: ELEC 390

# Main.py

# Importing all the necessary libraries
import data_extracting_features as de
import data_pre_processing as dpp
import data_visualization as dv
import data_processing as dp
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------
# --------------- Step 1 and 2: Data Collection & Storage ------------------------

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

window_size = 5  # 5 seconds
sample_rate = 100  # Assuming 100 samples per second, adjust if necessary

# Load data
data = dp.load_data(csv_file_paths)

# Segment data into 5-second windows
segments = dp.segment_data(data, window_size, sample_rate)

# Shuffle and create train and test splits
train_data, test_data = dp.create_splits(segments)


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


# Select the filtered data with window size 50
selected_window_size = 50
train_data_filtered_50 = train_data_filtered[f'window_size_{selected_window_size}']
test_data_filtered_50 = test_data_filtered[f'window_size_{selected_window_size}']

# Removing outliers
train_data_no_outliers = dpp.remove_outliers(train_data_filtered_50, threshold=1.5)
test_data_no_outliers = dpp.remove_outliers(test_data_filtered_50, threshold=1.5)

# Applying normalization
train_data_normalized = dpp.normalize_data(train_data_no_outliers)
test_data_normalized = dpp.normalize_data(test_data_no_outliers)


# Save data to HDF5 file
dp.write_to_hdf5(data, csv_file_paths, train_data_normalized, test_data_normalized)
original_data, train_data_normalized, test_data_normalized = dp.reading_from_hdf5()
#dp.display_data(original_data, train_data_normalized, test_data_normalized)

# --------------------------------------------------------------------------------
# -------------------- Step 5: Feature Extraction --------------------------------
# Convert the normalized data back into segments
train_data_normalized_segments = de.convert_to_segments(train_data_normalized, window_size, sample_rate)
test_data_normalized_segments = de.convert_to_segments(test_data_normalized, window_size, sample_rate)

# Use feature extraction
train_data_features = de.extract_features(train_data_normalized_segments)
test_data_features = de.extract_features(test_data_normalized_segments)

# Saving the extracted features into a csv file.
train_data_filename = "Data/train_data_features.csv"
test_data_filename = "Data/test_data_features.csv"

de.save_features_to_csv(train_data_features, test_data_features, train_data_filename, test_data_filename)

# --------------------------------------------------------------------------------
# -------------------- Step 3: Data Visualization --------------------------------

# Extract the original data from the dp file and visualize it with the dv file
for dataset_name in original_data.keys():
    print(f"Visualizing data for {dataset_name}")
    time, accelerationX, accelerationY, accelerationZ = dp.extract_time_and_acceleration(original_data, dataset_name)
    #dv.plot_general_acceleration_vs_time(time, accelerationX, accelerationY, accelerationZ)


# Visualization of the meta data
df_md_time_list = [dp.process_meta_data_time(time_file) for time_file in meta_data_time]
#dv.plot_md_time_gantt(df_md_time_list)
#dv.plot_md_time_bar(df_md_time_list)


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
