#
# Data Processing:  This reads the data from the CSV file, writes it to the HDF5, displays the data
#                   and then cleans the data for any visualization
#

import pandas as pd
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------------------------
# ------------ Reading, Writing, Storing and Printing the Data From the Files -------------------

# Reading the data from the csv file and storing it in one dataframe
def load_data(file_paths):
    dfs = []
    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)

        # One-hot encoding the 'Activity' column: 0 indicates it's True, 1 indicates it's False
        df = pd.concat([df.drop('Activity', axis=1), pd.get_dummies(df['Activity'], prefix='Activity')], axis=1)

        dfs.append(df)
    return pd.concat(dfs)


# Segmenting the dataframe into 5 second windows
def segment_data(df, window_size, sample_rate):
    n_samples = len(df)
    window_length = window_size * sample_rate
    segments = []
    for i in range(0, n_samples, window_length):
        segment = df.iloc[i:i + window_length].copy()
        if len(segment) == window_length:
            segments.append(segment)
    return segments


# Splitting the data into training data and testing data
def create_splits(segments, test_size=0.1):
    train_df, test_df = train_test_split(segments, test_size=test_size, random_state=42)
    train_df = pd.concat(train_df).reset_index(drop=True)
    test_df = pd.concat(test_df).reset_index(drop=True)
    return train_df, test_df


# Function to process the meta data
def process_meta_data_time(meta_data_time):
    df_md_time = pd.read_csv(meta_data_time)

    # Calculate the duration of each time event
    df_md_time['duration'] = df_md_time['experiment time'].diff()
    df_md_time.loc[0, 'duration'] = 0

    # Calculate the end time of each event
    df_md_time['end_time'] = df_md_time['experiment time'].shift(-1)
    df_md_time.loc[len(df_md_time) - 1, 'end time'] = df_md_time['experiment time'].iloc[-1]

    # Convert the system to datetime
    df_md_time['system_time_dt'] = pd.to_datetime(df_md_time['system time text'])

    return df_md_time


# Writing the data to a hdf5 file
def write_to_hdf5(original_data, file_paths, train_df, test_df):
    with h5py.File('Combined_data.hdf5', 'w') as h5f:
        start_idx = 0
        for file_path in file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path)

            # Calculate the number of rows for this CSV file
            rows_count = len(df)

            # Extract the relevant rows from the combined data
            person_data = original_data[start_idx:start_idx + rows_count]

            # Create a group for the current person and store their data
            person_group = h5f.create_group(file_name)
            person_group.create_dataset('data', data=person_data.to_numpy())

            start_idx += rows_count

        # Creating the dataset group with train and test groups
        dataset_group = h5f.create_group('dataset')
        train_group = dataset_group.create_group('train')
        test_group = dataset_group.create_group('test')

        # Storing the train and test data in their respective groups
        train_group.create_dataset('train_data', data=train_df.values)
        test_group.create_dataset('test_data', data=test_df.values)


# Reading the contents in the HDF5 file
def reading_from_hdf5(h5_filename='Combined_data.hdf5'):
    with h5py.File(h5_filename, 'r') as f:
        original_data = {}
        for dataset_name in f.keys():
            if dataset_name != 'dataset':
                original_data[dataset_name] = f[dataset_name]['data'][()]

        train_data = f['dataset']['train']
        test_data = f['dataset']['test']

        column_names = ['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                        'Absolute acceleration (m/s^2)', 'Activity_Walking', 'Activity_Jumping']

        train_data = pd.DataFrame(train_data['train_data'][()], columns=column_names)
        test_data = pd.DataFrame(test_data['test_data'][()], columns=column_names)

    return original_data, train_data, test_data


# Code for displaying the overall data
def display_data(original_data, train_data, test_data):
    # Displaying original data and data from the HDF5 file
    # Defining the column names
    column_names = ['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                    'Absolute acceleration (m/s^2)', 'Activity_Walking', 'Activity_Jumping']

    # Display the original data from each group member
    print("Original Data:")
    for dataset_name, data in original_data.items():
        print(f"\nOriginal data for {dataset_name}: ")
        print(pd.DataFrame(data, columns=column_names))

    # Printing the test and train data
    print("\nTrain Data (from HDF5):")
    print(train_data)

    print("\nTest Data (from HDF5):")
    print(test_data)


# -----------------------------------------------------------------------------------------
# ------------- Functions Needed for Accurate Data Visualization --------------------------

# Additional function to extract time and acceleration from the original data for the data_visualization file
def extract_time_and_acceleration(original_data, dataset_name):
    # Defining the column names:
    column_names = ['Time (s)', 'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
                    'Absolute acceleration (m/s^2)', 'Activity_Walking', 'Activity_Jumping']

    data = pd.DataFrame(original_data[dataset_name], columns=column_names)
    time = data['Time (s)']
    accelerationX = data['Acceleration x (m/s^2)']
    accelerationY = data['Acceleration y (m/s^2)']
    accelerationZ = data['Acceleration z (m/s^2)']

    return time, accelerationX, accelerationY, accelerationZ
