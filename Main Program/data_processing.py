# data_processing_test_code.py

import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Function to process the data from the csv files
def process_data(csv_file_paths):
    # Number of Data collected per second
    sample_rate = 100
    window_duration = 5  # In seconds
    rows_per_window = sample_rate * window_duration

    # Creating a dataframe to store all the data
    segmented_dfs = []

    # Reading all the CSV Files and loading them into the Pandas Dataframe
    for file_path in csv_file_paths:
        df = pd.read_csv(file_path)

        # Segmenting the data into 5 second windows and assigning labels
        windows = []
        labels = []

        for i in range(0, len(df), rows_per_window):
            window = df.iloc[i:i + rows_per_window]

            # Checking length of window to ensure that only complete 5-second windows are used.
            if len(window) == rows_per_window:
                windows.append(window)  # Appends if window is equal to rows_per_window
                labels.append(i // rows_per_window)  # Generating a unique label for each 5-second window

        # Concatenate the windows and labels data to create the final dataset
        segmented_df = pd.concat(windows).assign(window_label=labels * rows_per_window)
        segmented_dfs.append(segmented_df)

    # Combine the segmented dataframes from all csv files
    combined_df = pd.concat(segmented_dfs)
    # Split the combined dataframe into training data and test data
    train_df, test_df = train_test_split(combined_df, test_size=0.1, random_state=42,
                                         stratify=combined_df['window_label'])

    return train_df, test_df


# Function to write the data to the HDF5 file
def write_data_to_h5(train_df, test_df, csv_file_paths, h5_filename='data.h5'):
    # Writing and saving the training and testing dataset into the HDF5 file
    with h5py.File('data.h5', 'w') as h5f:
        # Create a group for the original data
        original_data_group = h5f.create_group('Original_data')

        # Iterating through the CSV files and storing the original data in the HDF5 file
        for csv_file in csv_file_paths:
            file_name = os.path.splitext(os.path.basename(csv_file))[0]
            df = pd.read_csv(csv_file)
            original_data_group.create_dataset(file_name, data=df.to_numpy())

        # Create groups for the training and test data
        training_group = h5f.create_group("Training")
        testing_group = h5f.create_group("Testing")

        # Storing the training and test data in the HDF5 file
        training_group.create_dataset('data', data=train_df.drop(columns='window_label').to_numpy())
        training_group.create_dataset('labels', data=train_df['window_label'].to_numpy())

        testing_group.create_dataset('data', data=test_df.drop(columns='window_label').to_numpy())
        testing_group.create_dataset('labels', data=test_df['window_label'].to_numpy())


# Function to read the data from the HDF5 file
def read_data_from_h5(h5_filename='data.h5'):
    # Reading the HDF5 file
    with h5py.File(h5_filename, 'r') as h5f:
        # Accessing the original data group
        original_data_group = h5f['Original_data']

        # Retrieve the original data
        original_data = {}
        for dataset_name in original_data_group.keys():
            original_data[dataset_name] = original_data_group[dataset_name][()]

        # Accessing the training and test data groups
        training_group = h5f['Training']
        testing_group = h5f['Testing']

        # Read the training data
        train_data = np.array(training_group['data'])
        train_labels = np.array(training_group['labels'])

        # Read the testing data
        test_data = np.array(testing_group['data'])
        test_labels = np.array(testing_group['labels'])

    return original_data, train_data, train_labels, test_data, test_labels


# Function to print the data
def display_data(original_data, train_data, train_labels, test_data, test_labels):
    # Display the original data
    print("Original Data:")
    for dataset_name, data in original_data.items():
        print(f"\nOriginal data for {dataset_name}: ")
        print(pd.DataFrame(data))

    # Display the training data
    print("\nTraining Data: ")
    print(pd.DataFrame(train_data))
    print("\nTraining Labels: ")
    print(pd.DataFrame(train_labels))

    # Display the testing data
    print("\nTesting Data: ")
    print(pd.DataFrame(test_data))
    print("\nTesting Labels: ")
    print(pd.DataFrame(test_labels))

