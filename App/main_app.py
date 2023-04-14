# Main.py file for the app

# Importing all the necessary libraries
import time
import joblib
import keyboard
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from real_time_experiment import url, setup_driver, fetch_data_from_phyphox, extract_accel_data
from Main_Program.data_extracting_features import extract_features, convert_to_segments, save_features_to_csv
from Main_Program.data_pre_processing import check_nan_null_values, handle_missing_values, \
    exponential_moving_average_filter, normalize_data, remove_outliers, moving_average_filter


# Load the saved model
log_reg_model = joblib.load('C:/Users/User/Desktop/QUEENS U/ELEC 390/ELEC 390 Final Project/Main_Program/log_reg_model.pkl')


def clean_data_columns(data):
    # Rename acceleration columns
    column_mapping = {}
    for col in data.columns:
        if 'x' in col.lower() and ('accel' in col.lower() or 'acc' in col.lower()):
            column_mapping[col] = 'Acceleration x (m/s^2)'
        elif 'y' in col.lower() and ('accel' in col.lower() or 'acc' in col.lower()):
            column_mapping[col] = 'Acceleration y (m/s^2)'
        elif 'z' in col.lower() and ('accel' in col.lower() or 'acc' in col.lower()):
            column_mapping[col] = 'Acceleration z (m/s^2)'

    data = data.rename(columns=column_mapping)

    return data


# Function to process the input CSV file, extract features, and predict the labels using the loaded model
def classify_motion(filepath):
    # Read the input CSV file
    data = pd.read_csv(filepath)
    data_name = "Input Data File"

    # Clean the input data file
    data = clean_data_columns(data)

    # Preprocess the data
    data_check_null = check_nan_null_values(data, data_name)

    # Handle Missing Values
    data_handle_missing_values = handle_missing_values(data, method='drop')

    # Apply Moving Average
    data_filtered_dict = moving_average_filter(data_handle_missing_values)

    # Select the filtered data with window size 50
    selected_window_size = 100
    data_filtered_window_size = data_filtered_dict[f'window_size_{selected_window_size}']

    data_normalized = normalize_data(data_filtered_window_size)

    # Segment the data
    segmented_data = convert_to_segments(data_filtered_window_size, window_size=5, sample_rate=100)

    # Extract features
    features = extract_features(segmented_data)

    # Saving Extracted Files to CSV
    features_filename = 'C:/Users/User/Desktop/QUEENS U/ELEC 390/ELEC 390 Final Project/Main_Program/Data/Features/Input_File_Features.csv'
    save_features_to_csv(features, features_filename)

    # Predict labels
    predictions = log_reg_model.predict(features)

    # Add a new 'Prediction' column with 'None' values
    data_normalized['Prediction'] = None

    # Assign predictions to the appropriate rows
    for idx, segment in enumerate(segmented_data):
        start_index = segment.index[0]
        end_index = segment.index[-1]
        prediction = predictions[idx]
        data_normalized.loc[start_index:end_index, 'Prediction'] = prediction

    # Fill 'None' values in the 'Prediction' column with the previous prediction value
    data_normalized['Prediction'].fillna(method='ffill', inplace=True)

    # Add a new 'Activity' column with 'Unknown' values
    data_normalized['Activity'] = 'Unknown'

    # Assign the activity based on the 'Prediction' column
    data_normalized.loc[data_normalized['Prediction'] == 0, 'Activity'] = 'Walking'
    data_normalized.loc[data_normalized['Prediction'] == 1, 'Activity'] = 'Jumping'

    # Save the output as a CSV file
    data_normalized.to_csv(
        'C:/Users/User/Desktop/QUEENS U/ELEC 390/ELEC 390 Final Project/Main_Program/Data/Output_From_App/output.csv',
        index=False)

    return data_check_null, data_normalized


# Creating a new function to display the results of the data
def display_results(results):
    # Create a new window
    results_window = tk.Toplevel(root)
    results_window.title("Results")
    results_window.geometry('1500x800')
    results_window.config(bg='#ADD8E6')

    # Add label with results to the window
    results_label = tk.Label(results_window, text="Results:",
                             font=("Helvetica", 20), fg="black", bg='#ADD8E6',
                             highlightbackground='#ADD8E6', highlightthickness=0)
    results_label.pack(pady=20)

    # Create a frame to hold the Text widget and the scrollbar
    text_frame = tk.Frame(results_window)
    text_frame.pack(pady=20)

    # Create a scrollbar and place it on the right side of the frame
    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Convert the DataFrame to a string and display it in a text widget
    results_text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10, width=170, height=40,
                           yscrollcommand=scrollbar.set)
    results_text.insert(tk.INSERT, results.to_string(index=False))
    results_text.pack(side=tk.LEFT, fill=tk.BOTH)

    # Connect the scrollbar to the text widget
    scrollbar.config(command=results_text.yview)


# Function to browse and open a file using a file dialog, then call classify_motion() and display the output file path
def browse_file():
    # Open the file dialog to let the user select a CSV file
    filepath = filedialog.askopenfilename()

    # Call classify_motion() to process the selected file and obtain the results
    _, results = classify_motion(filepath)

    # Display the output file path to the user
    display_results(results)


# -----------------------------------------------------------------------
# --------------- Code for the Real time Data ---------------------------
# Function to start the real-time experiment and process the data
def real_time_classification(sleep_interval=0.001, initial_wait=3, data_points=1000000, initial_data_points=500):
    # Set up the driver to collect data from the smartphone
    driver = setup_driver()

    # Wait for initial_wait seconds before starting to collect data
    time.sleep(initial_wait)
    global start_time
    start_time = time.time()

    # Create an empty DataFrame to store the collected data
    data = pd.DataFrame(
        columns=["Time (s)", "Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)",
                 "Absolute acceleration (m/s^2)"])

    # Set an index to keep track of the number of data points collected
    index = 0

    # Continue collecting data until the desired number of data_points is reached
    while len(data) < data_points:
        try:
            # Fetch the accelerometer data from the smartphone using the driver
            data_dict = fetch_data_from_phyphox(driver)

            # Convert the fetched data dictionary into a DataFrame
            df = pd.DataFrame([data_dict])
            df["Time (s)"] = time.time() - start_time  # Calculate relative time

            # Append the fetched data to the main DataFrame
            data.loc[index] = [time.time() - start_time, data_dict["accel_x"], data_dict["accel_y"],
                               data_dict["accel_z"], data_dict["abs_accel"]]

            index += 1

            # Start making predictions after collecting more than initial_data_points
            if len(data) > initial_data_points:
                # Clean the collected data by removing unnecessary columns
                clean_data = clean_data_columns(data)
                data_filtered = moving_average_filter(clean_data)
                selected_window = 100
                data_filtered_window_size = data_filtered[f'window_size_{selected_window}']
                data_ema_filtered = exponential_moving_average_filter(data_filtered_window_size, alpha=0.2)
                data_normalized = normalize_data(data_ema_filtered)

                # Segment the cleaned data into smaller windows for analysis
                segmented_data = convert_to_segments(data_normalized, window_size=1, sample_rate=100)

                # Extract features from the segmented data to be used for classification
                features = extract_features(segmented_data)

                # Predict labels for the data segments using the trained model
                if not features.empty:
                    prediction = log_reg_model.predict(features)

                    # Print the latest prediction and time to the terminal
                    latest_time = data["Time (s)"].iloc[-1]
                    if prediction[-1] == 0:
                        print(f"Time: {latest_time:.2f}s | Latest prediction: Walking")
                    elif prediction[-1] == 1:
                        print(f"Time: {latest_time:.2f}s | Latest prediction: Jumping")
                else:
                    print("No features extracted. Please check the segmentation and feature extraction steps.")

            # Sleep for the specified interval before fetching the data again
            time.sleep(sleep_interval)

            # Check if the escape key has been pressed, and exit the loop if it has
            if keyboard.is_pressed("esc"):
                break

        except KeyboardInterrupt:
            # Break the loop when the user presses Ctrl+C
            break

    # Close the driver when the loop ends
    driver.quit()


# -----------------------------------------------------------------------
# --------------- Code for the Instructions -----------------------------
# Function to display instructions in a new window
def instruction():
    # Create a new window
    instr_window = tk.Toplevel(root)
    instr_window.title("Instructions")
    instr_window.geometry('800x400')
    instr_window.config(bg='#ADD8E6')

    # Add label with instructions to the window
    instr_label = tk.Label(instr_window, text="Here are the instructions:",
                           font=("Helvetica", 20), fg="black", bg='#ADD8E6',
                           highlightbackground='#ADD8E6', highlightthickness=0)
    instr_label.pack(pady=20)

    instr_text = tk.Label(instr_window, text="1. There are two buttons: 'Start (CSV)' and 'Start (Real-time)'.\n"
                                             "\n2. For CSV file processing:"
                                             "\n   a. Click the 'Start (CSV)' button to select a CSV file."
                                             "\n   b. Wait for the file to be processed and the output file path to be displayed."
                                             "\n   c. Check the output file for predicted labels."
                                             "\n\n3. For real-time data processing:"
                                             "\n   a. Click the 'Start (Real-time)' button to begin real-time data collection and prediction."
                                             "\n   b. Perform the actions (walking or jumping) you want the model to classify."
                                             "\n   c. Observe the real-time predictions in the terminal."
                                             "\n   d. Press 'Ctrl + C' in the terminal to stop real-time data collection and prediction.",
                          font=("Helvetica", 12), fg="black", bg='#ADD8E6',
                          highlightbackground='#ADD8E6', highlightthickness=0)
    instr_text.pack(pady=20)


# Initialize the Tkinter window
root = tk.Tk()
root.title("Motion Mapper")
root.geometry('800x500')

# Define background image and place it in the window
bg = tk.PhotoImage(file='bgblu.png')
my_labelBG = tk.Label(root, image=bg)
my_labelBG.place(x=0, y=0, relwidth=1, relheight=1)


# Create custom button class with the desired style
class CustomButton(tk.Button):
    def __init__(self, *args, font_size=12, button_width=20, button_height=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(
            bg="light blue",
            fg="black",
            activebackground="light blue",
            activeforeground="black",
            highlightthickness=0,
            relief="solid",
            bd=2,  # Set border width to 2
            font=("Helvetica", font_size),
            height=1,  # Temporarily set to 1
            width=1  # Temporarily set to 1
        )
        self.grid_propagate(False)  # Prevent the button from resizing
        self.config(height=button_height, width=button_width)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.configure(bg="#97FFFF")

    def on_leave(self, e):
        self.configure(bg="light blue")


# Add title label to the window
my_title = tk.Label(root, text="Motion Mapper", font=("Helvetica", 50), fg="black")
my_title.pack(pady=(150, 20))

# Create a frame to hold the buttons
my_frame = tk.Frame(root)
my_frame.pack(pady=20)

# Add Start button to the frame and link it to browse_file() function
myStart_button = CustomButton(my_frame, text="Experimented Data", command=browse_file, font_size=13, button_width=20,
                              button_height=4)
myStart_button.grid(row=0, column=0, padx=20)

# Add About button to the frame and link it to instruction() function
myAbout_button = CustomButton(my_frame, text="Instructions", command=instruction, font_size=13, button_width=20,
                              button_height=4)
myAbout_button.grid(row=0, column=1, padx=20)

# Add the new button to the frame and link it to the start_real_time_experiment function
myRealTime_button = CustomButton(my_frame, text="Real Time Data", command=real_time_classification, font_size=13,
                                 button_width=20, button_height=4)
myRealTime_button.grid(row=0, column=2, padx=20)

# Run the Tkinter window
tk.mainloop()
