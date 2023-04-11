# Main.py file for the app

from tkinter import filedialog
import tkinter as tk
import pandas as pd
import joblib
from Main_Program.data_extracting_features import extract_features, convert_to_segments, save_features_to_csv
from Main_Program.data_pre_processing import check_nan_null_values, handle_missing_values, exponential_moving_average_filter, normalize_data, remove_outliers, \
    moving_average_filter


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

    # Applying the exponential moving average
    data_filtered = exponential_moving_average_filter(data_filtered_window_size)

    data_no_outliers = remove_outliers(data_filtered, threshold=3)
    data_normalized = normalize_data(data_no_outliers)

    # Segment the data
    segmented_data = convert_to_segments(data_normalized, window_size=5, sample_rate=100)

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
    results_window.geometry('1300x800')
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
    results_text = tk.Text(text_frame, wrap=tk.WORD, padx=10, pady=10, width=150, height=40, yscrollcommand=scrollbar.set)
    results_text.insert(tk.INSERT, results.to_string(index=False))
    results_text.pack(side=tk.LEFT, fill=tk.BOTH)

    # Connect the scrollbar to the text widget
    scrollbar.config(command=results_text.yview)


# Function to browse and open a file using a file dialog, then call classify_motion() and display the output file path
def browse_file():
    filepath = filedialog.askopenfilename()
    _, results = classify_motion(filepath)
    display_results(results)
    my_label1.config(text=f"Output file: output.csv")


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

    instr_text = tk.Label(instr_window, text="1. Click the 'Start' button to select a CSV file.\n"
                                             "\n2. Wait for the file to be processed and the output file path to be displayed.\n"
                                             "\n3. Check the output file for predicted labels.",
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

# Define button images
start_button = tk.PhotoImage(file='StartB.png')
about_button = tk.PhotoImage(file='AboutB.png')

# Add title label to the window
my_title = tk.Label(root, text="Motion Mapper", font=("Helvetica", 50), fg="black")
my_title.pack(pady=(150, 20))

# Create a frame to hold the buttons
my_frame = tk.Frame(root)
my_frame.pack(pady=20)

# Add Start button to the frame and link it to browse_file() function
myStart_button = tk.Button(my_frame, image=start_button, command=browse_file)
myStart_button.grid(row=0, column=0, padx=20)

# Add label to display output file path
my_label1 = tk.Label(root, text="")
my_label1.pack(pady=20)

# Add About button to the frame and link it to instruction() function
myAbout_button = tk.Button(my_frame, image=about_button, command=instruction)
myAbout_button.grid(row=0, column=1, padx=20)

# Add label to display instructions
my_label2 = tk.Label(root, text="")
my_label2.pack(pady=20)

# Load the saved model
log_reg_model = joblib.load(
    'C:/Users/User/Desktop/QUEENS U/ELEC 390/ELEC 390 Final Project/Main_Program/log_reg_model.pkl')

# Run the Tkinter window
tk.mainloop()
