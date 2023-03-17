
# Main.py

# Importing all the necessary libraries
import h5py
import matplotlib.pyplot as plt
from data_processing import process_data, write_data_to_h5, read_data_from_h5, display_data


csv_file_paths = [
    'Data/Raw_Data_Harshil.csv',
    'Data/Raw_Data_Shams.csv',
    'Data/Raw_Data_Jasmine.csv'
]

# Read and segment the data
train_df, test_df = process_data(csv_file_paths)

# Save the data to the HDF5 file
write_data_to_h5(train_df, test_df, csv_file_paths)

# Read the data from the HDF5 file
original_data, train_data, train_labels, test_data, test_labels = read_data_from_h5()

# Display the data using the display_data() function
display_data(original_data, train_data, train_labels, test_data, test_labels)


# Creating the class walking with all the necessary diagrams.
class Walking:
    def __init__(self, time, accelerationX, accelerationY, accelerationZ):
        self.time = time
        self.accelerationX = accelerationX
        self.accelerationY = accelerationY
        self.accelerationZ = accelerationZ

    def plot_acceleration(self):
        # Plotting a line plot the graphs for each Axes
        plt.plot(self.time, self.accelerationX, color='red', label="Acceleration X")
        plt.plot(self.time, self.accelerationY, color='green', label="Acceleration Y")
        plt.plot(self.time, self.accelerationZ, color='blue', label="Acceleration Z")
        plt.title("Walking Acceleration [m/s^2] vs Time Graph [sec]")
        plt.legend()
        plt.show()

        # Plotting a histogram of the walking acceleration volumes
        plt.hist([self.accelerationX, self.accelerationY, self.accelerationZ], bins=20, color=['red', 'green', 'blue'],
                 alpha=0.5)
        plt.title("Walking Acceleration Histogram")
        plt.xlabel("Acceleration [m/s^2]")
        plt.ylabel("Count")
        plt.legend(["X", "Y", "Z"])
        plt.show()

        # Scatter plot of walking acceleration Y vs X
        plt.scatter(self.accelerationX, self.accelerationY, alpha=0.5, s=1)
        plt.title("Walking Acceleration Y vs X Scatter Plot")
        plt.xlabel("Acceleration X [m/s^2]")
        plt.ylabel("Acceleration Y [m/s^2]")
        plt.show()


# Creating the class Jumping with all the necessary diagrams.
class Jumping:
    def __init__(self, time, accelerationX, accelerationY, accelerationZ):
        self.time = time
        self.accelerationX = accelerationX
        self.accelerationY = accelerationY
        self.accelerationZ = accelerationZ

    def plot_acceleration(self):
        # Plotting all the graphs for each Axes
        plt.plot(self.time, self.accelerationX, color='red', label="Acceleration X")
        plt.plot(self.time, self.accelerationY, color='green', label="Acceleration Y")
        plt.plot(self.time, self.accelerationZ, color='blue', label="Acceleration Z")
        plt.title("Jumping Acceleration [m/s^2] vs Time Graph [sec]")
        plt.legend()
        plt.show()

        # Plotting a histogram of the jumping acceleration volumes
        plt.hist([self.accelerationX, self.accelerationY, self.accelerationZ], bins=20, color=['red', 'green', 'blue'],
                 alpha=0.5)
        plt.title("Jumping Acceleration Histogram")
        plt.xlabel("Acceleration [m/s^2]")
        plt.ylabel("Count")
        plt.legend(["X", "Y", "Z"])
        plt.show()

        # Scatter plot of jumping acceleration Y vs X
        plt.scatter(self.accelerationX, self.accelerationY, alpha=0.5, s=1)
        plt.title("Jumping Acceleration Y vs X Scatter Plot")
        plt.xlabel("Acceleration X [m/s^2]")
        plt.ylabel("Acceleration Y [m/s^2]")
        plt.show()
