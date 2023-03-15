import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Creating the class walking
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
        plt.hist([self.accelerationX, self.accelerationY, self.accelerationZ], bins=20, color=['red', 'green', 'blue'], alpha=0.5)
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


# Creating the class Jumping
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
        plt.hist([self.accelerationX, self.accelerationY, self.accelerationZ], bins=20, color=['red', 'green', 'blue'], alpha=0.5)
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


df = pd.read_csv("../All_Group_Member_Shams_Harshil_Jasmine-Data.csv")
timeSec = df['Total Sec']
accelerationX = df["Acceleration x (m/s^2)"]
accelerationY = df["Acceleration y (m/s^2)"]
accelerationZ = df["Acceleration z (m/s^2)"]


# Create Walking and Jumping objects based on conditions for each activity
walking_mask = ((accelerationX < 20) & (accelerationX > -20)) & ((accelerationY < 20) & (accelerationY > -20)) & ((accelerationZ < 15) & (accelerationZ > -15))
walking_data = df[walking_mask]
walking = Walking(walking_data['Total Sec'], walking_data['Acceleration x (m/s^2)'], walking_data['Acceleration y (m/s^2)'], walking_data['Acceleration z (m/s^2)'])

jumping_mask = ~walking_mask
jumping_data = df[jumping_mask]
jumping = Jumping(jumping_data['Total Sec'], jumping_data['Acceleration x (m/s^2)'], jumping_data['Acceleration y (m/s^2)'], jumping_data['Acceleration z (m/s^2)'])

# Plot acceleration data for Walking and Jumping activities
walking.plot_acceleration()
jumping.plot_acceleration()
