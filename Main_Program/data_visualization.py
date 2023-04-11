#
# Data Visualization: Task 3

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram


# Visualizing the dataset with the raw data from all three CSV Files
def plot_general_acceleration_vs_time(time, accelerationX, accelerationY, accelerationZ):
    plt.plot(time, accelerationX, color='red', label="Acceleration X")
    plt.plot(time, accelerationY, color='green', label="Acceleration Y")
    plt.plot(time, accelerationZ, color='blue', label="Acceleration Z")
    plt.title("General Acceleration [m/s^2] vs Time [sec] Line Plot")
    plt.legend()
    plt.show()


# Visualizing the meta-data - time both through a gantt chart and through a bar plot
def plot_md_time_gantt(df_md_time_list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(wspace=0.3)

    for i, (df_md_time, ax) in enumerate(zip(df_md_time_list, axes)):
        for j, row in df_md_time.iterrows():
            ax.barh(j,
                    row['end_time'] - row['experiment time'],
                    left=row['experiment time'],
                    height=0.4,
                    align='center',
                    color='blue')

            ax.text(row['experiment time'], j, row['event'], ha='right', va='center')

        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_yticks([])
        ax.set_title(f'Gantt Chart {i + 1}')

        # Optional: Display the system time on the secondary x-axis
        ax2 = ax.twiny()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax2.set_xlabel('System Time', fontsize=8)
        ax2.set_xlim(df_md_time['system_time_dt'].min(), df_md_time['system_time_dt'].max())

    plt.show()


# Creating the bar plot
def plot_md_time_bar(df_md_time_list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(wspace=0.3)

    for i, (df_md_time, ax) in enumerate(zip(df_md_time_list, axes)):
        ax.bar(np.arange(len(df_md_time)), df_md_time['duration'], tick_label=df_md_time['event'])

        # Set the chart labels
        ax.set_xlabel('Event', fontsize=9)
        ax.set_ylabel('Duration (s)')
        ax.set_title(f'Duration of Events From Meta Data {i + 1}')

        # Reduce the font size of the x-axis and y-axis tick labels
        ax.tick_params(axis='both', which='major', labelsize=5)

    # Display the chart
    plt.show()


# Separating the Walking and Jumping Activities into classes
class Activity:
    def __init__(self, data, data_source):
        self.data = data
        self.data_source = data_source

    def filter_activity(self, activity):
        return self.data[self.data[f'Activity_{activity}'] == 1]

    # Code to plot the scatter plot
    def plot_scatter(self, activity):
        data = self.filter_activity(activity)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.subplots_adjust(wspace=0.3)

        axes[0].scatter(data['Time (s)'], data['Acceleration x (m/s^2)'], color='red', label="Acceleration X",
                        alpha=0.5, s=6)
        axes[0].set_title(f"Acceleration X vs Time")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Acceleration (m/s^2)")

        axes[1].scatter(data['Time (s)'], data['Acceleration y (m/s^2)'], color='green', label="Acceleration Y",
                        alpha=0.5, s=6)
        axes[1].set_title(f"Acceleration Y vs Time")
        axes[1].set_xlabel("Time (s)")

        axes[2].scatter(data['Time (s)'], data['Acceleration z (m/s^2)'], color='blue', label="Acceleration Z",
                        alpha=0.5, s=6)
        axes[2].set_title(f"Acceleration Z vs Time")
        axes[2].set_xlabel("Time (s)")

        fig.suptitle(f"{activity}: Scatter Plots of Accelerations vs Time of the {self.data_source} Data", fontsize=14)
        plt.show()

    # Code for the histogram
    def plot_histogram(self, activity):
        data = self.filter_activity(activity)
        plt.hist(data['Acceleration x (m/s^2)'], bins=50, color='red', alpha=0.5, label="Acceleration X")
        plt.hist(data['Acceleration y (m/s^2)'], bins=50, color='green', alpha=0.5, label="Acceleration Y")
        plt.hist(data['Acceleration z (m/s^2)'], bins=50, color='blue', alpha=0.5, label="Acceleration Z")
        plt.title(f"{activity}: Histogram of Acceleration of the {self.data_source} Data")
        plt.xlabel("Acceleration (m/s^2)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    # Code for the bar plot of the mean
    def plot_mean_bar(self, activity):
        data = self.filter_activity(activity)
        mean_accelerations = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']].mean()

        plt.bar(['Acceleration X', 'Acceleration Y', 'Acceleration Z'], mean_accelerations,
                color=['darkred', 'darkgreen', 'midnightblue'], edgecolor='black', linewidth=1)
        plt.title(f"{activity}: Mean Accelerations for the {self.data_source} Data")
        plt.xlabel("Acceleration Axis")
        plt.ylabel("Mean Acceleration (m/s^2)")
        plt.show()

    # Plot for the spectrogram
    def plot_spectrogram(self, activity):
        data = self.filter_activity(activity)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)

        acceleration_axes = ['x', 'y', 'z']
        acceleration_labels = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)']
        colors = ['Reds', 'Greens', 'Blues']

        for i, (ax, accel_axis, accel_label, cmap) in enumerate(
                zip(axes, acceleration_axes, acceleration_labels, colors)):
            # Compute the spectrogram
            f, t, Sxx = spectrogram(data[f'Acceleration {accel_axis} (m/s^2)'], fs=50, nperseg=128, noverlap=64)

            # Plot the spectrogram
            im = ax.pcolormesh(t, f, Sxx, cmap=cmap, shading='auto')
            fig.colorbar(im, ax=ax, label='Power')

            # Set the plot labels and titles
            ax.set_title(f'{activity}: Spectrogram of {accel_label} for the {self.data_source} Data')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')

        plt.show()
