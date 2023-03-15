import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../All_Group_Member_Shams_Harshil_Jasmine-Data.csv")
timeSec = df['Total Sec']

accelerationX = df["Acceleration x (m/s^2)"]
accelerationY = df["Acceleration y (m/s^2)"]
accelerationZ = df["Acceleration z (m/s^2)"]

# Create a boolean mask for walking vs jumping
walking_mask = ((accelerationX < 20) & (accelerationX > -20)) & ((accelerationY < 20) & (accelerationY > -20)) & ((accelerationZ < 15) & (accelerationZ > -15))
jumping_mask = ~walking_mask


# ----------- First Graph -------------
plt.plot(timeSec[walking_mask], accelerationX[walking_mask], color='red', label="Walking")
plt.plot(timeSec[jumping_mask], accelerationX[jumping_mask], color='blue', label="Jumping")
plt.title("Acceleration X [m/s^2] vs Time Graph [sec]")
plt.legend()
plt.show()


# ------------ Second Graph --------------
plt.plot(timeSec[walking_mask], accelerationY[walking_mask], color='purple', label="Walking")
plt.plot(timeSec[jumping_mask], accelerationY[jumping_mask], color='orange', label="Jumping")
plt.title("Acceleration Y [m/s^2] vs Time Graph [sec]")
plt.legend()
plt.show()


# ------------ Third Graph --------------
plt.plot(timeSec[walking_mask], accelerationZ[walking_mask], color='green', label="Walking")
plt.plot(timeSec[jumping_mask], accelerationZ[jumping_mask], color='blue', label="Jumping")
plt.title("Acceleration Z [m/s^2] vs Time Graph [sec]")
plt.legend()
plt.show()