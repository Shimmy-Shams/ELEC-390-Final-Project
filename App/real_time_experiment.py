# This code will take the real time data from the Phyphox URL and then give the predictions in real time

# Importing the necessary libraries
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Enter URL here
url = "http://192.168.2.39/"


def setup_driver():
    # Initialize the Chrome WebDriver and open the provided URL
    driver = webdriver.Chrome()
    driver.get(url)

    # Click on the "Simple" mode
    driver.find_element(By.XPATH, "//li[text()='Simple']").click()

    return driver


def fetch_data_from_phyphox(driver):
    # Wait for the presence of the acceleration elements and fetch them
    accel_x_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "element10")))
    accel_y_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "element11")))
    accel_z_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "element12")))
    abs_accel_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "element13")))

    # Extract acceleration data for each axis
    accel_x = extract_accel_data(accel_x_element)
    accel_y = extract_accel_data(accel_y_element)
    accel_z = extract_accel_data(accel_z_element)
    abs_accel = extract_accel_data(abs_accel_element)

    # Create a dictionary containing the time and acceleration data
    data = {
        "time": time.time(),
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "abs_accel": abs_accel,
    }

    return data


def extract_accel_data(element):
    # Find the value number element within the provided element
    value_number = element.find_element(By.CSS_SELECTOR, ".value > .valueNumber")
    value_text = value_number.text

    # If the value text is empty, return 0.0
    if value_text == "":
        return 0.0

    # Convert the value text to a float and return it
    data_value = float(value_text)
    return data_value
