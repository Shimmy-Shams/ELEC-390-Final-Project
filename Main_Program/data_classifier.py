#
# Data Classifier:  This code checks creates a classifier by training a logistic regression model.
#                   It will classify the data into walking or jumping class. Once the training is complete it will
#                   apply on the test set and record the accuracy.
#


# Importing the necessary libraries
import data_extracting_features as de
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Creating the function to train the logistic regression model
def train_and_evaluate_logistic_regression(train_data_normalized_y, test_data_normalized_y,
                                           train_data_features, test_data_features):
    # Extract features from the train_data_normalized
    # Assuming 'Activity_Walking' column represents the walking class (1 for walking and 0 for jumping)
    X_train = train_data_features
    y_train = train_data_normalized_y

    # Splitting the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create the logistic regression model
    log_reg = LogisticRegression(max_iter=10000)

    # Train the model
    log_reg.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = log_reg.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print("Validation Accuracy:", val_accuracy)
    print("Classification Report:\n", classification_report(y_val, y_val_pred,  zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

    # Extract features from the test_data_normalized
    # Assuming 'Activity_Walking' column represents the walking class (1 for walking and 0 for jumping)
    X_test = test_data_features
    y_test = test_data_normalized_y

    # Evaluate the model on the test set
    y_test_pred = log_reg.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy:", test_accuracy)
    print("Classification Report:\n", classification_report(y_test, y_test_pred,  zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    return log_reg
