#
# Data Classifier:  This code checks creates a classifier by training a logistic regression model.
#                   It will classify the data into walking or jumping class. Once the training is complete it will
#                   apply on the test set and record the accuracy.
#


# Importing the necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Creating the function to train the logistic regression model
def train_and_evaluate_logistic_regression(train_data_normalized_y, test_data_normalized_y,
                                           train_data_features, test_data_features):
    # Extract features from the train_data_normalized
    X_train = train_data_features
    y_train = train_data_normalized_y

    # Splitting the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create the logistic regression model
    log_reg = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)

    # Create a dictionary of hyperparameters to tune
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Retrieve the best model
    best_log_reg = grid_search.best_estimator_

    # Evaluate the model on the validation set
    y_val_pred = best_log_reg.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print("Validation Accuracy:", val_accuracy)
    print("Classification Report:\n", classification_report(y_val, y_val_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

    # Extract features from the test_data_normalized
    X_test = test_data_features
    y_test = test_data_normalized_y

    # Evaluate the model on the test set
    y_test_pred = best_log_reg.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy:", test_accuracy)
    print("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    return best_log_reg
