import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(file_path):
    ## Load the data from a CSV file.
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None

def preprocess_data(data):
    ## Preprocess the data by keeping only the specified columns and separating features and target variable.
    columns_to_keep = ['Age', 'Sex', 'Chest pain type', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
    data = data.loc[:, columns_to_keep]
    # Convert target variable to numerical
    data.loc[:, 'Heart Disease'] = data['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    X = data.drop(columns='Heart Disease', axis=1)
    Y = data['Heart Disease']
    return X, Y

def split_data(X, Y, test_size=0.25, random_state=2):
    ## Split the data into training and testing sets.
    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)

def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_train, y_train, x_test, y_test):
    ## Evaluate the model on training and testing data.
    # Training data evaluation
    x_train_predict = model.predict(x_train)
    train_mse = mean_squared_error(y_train, x_train_predict)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, x_train_predict)
    train_r2 = r2_score(y_train, x_train_predict)
    print(f"Training data - MSE: {train_mse}, RMSE: {train_rmse}, MAE: {train_mae}, R2: {train_r2}")

    # Testing data evaluation
    x_test_predict = model.predict(x_test)
    test_mse = mean_squared_error(y_test, x_test_predict)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, x_test_predict)
    test_r2 = r2_score(y_test, x_test_predict)
    print(f"Test data - MSE: {test_mse}, RMSE: {test_rmse}, MAE: {test_mae}, R2: {test_r2}")

    # Calculate accuracy-like metric
    threshold = 0.5  
    accuracy_train = np.mean(np.abs(y_train - x_train_predict) <= threshold)
    accuracy_test = np.mean(np.abs(y_test - x_test_predict) <= threshold)
    print(f"Training data - Accuracy within {threshold}: {accuracy_train}")
    print(f"Test data - Accuracy within {threshold}: {accuracy_test}")

def predict_inputs(model, inputs):
    ## Predict the class for given test inputs.
    feature_names = ['Age', 'Sex', 'Chest pain type', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    for i, test_input in enumerate(inputs, start=1):
        test_input_df = pd.DataFrame([test_input], columns=feature_names)
        try:
            prediction = model.predict(test_input_df)
            print(f'Prediction for input {i}: {prediction}')
        except Exception as e:
            print(f"Error predicting input {i}: {e}")

def main():
    ## Main function to load data, preprocess, train model, evaluate and predict.
    file_path = 'C:/Users/aryan/Desktop/Machine Learning/Aryan/Heart_Disease_Prediction.csv'
    data = load_data(file_path)
    if data is not None:
        X, Y = preprocess_data(data)
        x_train, x_test, y_train, y_test = split_data(X, Y)
        model = train_model(x_train, y_train)
        evaluate_model(model, x_train, y_train, x_test, y_test)

        # Test inputs for prediction (make sure they match the new feature set after dropping columns)
        test_inputs = [
            (50, 1, 0, 140, 1, 2.6, 2, 0, 3),
            (58, 0, 2, 150, 0, 1.0, 1, 1, 7)
        ]
        predict_inputs(model, test_inputs)

if __name__ == "__main__":
    main()
