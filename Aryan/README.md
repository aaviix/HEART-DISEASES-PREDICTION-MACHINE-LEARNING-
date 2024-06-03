# Heart Disease Prediction Model

This project uses a Linear Regression model to predict the presence of heart disease based on various health and demographic factors. The model is trained and evaluated using a dataset containing information about patients and their health metrics.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Model](#model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Installation](#installation)

## Project Description

This project involves the following steps:
1. **Loading Data:** Load the dataset from a CSV file.
2. **Preprocessing Data:** Select specific columns, and convert categorical data to numerical values.
3. **Splitting Data:** Split the dataset into training and testing sets.
4. **Training Model:** Train a Linear Regression model on the training data.
5. **Evaluating Model:** Evaluate the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²) metrics.
6. **Predicting Inputs:** Predict the presence of heart disease for new inputs.

## Data

The dataset used in this project contains the following columns:
- Age
- Sex
- Chest pain type
- Max HR
- Exercise angina
- ST depression
- Slope of ST
- Number of vessels fluro
- Thallium
- Heart Disease (target variable)

The target variable, 'Heart Disease', is converted from categorical values ('Absence', 'Presence') to numerical values (0, 1).

## Model

A Linear Regression model is used in this project. This model is trained to predict the presence of heart disease based on the selected features.

## Evaluation

The model is evaluated using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Additionally, an accuracy-like metric is calculated based on a threshold value to determine the model's performance on training and testing data.

## Usage

1. **Load Data:** Modify the `file_path` variable in the `main` function to point to the CSV file containing the dataset.
2. **Run the Script:** Execute the `main` function to load data, preprocess, train, evaluate, and make predictions.

```python
if __name__ == "__main__":
    main()
