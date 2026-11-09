# Experiment 3: Ridge and Lasso Regression

## Algorithm

1. Import the required libraries for data handling, regression, evaluation, and plotting.
2. Load the built-in Diabetes dataset from scikit-learn.
3. Create a data table using the input features and target values.
4. Separate the input features (X) and target (y).
5. Split the dataset into training and testing sets.
6. Train a Linear Regression model using the training data.
7. Train a Ridge Regression model using RidgeCV with different alpha values.
8. Train a Lasso Regression model using LassoCV.
9. Predict the target values using all three models.
10. Calculate and print the MSE and R² score for each model.
11. Plot actual and predicted values, compare model coefficients, and print the best alpha values.
