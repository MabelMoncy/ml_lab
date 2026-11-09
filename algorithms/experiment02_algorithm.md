# Experiment 2: Polynomial Regression

## Algorithm

1. Import the required libraries for data handling, plotting, regression, and evaluation.
2. Load the MPG dataset using Seaborn.
3. Fill the missing horsepower values with the median value.
4. Remove the `name` column and select `displacement` as the input and `mpg` as the target.
5. Split the data into training and testing sets.
6. Train a Linear Regression model using the training data.
7. Predict the test values and calculate MSE and R² for Linear Regression.
8. Create Polynomial Features for degrees 2, 3, and 5.
9. Train a Linear Regression model for each polynomial degree and calculate its MSE and R².
10. Print the performance results for the linear and polynomial models.
11. Plot the data points and all regression lines, then save the graph as `exp2_output.png`.
