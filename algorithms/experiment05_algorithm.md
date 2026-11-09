# Experiment 5: Logistic Regression using MLE and MAP (L1 and L2)

## Algorithm

1. Import the required libraries for data handling, Logistic Regression, scaling, evaluation, and plotting.
2. Load the built-in Breast Cancer dataset from scikit-learn.
3. Split the input features and target values into training and testing sets.
4. Apply StandardScaler to the training and testing features.
5. Create three Logistic Regression models: MLE, L2 MAP, and L1 MAP.
6. Train all three models using the scaled training data.
7. Predict the test results using each model.
8. Calculate and print the accuracy of all three models.
9. Print the classification report for each model.
10. Display the first ten feature coefficients for comparison.
11. Plot the absolute coefficient values of the three models and save the graph.
12. Count and print the number of zero coefficients in the L1 model.
