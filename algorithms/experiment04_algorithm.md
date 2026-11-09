# Experiment 4: Logistic Regression with and without Feature Scaling

## Algorithm

1. Import the required libraries for data handling, Logistic Regression, scaling, evaluation, and plotting.
2. Read the `diabetes.csv` dataset.
3. Separate the input features (X) and target values (y).
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression model using the original, unscaled training data.
6. Predict the test classes and probabilities, then calculate Accuracy, Precision, Recall, and F1 Score.
7. Display the confusion matrix and plot the ROC curve for the model without scaling.
8. Apply StandardScaler to the training and testing features.
9. Train another Logistic Regression model using the scaled training data.
10. Predict the test classes and probabilities, then calculate the same four scores.
11. Display the confusion matrix and plot the ROC curve for the model with scaling.
