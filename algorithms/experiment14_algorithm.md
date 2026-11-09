# Experiment 14: Bootstrapping and K-Fold Cross Validation

## Algorithm

1. Import the required libraries for dataset loading, Logistic Regression, scaling, and evaluation.
2. Load the built-in Iris dataset and separate the features (X) and target (y).
3. Standardize the feature values using StandardScaler.
4. Set the number of samples and create empty lists to store bootstrap scores.
5. Repeat the bootstrapping process 100 times by selecting samples with replacement.
6. Find the Out-of-Bag (OOB) samples that are not selected in the bootstrap sample.
7. If OOB samples are available, train a Logistic Regression model using the bootstrap sample.
8. Predict the OOB samples and store the accuracy and macro F1 score.
9. Print the mean accuracy, mean F1 score, and standard deviation of bootstrap accuracy.
10. Create a 5-fold cross-validation method with shuffling and a fixed random state.
11. Train and test a Logistic Regression model for each fold, then store accuracy and macro F1 score.
12. Print the mean accuracy, mean F1 score, and standard deviation of the K-Fold accuracy.
