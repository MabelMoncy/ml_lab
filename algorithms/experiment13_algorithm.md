# Experiment 13: K-Means Clustering on Digits Dataset

## Algorithm

1. Import the required libraries for dataset loading, scaling, K-Means, evaluation, and plotting.
2. Load the built-in Digits dataset from scikit-learn.
3. Select the feature matrix from the dataset.
4. Standardize the features using StandardScaler.
5. Set the number of clusters from 2 to 14.
6. For each K value, create a K-Means model using K-Means++ initialization.
7. Fit the model and get the cluster labels for all data points.
8. Calculate and store the inertia value for each K.
9. Calculate and store the silhouette score for each K.
10. Plot K against inertia and save the Elbow Method graph.
11. Plot K against silhouette score and save the Silhouette Score graph.
12. Print the inertia and silhouette score for every K value.
