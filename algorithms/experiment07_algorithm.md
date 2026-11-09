# Experiment 7: K-Nearest Neighbors Classification on Fashion-MNIST

## Algorithm

1. Import the required libraries for loading data, KNN classification, scaling, and evaluation.
2. Load the Fashion-MNIST training and testing datasets.
3. Flatten each 28 × 28 image into a one-dimensional feature array.
4. Normalize the pixel values by dividing them by 255.
5. Set the K values as 1, 3, and 5.
6. Select the first 10,000 training images and their labels.
7. Select the first 2,000 testing images and their labels.
8. For each K value, create a KNN classifier and train it using the selected training data.
9. Predict the labels of the selected testing images.
10. Calculate and print the accuracy and classification report for each K value.
