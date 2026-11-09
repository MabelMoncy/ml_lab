# Experiment 10: Performance Comparison of SVM Kernels on Fashion-MNIST

## Algorithm

1. Import the required libraries for Fashion-MNIST, SVM, and accuracy calculation.
2. Load the Fashion-MNIST training and testing datasets.
3. Select the first 5,000 training images and the first 1,000 testing images.
4. Flatten each image into a one-dimensional feature array.
5. Normalize the pixel values by dividing them by 255.
6. Select the Linear, Polynomial, and RBF kernels one by one.
7. Create an SVM model using the selected kernel.
8. Train the model using the training images and labels.
9. Predict the labels of the testing images.
10. Calculate and print the accuracy of each SVM model.
11. Compare the accuracy results of the three kernels.
