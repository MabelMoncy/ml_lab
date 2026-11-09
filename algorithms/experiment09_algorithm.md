# Experiment 9: Linear SVM Classification on Iris Dataset

## Algorithm

1. Import NumPy, Matplotlib, and the required SVM and dataset modules.
2. Load the built-in Iris dataset.
3. Select the first two features: Sepal Length and Sepal Width.
4. Convert the target into two classes: Setosa and Non-Setosa.
5. Create a Linear SVM model with C = 1.
6. Train the model using the selected features and binary target values.
7. Plot the data points using the two selected features.
8. Create a grid of points over the plot area and calculate the model's decision values.
9. Draw the decision boundary and the two margin lines.
10. Mark the support vectors on the graph.
11. Add axis labels and a title, then save the graph as `exp9_svm_decision_boundary_output.png`.
