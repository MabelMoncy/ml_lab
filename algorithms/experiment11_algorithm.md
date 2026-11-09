# Experiment 11: Multilayer Feed-Forward Network on Wine Quality Dataset

## Algorithm

1. Import the required libraries for data handling, preprocessing, neural networks, and plotting.
2. Read the `WineQT.csv` dataset.
3. Separate the input features from the quality column.
4. Encode the quality values into numbers and convert them into categorical form.
5. Split the data into training and testing sets.
6. Apply StandardScaler to the training and testing features.
7. Create a function to build an MLP model with the given hidden-layer sizes.
8. Set up three architectures: Simple, Medium, and Deep.
9. For each architecture, build and train the model using ReLU, Adam, and 50 epochs.
10. Evaluate each trained model on the test data and print its accuracy.
11. Plot and save the accuracy comparison of the three models.
