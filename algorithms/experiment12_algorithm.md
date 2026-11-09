# Experiment 12: Neural Network using Different Activation Functions

## Algorithm

1. Import the required libraries for MNIST, neural networks, timing, and plotting.
2. Load the MNIST training and testing datasets.
3. Normalize the image pixel values by dividing them by 255.
4. Convert the target labels into categorical form.
5. Create a function to build a neural network with Flatten, Dense 128, Dense 64, and Dense 10 layers.
6. Compile the model using Adam optimizer and categorical cross-entropy loss.
7. Select the activation functions Sigmoid, Tanh, and ReLU.
8. For each activation function, create and train the model for 5 epochs with batch size 128.
9. Record the training time and evaluate the model on the test data.
10. Print the accuracy and training time for each activation function.
11. Plot and save the validation accuracy and validation loss graphs.
