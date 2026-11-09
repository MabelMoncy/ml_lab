# EXPERIMENT 12: Neural Network using Different Activation Functions on MNIST

import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create model
def create_model(activation):
    model = Sequential([
        Input(shape=(28, 28)), Flatten(),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train models with different activation functions
activations = ['sigmoid', 'tanh', 'relu']
results, histories = {}, {}

for act in activations:
    print(f"\nTraining with {act} activation")
    model = create_model(act)
    start = time.time()
    history = model.fit(X_train, y_train, epochs=5, batch_size=128,
                        validation_split=0.2, verbose=1)
    end = time.time()
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[act] = {'accuracy': acc, 'time': end - start}
    histories[act] = history

# Print accuracy and training time
for act in results:
    print(f"{act} -> Accuracy: {results[act]['accuracy']:.4f}, Time: {results[act]['time']:.2f}s")

# Validation Accuracy Graph
plt.figure()
for act in activations:
    plt.plot(histories[act].history['val_accuracy'], label=act)
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("exp12_validation_accuracy_comparison.png", bbox_inches="tight")

# Validation Loss Graph
plt.figure()
for act in activations:
    plt.plot(histories[act].history['val_loss'], label=act)
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("exp12_validation_loss_comparison.png", bbox_inches="tight")
