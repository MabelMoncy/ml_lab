# EXPERIMENT 10: SVM Kernel Comparison on Fashion MNIST

from keras.datasets import fashion_mnist
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load and reduce dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]

# Normalize and flatten
X_train = X_train.reshape(len(X_train), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0

# Train and test SVM models
for kernel in ['linear', 'poly', 'rbf']:
    model = svm.SVC(kernel=kernel, degree=3)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    name = 'Polynomial' if kernel == 'poly' else kernel.capitalize()
    print(name, "SVM Accuracy:", accuracy_score(y_test, pred))
