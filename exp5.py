# EXPERIMENT 5: Logistic Regression using MLE and MAP (L1 & L2)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset and split
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train MLE, L2 MAP and L1 MAP models
models = {
    "MLE": LogisticRegression(penalty=None, max_iter=5000),
    "L2 MAP": LogisticRegression(penalty='l2', C=1.0, max_iter=5000),
    "L1 MAP": LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
}
for model in models.values():
    model.fit(X_train, y_train)

# Predictions
pred = {name: model.predict(X_test) for name, model in models.items()}

# Accuracy
print("\n=== Accuracy ===")
for name in models:
    print(name + " Accuracy:", accuracy_score(y_test, pred[name]))

# Classification reports
for name in models:
    print(f"\n=== Classification Report: {name} ===")
    print(classification_report(y_test, pred[name]))

# Coefficient comparison
coeffs = pd.DataFrame({
    "Feature": data.feature_names,
    "MLE": models["MLE"].coef_[0],
    "L2": models["L2 MAP"].coef_[0],
    "L1": models["L1 MAP"].coef_[0]
})
print("\n=== Sample Coefficients ===")
print(coeffs.head(10))

# Visualization
plt.figure()
for name, label in zip(models, ["MLE", "L2 (MAP)", "L1 (MAP)"]):
    plt.plot(np.abs(models[name].coef_[0]), label=label)
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Magnitude")
plt.title("Coefficient Comparison: MLE vs MAP")
plt.legend()
plt.grid()
plt.savefig("exp5_coefficient_comparison.png")

# L1 sparsity check
num_zero = np.sum(models["L1 MAP"].coef_[0] == 0)
print("\nNumber of zero coefficients in L1 (feature selection):", num_zero)
