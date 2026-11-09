# EXPERIMENT 4: Logistic Regression with and without Feature Scaling

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Dataset and split
data = pd.read_csv("diabetes.csv")
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# WITHOUT FEATURE SCALING
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]

print("Without Scaling:")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix (Without Scaling)")
plt.savefig("exp4_without_scaling_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, prob)
auc = roc_auc_score(y_test, prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Without Scaling)")
plt.legend()
plt.savefig("exp4_without_scaling_roc_curve.png")

# WITH FEATURE SCALING
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
pred, prob = model.predict(X_test_s), model.predict_proba(X_test_s)[:, 1]

print("\nWith Scaling:")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Confusion Matrix (With Scaling)")
plt.savefig("exp4_with_scaling_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, prob)
auc = roc_auc_score(y_test, prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (With Scaling)")
plt.legend()
plt.savefig("exp4_with_scaling_roc_curve.png")
