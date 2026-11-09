
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

# Load dataset
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target
n = len(X)

# Bootstrapping
acc, f1 = [], []

for i in range(100):
    idx = np.random.choice(n, n, replace=True)
    oob = list(set(range(n)) - set(idx))

    if len(oob) == 0:
        continue

    model = LogisticRegression(max_iter=200)
    model.fit(X[idx], y[idx])
    pred = model.predict(X[oob])

    acc.append(accuracy_score(y[oob], pred))
    f1.append(f1_score(y[oob], pred, average='macro'))

print("Bootstrapping Results:")
print("Mean Accuracy:", np.mean(acc))
print("Mean F1 Score:", np.mean(f1))
print("Std Accuracy:", np.std(acc))

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc, f1 = [], []

for train, test in kf.split(X):
    model = LogisticRegression(max_iter=200)
    model.fit(X[train], y[train])
    pred = model.predict(X[test])

    acc.append(accuracy_score(y[test], pred))
    f1.append(f1_score(y[test], pred, average='macro'))

print("\nK-Fold Cross Validation Results:")
print("Mean Accuracy:", np.mean(acc))
print("Mean F1 Score:", np.mean(f1))
print("Std Accuracy:", np.std(acc))
