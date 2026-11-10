# EXPERIMENT 3: Ridge and Lasso Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Models
lin = LinearRegression().fit(X_train, y_train)
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X_train, y_train)
lasso = LassoCV(cv=5, max_iter=10000).fit(X_train, y_train)
preds = [lin.predict(X_test), ridge.predict(X_test), lasso.predict(X_test)]
names = ["Linear", "Ridge", "Lasso"]

# Evaluation
print("\nModel Performance:")
print("-------------")
print("Model\t\tMSE\t\tR2")
print(".-")
for name, p in zip(names, preds):
    print(f"{name}\t\t{mean_squared_error(y_test, p):.4f}\t{r2_score(y_test, p):.4f}")

# Graph 1: Actual vs Predicted
plt.figure()
for name, p in zip(names, preds):
    plt.scatter(y_test, p, label=name)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Comparison")
plt.legend()
plt.savefig("exp3_actual_vs_predicted.png")

# Graph 2: Coefficient Comparison
coef = pd.DataFrame({'Feature': X.columns, 'Linear': lin.coef_,
                     'Ridge': ridge.coef_, 'Lasso': lasso.coef_})
print("\nCoefficients:")
print(coef)
coef.set_index('Feature').plot(kind='bar')
plt.title("Coefficient Comparison (Shrinkage Effect)")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.savefig("exp3_coefficient_comparison.png")

# Best Alpha Values
print("\nBest Alpha Values:")
print("Ridge Alpha:", ridge.alpha_)
print("Lasso Alpha:", lasso.alpha_)
