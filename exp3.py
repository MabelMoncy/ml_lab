
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin = LinearRegression().fit(X_train, y_train)
ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X_train, y_train)
lasso = LassoCV(cv=5, max_iter=10000).fit(X_train, y_train)

models = {
    "Linear": lin,
    "Ridge": ridge,
    "Lasso": lasso
}
pred = {name: model.predict(X_test) for name, model in models.items()}

print("\nModel Performance:")
print("Model\t\tMSE\t\tR2")
print(".-")
for name, y_pred in pred.items():
    print(f"{name}\t\t{mean_squared_error(y_test, y_pred):.4f}\t{r2_score(y_test, y_pred):.4f}")

# Actual vs Predicted
for name, y_pred in pred.items():
    plt.scatter(y_test, y_pred, label=name)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Comparison")
plt.legend()
plt.show()

# Coefficient Comparison
coef_df = pd.DataFrame({
    name: model.coef_ for name, model in models.items()
}, index=X.columns)

print("\nCoefficients:")
print(coef_df)

coef_df.plot(kind="bar")
plt.title("Coefficient Comparison (Shrinkage Effect)")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.show()

print("\nBest Alpha Values:")
print("Ridge Alpha:", ridge.alpha_)
print("Lasso Alpha:", lasso.alpha_)
