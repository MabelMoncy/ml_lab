# EXPERIMENT 2: Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
df = sns.load_dataset('mpg')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
df = df.drop('name', axis=1)
X, y = df[['displacement']], df['mpg']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Linear Regression
lin = LinearRegression().fit(X_train, y_train)
pred = lin.predict(X_test)
print("Linear Regression:")
print("MSE:", round(mean_squared_error(y_test, pred), 4))
print("R²:", round(r2_score(y_test, pred), 4))

# Polynomial Regression
degrees, models = [2, 3, 5], {}
print("\nPolynomial Regression:")
print("Degree\tMSE\t\tR2")
print("-----------------------------------")
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    model = LinearRegression().fit(poly.fit_transform(X_train), y_train)
    p = model.predict(poly.transform(X_test))
    print(f"{d}\t{mean_squared_error(y_test, p):.4f}\t{r2_score(y_test, p):.4f}")
    models[d] = (model, poly)

# Graph
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', s=20, label='Data')
r = pd.DataFrame(np.linspace(X.displacement.min(), X.displacement.max(), 100),
                 columns=['displacement'])
plt.plot(r, lin.predict(r), color='red', label='Linear')
for d, c in zip(degrees, ['yellow', 'green', 'orange']):
    model, poly = models[d]
    plt.plot(r, model.predict(poly.transform(r)), color=c, label=f'Degree {d}')
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial vs Linear Regression (Error-Free)")
plt.legend()
plt.savefig("exp2_output.png")
