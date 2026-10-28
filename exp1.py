# Experiment 1: Simple Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/sample_data/housing.csv")
X = df['total_rooms'].head(150).values.reshape(-1, 1)
y = df['median_house_value'].head(150).values.reshape(-1, 1)

mx, my = np.mean(X), np.mean(y)
b1 = np.sum((X-mx)*(y-my)) / np.sum((X-mx)**2)
b0 = my - b1*mx

y_pred = b0 + b1*X
mse = np.mean((y-y_pred)**2)
r2 = 1 - np.sum((y-y_pred)**2)/np.sum((y-my)**2)

print("Slope (b1):", b1)
print("Intercept (b0):", b0)
print("MSE:", mse)
print("R²:", r2)

plt.scatter(X, y, color="blue", s=20)
plt.plot(X, y_pred, color="red")
plt.xlabel("Total Rooms")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression (150 Samples)")
plt.show()
