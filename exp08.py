# EXPERIMENT 8: Decision Tree using ID3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Load and preprocess dataset
df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Customer-level data
customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum', 'UnitPrice': 'mean',
    'TotalSpend': 'sum', 'Country': 'first'
}).reset_index()

# Create target
threshold = customer_df['TotalSpend'].median()
customer_df['HighValue'] = (customer_df['TotalSpend'] > threshold).astype(int)

# Encode country
customer_df['Country'] = LabelEncoder().fit_transform(customer_df['Country'])

# Features and target
X = customer_df[['Quantity', 'UnitPrice', 'Country']]
y = customer_df['HighValue']

# Train ID3 Decision Tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model.fit(X, y)

# Visualize and save tree
plt.figure(figsize=(20, 8))
plot_tree(model, feature_names=X.columns,
          class_names=['Low', 'High'], filled=True)
plt.title("Customer Segmentation using Decision Tree (ID3)")
plt.savefig("exp8_decision_tree_id3.png", bbox_inches="tight")

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))
