# Experiment 8: Decision Tree Classification using ID3

## Algorithm

1. Import the required libraries for data handling, encoding, Decision Tree, and plotting.
2. Read the `Online Retail.xlsx` dataset.
3. Remove rows with missing CustomerID values.
4. Remove rows with zero or negative Quantity and UnitPrice values.
5. Calculate TotalSpend by multiplying Quantity and UnitPrice.
6. Group the data by CustomerID and calculate total Quantity, average UnitPrice, total spending, and Country.
7. Find the median TotalSpend and create the HighValue target column.
8. Convert the Country values into numbers using LabelEncoder.
9. Select Quantity, UnitPrice, and Country as input features, and HighValue as the target.
10. Train a Decision Tree model using entropy and a maximum depth of 4.
11. Plot and save the decision tree, then calculate and print feature importance.
