import matplotlib.pyplot as plt

from pre_processing import load_data, split_data, standardize_data

X, Y, outlier_indices = load_data()
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

plt.figure(figsize=(10, 5))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
plt.title('Scatter plot of feature 0 vs feature 1')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
