from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from pre_processing import load_data, split_data, standardize_data

X, Y, outlier_indices = load_data()
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

param_distributions = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if 'penalty' == 'elasticnet' else None,
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga'],
    'max_iter': list(range(100, 800, 100))
}

log_reg = LogisticRegression()

random_search = RandomizedSearchCV(estimator=log_reg, param_distributions=param_distributions, cv=5, scoring='accuracy',
                                   error_score='raise',
                                   n_jobs=-1, n_iter=50)

random_search.fit(X_train_scaled, Y_train)

print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)
