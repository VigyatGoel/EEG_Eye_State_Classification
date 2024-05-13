from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from pre_processing import load_data, split_data, standardize_data

X, Y, outlier_indices = load_data()
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

param_distributions = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'n_estimators': [300, 400, 500, 800],
    'subsample': [1.0, 0.5, 0.25, 0.1],
    'criterion': ['friedman_mse'],
    'max_depth': [3, 5, 10, 15, 20],
    'max_features': ['sqrt', 'log2', None]
}

gbc = GradientBoostingClassifier()

random_search = RandomizedSearchCV(estimator=gbc, param_distributions=param_distributions, cv=5, scoring='accuracy',
                                   error_score='raise',
                                   n_jobs=-1, n_iter=20)

random_search.fit(X_train_scaled, Y_train)

print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)
