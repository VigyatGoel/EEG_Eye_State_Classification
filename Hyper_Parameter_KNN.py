from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

from pre_processing import load_data, split_data, standardize_data

X, Y, outlier_indices = load_data()
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

param_distributions = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],

}

knn = KNeighborsClassifier()

random_search = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_distributions,
    cv=5,
    scoring='accuracy',
    error_score='raise',
    n_jobs=-1, n_iter=50
)

random_search.fit(X_train_scaled, Y_train)

print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)
