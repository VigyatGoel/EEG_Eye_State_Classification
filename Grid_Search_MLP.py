from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

from pre_processing import load_data, split_data, standardize_data

X, Y, outlier_indices = load_data()
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)


param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}


mlp = MLPClassifier(max_iter=4000)

random_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_distributions, cv=5, scoring='accuracy',
                                   error_score='raise',
                                   n_jobs=-1, n_iter=50)

random_search.fit(X_train_scaled, Y_train)

print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)
