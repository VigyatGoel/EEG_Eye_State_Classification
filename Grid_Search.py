from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from pre_processing import load_data, split_data, standardize_data

X, Y, outlier_indices = load_data()
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

param_grid = {
    'n_estimators': [200],
    'max_features': ['sqrt'],
    'max_depth': [40, 50, 60],
    'criterion': ['gini', 'entropy', "log_loss"]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', error_score='raise',
                           n_jobs=-1)

grid_search.fit(X_train_scaled, Y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
