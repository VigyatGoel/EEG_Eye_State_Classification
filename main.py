from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pre_processing import load_data, split_data, standardize_data


def train_model(X_train, Y_train, model):
    model.fit(X_train, Y_train)
    return model


def evaluate_model(X_test, Y_test, model):
    Y_predicted = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predicted)
    f1 = f1_score(Y_test, Y_predicted)
    print("Accuracy is: ", accuracy * 100, "%")
    print("F1 score is: ", f1)


def main():
    X, Y, outlier_indices = load_data()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 42)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    models = [
        LogisticRegression(),
        RandomForestClassifier(
            criterion="entropy",
            max_depth=50,
            max_features="sqrt",
            n_estimators=100,
            random_state=42
        ),
        GradientBoostingClassifier(
            subsample=0.5,
            n_estimators=500,
            max_features='log2',
            max_depth=20,
            loss='exponential',
            learning_rate=0.1,
            criterion='friedman_mse'
        ),
        SVC(
            kernel='rbf',
            gamma=1,
            C=100,
            random_state=42
        ),
        KNeighborsClassifier(
            weights='distance',
            n_neighbors=3,
            metric='minkowski'
        ),
        DecisionTreeClassifier(
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_depth=20,
            criterion='entropy'
        ),
        Perceptron(
            penalty='elasticnet',
            max_iter=1000,
            fit_intercept=True,
            eta0=0.1,
            alpha=0.1
        ),
        MLPClassifier(
            max_iter=1000,
            solver='adam',
            learning_rate='adaptive',
            hidden_layer_sizes=(100, 50),
            alpha=0.01,
            activation='tanh',
            random_state=42,
        )
    ]

    for model in models:
        scores = cross_val_score(model, X_train_scaled, Y_train, cv=5)
        print(f"Cross-validation scores for {model.__class__.__name__}: {scores}")
        print(f"Average cross-validation score for {model.__class__.__name__}: {scores.mean()}")

        trained_model = train_model(X_train_scaled, Y_train, model)
        print(f"Evaluating model: {model.__class__.__name__}")
        evaluate_model(X_test_scaled, Y_test, trained_model)


if __name__ == "__main__":
    main()
