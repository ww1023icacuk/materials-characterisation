# note to self: models only work when X is pre-scaled
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class LogisticModel:

    def __init__(self, C=1.0, max_iter=1000, penalty="l2", solver="lbfgs"):
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty
        self.solver = solver

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            penalty=self.penalty,
            solver=self.solver
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # returns probability for each class
        return self.model.predict_proba(X)


class KNNModel:

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # KNN supports predict_proba
        return self.model.predict_proba(X)


class SVMModel:

    def __init__(self, kernel="rbf", C=1.0, gamma="scale", probability=True):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # only works because we set probability=True
        return self.model.predict_proba(X)


class RFModel:

    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def feature_importances(self):
        # for Dataset 1 feature importance plots
        return self.model.feature_importances_


def get_default_models():

    return {
        "logistic": LogisticModel(),
        "knn": KNNModel(n_neighbors=5),
        "svm_rbf": SVMModel(kernel="rbf", C=1.0, gamma="scale"),
        "random_forest": RFModel(n_estimators=200, max_depth=None),
    }