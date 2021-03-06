import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def skip(id: int) -> bool:
    failed = []
    return id in failed


def setup():
    pass


def main(fold, dummy: bool, score: bool = True) -> float:
    setup()
    X_train, y_train, X_test, y_test = fold
    X_train = SimpleImputer().fit_transform(X_train)
    X_test = SimpleImputer().fit_transform(X_test)

    estimator = DummyClassifier() if dummy else RandomForestClassifier()
    estimator.fit(X_train, y_train)

    if score:
        predictions = estimator.predict(X_test)
        return 1 - sklearn.metrics.accuracy_score(y_test, predictions)
    else:
        # predictions = estimator.predict_proba(X_test)
        # return sklearn.metrics.log_loss(y_test, predictions), estimator
        predictions = estimator.predict_proba(X_test)
        return sklearn.metrics.roc_auc_score(y_test, predictions[:, 1]), estimator
