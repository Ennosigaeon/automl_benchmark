import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from benchmark import OpenMLBenchmark


def setup():
    pass


def main(bm: OpenMLBenchmark, dummy: bool) -> float:
    X_train = SimpleImputer().fit_transform(bm.X_train)
    y_train = bm.y_train
    X_test = SimpleImputer().fit_transform(bm.X_test)
    y_test = bm.y_test

    estimator = DummyClassifier() if dummy else RandomForestClassifier()
    estimator.fit(X_train, y_train)

    predictions = estimator.predict(X_test)
    score = 1 - sklearn.metrics.accuracy_score(y_test, predictions)
    return score
