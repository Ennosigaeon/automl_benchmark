from typing import List

import sklearn
from sklearn.pipeline import Pipeline
from tpot import TPOTClassifier


def skip(id: int) -> bool:
    failed = []
    return id in failed


def setup():
    pass


def main(fold, timeout: int, run_timeout: int, jobs: int, score: bool = True):
    setup()
    X_train, y_train, X_test, y_test = fold

    pipeline_optimizer = TPOTClassifier(
        max_time_mins=timeout / 60,
        max_eval_time_mins=run_timeout / 60,
        scoring='roc_auc',
        n_jobs=jobs,
        verbosity=1
    )
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.fitted_pipeline_)
    if score:
        predictions = pipeline_optimizer.predict(X_test)
        return 1 - sklearn.metrics.accuracy_score(y_test, predictions)
    else:
        predictions = pipeline_optimizer.predict_proba(X_test)
        return sklearn.metrics.roc_auc_score(y_test, predictions[:, 1]), pipeline_optimizer


# noinspection PyUnresolvedReferences
def load_model(input: str):
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import PolynomialFeatures, MaxAbsScaler, StandardScaler, MinMaxScaler
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFwe, f_classif
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import Normalizer, RobustScaler, FunctionTransformer, Binarizer
    from tpot.builtins import OneHotEncoder, StackingEstimator
    from tpot.builtins import ZeroCount
    from xgboost import XGBClassifier
    from copy import copy

    pipeline: Pipeline = eval(input)
    return pipeline


# noinspection PyUnresolvedReferences
def load_pipeline(input: str) -> List[List[str]]:
    from sklearn.pipeline import FeatureUnion
    from tpot.builtins import OneHotEncoder, StackingEstimator
    pipeline: Pipeline = load_model(input)

    res = []

    def _map_algo(algo):
        if isinstance(algo, StackingEstimator):
            _map_algo(algo.estimator)
        elif isinstance(algo, FeatureUnion):
            assert len(algo.transformer_list) == 2
            for t in sorted(algo.transformer_list):
                _map_algo(t[1])
        else:
            res.append(type(algo).__name__)

    for s in pipeline.steps:
        _map_algo(s[1])

    return [res]
