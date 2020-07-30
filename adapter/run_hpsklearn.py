import time
import traceback
from typing import List

import hpsklearn
import hyperopt
import sklearn
from sklearn.pipeline import Pipeline


def skip(id: int) -> bool:
    failed = []
    return id in failed


def setup():
    pass


def main(fold, timeout: int, run_timeout: int):
    def run():
        setup()
        X_train, y_train, X_test, y_test = fold
        estimator = hpsklearn.HyperoptEstimator(
            preprocessing=hpsklearn.components.any_preprocessing('pp'),
            classifier=hpsklearn.components.any_classifier('clf'),
            algo=hyperopt.tpe.suggest,
            trial_timeout=run_timeout,
            max_evals=-1,
            timeout=timeout,
            seed=int(time.time())
        )
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        print(estimator.best_model())
        return 1 - sklearn.metrics.accuracy_score(y_test, predictions)

    for j in range(100):
        print('Attempt {}...'.format(j))
        try:
            return run()
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
    else:
        traceback.print_exc()
        return 1


# noinspection PyUnresolvedReferences
def load_model(input: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Normalizer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.decomposition import PCA
    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from numpy import nan

    dict = eval(input)

    assert dict['ex_preprocs'] == ()
    steps = []
    if dict['preprocs'] != ():
        steps.append(('preprocs', dict['preprocs'][0]))
    steps.append(('learner', dict['learner']))
    pipeline = Pipeline(steps=steps)
    return pipeline


def load_pipeline(input: str) -> List[List[str]]:
    pipeline = load_model(input)

    res = []
    for s in pipeline.steps:
        res.append(type(s[1]).__name__)
    return [res]
