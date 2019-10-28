import time
import traceback

import hpsklearn
import hyperopt
import sklearn

from benchmark import OpenMLBenchmark


def skip(id: int) -> bool:
    failed = []
    return id in failed


def setup():
    pass


def main(bm: OpenMLBenchmark, timeout: int, run_timeout: int):
    def run():
        X_train = bm.X_train
        y_train = bm.y_train
        X_test = bm.X_test
        y_test = bm.y_test
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
        score = 1 - sklearn.metrics.accuracy_score(y_test, predictions)
        return score

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
