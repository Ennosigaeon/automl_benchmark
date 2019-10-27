import time
import traceback

import hpsklearn
import hyperopt
import sklearn

from benchmark import OpenMLBenchmark


def setup():
    pass


def main(bm: OpenMLBenchmark, timeout: int, run_timeout: int):
    def run():
        avg_score = 0
        for fold in bm.folds:
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
            avg_score += 1 - sklearn.metrics.accuracy_score(y_test, predictions)
        return avg_score / len(bm.folds)

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
