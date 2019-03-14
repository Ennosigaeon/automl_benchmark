import time

import hpsklearn
import humanfriendly
import hyperopt
import sklearn

from benchmark import OpenMLBenchmark

max_evals = 325
run_timeout = 60


def main(bm: OpenMLBenchmark):
    start = time.time()
    X_train = bm.X_train
    y_train = bm.y_train
    X_test = bm.X_test
    y_test = bm.y_test
    estimator = hpsklearn.HyperoptEstimator(
        preprocessing=hpsklearn.components.any_preprocessing('pp'),
        classifier=hpsklearn.components.any_classifier('clf'),
        algo=hyperopt.tpe.suggest,
        trial_timeout=run_timeout,
        max_evals=max_evals,
        seed=int(start)
    )
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    print('Misclassification rate', 1 - sklearn.metrics.accuracy_score(y_test, predictions))
    print('Duration', humanfriendly.format_timespan(time.time() - start))


if __name__ == '__main__':
    print('Max Evals: ', max_evals)
    print('Run Timeout: ', run_timeout)

    task_ids = [22, 37, 2079, 3543, 3899, 3913, 3917, 9950, 9980]
    for task in task_ids:
        print('Starting task {}'.format(task))
        bm = OpenMLBenchmark(task)

        try:
            main(bm)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            print(e)

