import datetime
import itertools
import sys
import time
import traceback

import hpsklearn
import humanfriendly
import hyperopt
import sklearn

from benchmark import OpenMLBenchmark

timeout = 3600  # in seconds
run_timeout = 600  # in seconds


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
        max_evals=-1,
        timeout=timeout,
        seed=int(start)
    )
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    print(estimator.best_model())
    print('Misclassification rate', 1 - sklearn.metrics.accuracy_score(y_test, predictions))
    print('Duration', humanfriendly.format_timespan(time.time() - start))


if __name__ == '__main__':
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    task_ids = [[3, 12, 31, 53], [3917, 3945, 7593, 9952], [9977, 9981, 10101, 14965], [34539, 146195, 146212, 146818],
                [146821, 146822, 146825, 167119], [167120, 168329, 168330, 168331], [168332, 168335, 168337, 168338],
                [168868, 168908, 168909, 168910], [168911, 168912, 189354, 189355, 189356]]

    idx = None
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        print('Using chunk {}/8'.format(idx))
        task_ids = task_ids[idx]
    else:
        task_ids = list(itertools.chain.from_iterable(task_ids))

    for task in task_ids:
        print('#######\nStarting task {}\n#######'.format(task))
        for i in range(10):
            print('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
            bm = OpenMLBenchmark(task)

            for j in range(100):
                print('Attempt {}...'.format(j))
                try:
                    main(bm)
                    break
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
            else:
                traceback.print_exc()
                print('Misclassification rate', 1)
