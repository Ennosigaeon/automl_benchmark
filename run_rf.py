import datetime
import traceback

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from benchmark import OpenMLBenchmark

timeout = 3600  # in seconds
run_timeout = 360  # in seconds
jobs = 4


def main(bm: OpenMLBenchmark):
    X_train = SimpleImputer().fit_transform(bm.X_train)
    y_train = bm.y_train
    X_test = SimpleImputer().fit_transform(bm.X_test)
    y_test = bm.y_test

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    print('Misclassification rate', 1 - sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    for i in range(15):
        print('#######\nIteration {}\n#######'.format(i))
        print('Timeout: ', timeout)
        print('Run Timeout: ', run_timeout)

        task_ids = [15, 23, 24, 29, 3021, 41, 2079, 3543, 3560, 3561,
                    3904, 3946, 9955, 9985, 7592, 14969, 14968, 14967, 125920, 146606]
        for task in task_ids:
            print('Starting task {} at {}'.format(task, datetime.datetime.now().time()))

            try:
                bm = OpenMLBenchmark(task)
                main(bm)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                traceback.print_exc()
                print('Misclassification rate', 1)
