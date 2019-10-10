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
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    task_ids = [3, 12, 31, 53, 3917, 3945, 7593, 9952, 9977, 9981, 10101, 14965, 34539, 146195, 146212, 146818,
                146821, 146822, 146825, 167119, 167120, 168329, 168330, 168331, 168332, 168335,
                # 168337, 168338,
                168868, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189356]
    for task in task_ids:
        print('#######\nStarting task {}\n#######'.format(task))
        for i in range(10):
            print('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
            try:
                bm = OpenMLBenchmark(task)
                main(bm)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                traceback.print_exc()
                print('Misclassification rate', 1)
