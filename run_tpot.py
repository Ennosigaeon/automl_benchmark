import datetime

import traceback
from tpot import TPOTClassifier

from benchmark import OpenMLBenchmark

timeout = 3600  # in seconds
run_timeout = 360
jobs = 2


def main(bm: OpenMLBenchmark):
    X_train = bm.X_train
    y_train = bm.y_train
    X_test = bm.X_test
    y_test = bm.y_test

    pipeline_optimizer = TPOTClassifier(
        max_time_mins=timeout / 60,
        max_eval_time_mins=run_timeout / 60,
        scoring='accuracy',
        n_jobs=jobs,
        verbosity=1
    )
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.fitted_pipeline_)
    print('Misclassification Rate', 1 - pipeline_optimizer.score(X_test, y_test))


if __name__ == '__main__':
    for i in range(10):
        print('#######\nIteration {}\n#######'.format(i))
        print('Timeout: ', timeout)
        print('Run Timeout: ', run_timeout)

        task_ids = [3, 12, 31, 53, 3917, 3945, 7593, 9952, 9977, 9981, 10101, 14965, 34539, 146195, 146212, 146818,
                    146821, 146822, 146825, 167119, 167120, 168329, 168330, 168331, 168332, 168335, 168337, 168338,
                    168868, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189356]
        for task in task_ids:
            try:
                print('Starting task {} at {}'.format(task, datetime.datetime.now().time()))
                bm = OpenMLBenchmark(task)
                main(bm)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                traceback.print_exc()
                print('Misclassification rate', 1)
