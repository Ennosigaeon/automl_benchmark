import numpy as np
from tpot import TPOTClassifier

from benchmark import OpenMLBenchmark

timeout = 60
run_timeout = 30
jobs = 4


def main(bm: OpenMLBenchmark):
    X_train = np.concatenate((bm.X_valid, bm.X_train))
    y_train = np.concatenate((bm.y_valid, bm.y_train))
    X_test = bm.X_test
    y_test = bm.y_test

    pipeline_optimizer = TPOTClassifier(
        max_time_mins=timeout / 60,
        max_eval_time_mins=run_timeout / 60,
        scoring='accuracy',
        n_jobs=jobs,
        verbosity=3
    )
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.fitted_pipeline_)
    print('Misclassification Rate', 1 - pipeline_optimizer.score(X_test, y_test))


if __name__ == '__main__':
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    task_ids = [22, 37, 2079, 3543, 3899, 3913, 3917, 9950, 9980, 14966]
    for task in task_ids:
        print('Starting task {}'.format(task))
        bm = OpenMLBenchmark(task)

        main(bm)
