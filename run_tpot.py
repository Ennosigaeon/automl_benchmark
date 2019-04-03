import datetime

from tpot import TPOTClassifier

from benchmark import OpenMLBenchmark

timeout = 3600
run_timeout = 360
jobs = 4


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
        verbosity=2
    )
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.fitted_pipeline_)
    print('Misclassification Rate', 1 - pipeline_optimizer.score(X_test, y_test))


if __name__ == '__main__':
    for i in range(4):
        print('#######\nIteration {}\n#######'.format(i))
        print('Timeout: ', timeout)
        print('Run Timeout: ', run_timeout)

        task_ids = [15, 23, 29, 3021, 41, 2079, 3560, 3561, 3904, 3946, 9955, 9985, 7592, 14969, 146606]
        task_ids = [24, 3543, 7948, 14967, 125920]
        for task in task_ids:
            print('Starting task {} at {}'.format(task, datetime.datetime.now().time()))
            bm = OpenMLBenchmark(task)

            main(bm)
