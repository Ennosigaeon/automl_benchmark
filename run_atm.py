import datetime
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from benchmark import OpenMLBenchmark

timeout = 60  # in minutes
# run_timeout = 30
jobs = 4


def main(bm: OpenMLBenchmark):
    X_train = bm.X_train
    y_train = bm.y_train
    X_test = bm.X_test
    y_test = bm.y_test

    headers = bm.column_names + ['class']
    train = np.c_[X_train, y_train]
    test = np.c_[X_test, y_test]

    os.mkdir('/tmp/atm/{}'.format(bm.task_id))
    train_path = '/tmp/atm/{}/train.csv'.format(bm.task_id)
    test_path = '/tmp/atm/{}/test.csv'.format(bm.task_id)
    pd.DataFrame(train, columns=headers).to_csv(train_path, index=None)
    pd.DataFrame(test, columns=headers).to_csv(test_path, index=None)

    sql_path = '{}/assets/atm_sql.yaml'.format(os.getcwd())
    cmd = 'atm enter_data --sql-config {sql} --train-path {train_path} --test-path {test_path}' \
          ' --budget-type walltime --budget {budget} --metric accuracy --name {name}' \
        .format(sql=sql_path, train_path=train_path, test_path=test_path, budget=timeout, name=bm.task_id)
    subprocess.call(cmd, shell=True)

    cmd = 'atm worker --no-save --sql-config {}'.format(sql_path)

    procs = [subprocess.Popen(cmd, shell=True) for i in range(jobs)]

    start = time.time()
    while time.time() - start <= 1.05 * timeout:
        if any(p.poll() is None for p in procs):
            time.sleep(10)
        else:
            break
    else:
        print('Grace period exceed. Killing workers.')
        for p in procs:
            p.terminate()


if __name__ == '__main__':
    for i in range(4):
        print('#######\nIteration {}\n#######'.format(i))

        try:
            shutil.rmtree('/tmp/atm/')
        except OSError as e:
            pass
        os.mkdir('/tmp/atm/')

        print('Timeout: ', timeout)

        task_ids = [15, 23, 24, 29, 3021, 41, 2079, 3543, 3560, 3561,
                    3904, 3946, 9955, 9985, 7592, 14969, 14968, 14967, 125920, 146606]
        for task in task_ids:
            print('Starting task {} at {}'.format(task, datetime.datetime.now().time()))
            bm = OpenMLBenchmark(task)

            main(bm)

###########
# Get results via
#
# SELECT ds.name, 1 - max(cs.test_judgment_metric) as 'Misclassification rate' FROM classifiers cs
# JOIN dataruns dr ON cs.datarun_id = dr.id
# JOIN datasets ds ON dr.dataset_id = ds.id
# GROUP BY cs.datarun_id
