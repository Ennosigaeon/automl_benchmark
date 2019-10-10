import datetime
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import traceback

from benchmark import OpenMLBenchmark

timeout = 3600  # in seconds
jobs = 2


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
        .format(sql=sql_path, train_path=train_path, test_path=test_path, budget=timeout // 60, name=bm.task_id)
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

    # Only used to mark datarun as finished. Should terminate immediately
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()


if __name__ == '__main__':
    print('Timeout: ', timeout)

    task_ids = [3, 12, 31, 53, 3917, 3945, 7593, 9952, 9977, 9981, 10101, 14965, 34539, 146195, 146212, 146818,
                146821, 146822, 146825, 167119, 167120, 168329, 168330, 168331, 168332, 168335, 168337, 168338,
                168868, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189356]
    for task in task_ids:
        print('#######\nStarting task {}\n#######'.format(task))
        for i in range(10):
            try:
                shutil.rmtree('/tmp/atm/')
            except OSError as e:
                pass
            os.mkdir('/tmp/atm/')

            try:
                print('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
                bm = OpenMLBenchmark(task)
                main(bm)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                traceback.print_exc()
                print('Misclassification rate', 1)

###########
# Get results via
#
# SELECT ds.name, 1 - max(cs.test_judgment_metric) as 'Misclassification rate' FROM classifiers cs
# JOIN dataruns dr ON cs.datarun_id = dr.id
# JOIN datasets ds ON dr.dataset_id = ds.id
# GROUP BY cs.datarun_id
# ORDER BY CAST(name AS INTEGER)
