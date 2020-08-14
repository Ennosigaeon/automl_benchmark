import os
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import sklearn
from atm import ATM, Model
from sklearn.pipeline import Pipeline
from typing import List

from benchmark import OpenMLBenchmark


def skip(id: int) -> bool:
    failed = []
    return id in failed


def setup():
    try:
        shutil.rmtree('/tmp/atm/')
    except OSError as e:
        pass
    os.mkdir('/tmp/atm/')


def main(fold, bm: OpenMLBenchmark, timeout: int, jobs: int, score: bool = True) -> float:
    setup()
    X_train, y_train, X_test, y_test = fold

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

    procs = [subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid) for i in range(jobs)]

    start = time.time()
    while time.time() - start <= 1.05 * timeout:
        if any(p.poll() is None for p in procs):
            time.sleep(10)
        else:
            break
    else:
        print('Grace period exceed. Killing workers.')
        for p in procs:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.terminate()

    # Only used to mark datarun as finished. Should terminate immediately
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

    atm = ATM()
    dataruns = atm.db.get_dataruns(ignore_complete=False, ignore_running=True, ignore_pending=True)
    datarun = max(dataruns, key=lambda d: d.id)
    best = datarun.get_best_classifier()

    hp = atm.db.get_hyperpartition(best.hyperpartition_id)
    model = Model(hp.method, best.hyperparameter_values, '', '')
    model._make_pipeline()
    pipeline = model.pipeline

    pipeline.fit(X_train, y_train)

    if score:
        predictions = pipeline.predict(X_test)
        return 1 - sklearn.metrics.accuracy_score(y_test, predictions)
    else:
        predictions = pipeline.predict_proba(X_test)
        return sklearn.metrics.roc_auc_score(y_test, predictions[:, 1]), pipeline


# noinspection PyUnresolvedReferences
def load_model(input: str):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    pipeline: Pipeline = eval(input)
    return pipeline


# noinspection PyUnresolvedReferences
def load_pipeline(input: str) -> List[List[str]]:
    pipeline: Pipeline = load_model(input)
    res = []
    for s in pipeline.steps:
        res.append(type(s[1]).__name__)

    return [res]

###########
# Get results via
#
# SELECT ds.name, 1 - max(cs.test_judgment_metric) as 'Misclassification rate' FROM classifiers cs
# JOIN dataruns dr ON cs.datarun_id = dr.id
# JOIN datasets ds ON dr.dataset_id = ds.id
# GROUP BY cs.datarun_id
# ORDER BY CAST(name AS INTEGER)
