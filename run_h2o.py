import datetime
import itertools
import random
import shutil
import sys
import tempfile
import traceback

import h2o
import numpy as np
import pandas as pd
import sklearn
from h2o.automl import H2OAutoML

from benchmark import OpenMLBenchmark

timeout = 3600  # in seconds
run_timeout = 600  # in seconds
jobs = 4


def main(bm: OpenMLBenchmark):
    try:
        log_dir = tempfile.mkdtemp()

        h2o.init(nthreads=jobs, port=str(60000 + random.randrange(0, 5000)), ice_root=log_dir)
        h2o.no_progress()
        X_test = bm.X_test
        y_test = bm.y_test

        train = np.append(bm.X_train, np.atleast_2d(bm.y_train).T, axis=1)
        df_train = pd.DataFrame(data=train[0:, 0:],
                                index=[i for i in range(train.shape[0])],
                                columns=['f' + str(i) for i in range(train.shape[1] - 1)] + ['class'])
        df_test = pd.DataFrame(data=X_test,
                               index=[i for i in range(X_test.shape[0])],
                               columns=['f' + str(i) for i in range(X_test.shape[1])])
        train = h2o.H2OFrame(df_train)
        test = h2o.H2OFrame(df_test)

        for i in range(len(bm.categorical)):
            if bm.categorical[i]:
                train[i] = train[i].asfactor()
                test[i] = test[i].asfactor()
        train['class'] = train['class'].asfactor()

        aml = H2OAutoML(max_runtime_secs=timeout,
                        max_runtime_secs_per_model=run_timeout)
        aml.train(y='class', training_frame=train)

        predictions = aml.leader.predict(test)
        params = aml.leader.get_params()
        del params['model_id']
        del params['training_frame']
        del params['validation_frame']

        for key in params.keys():
            params[key] = params[key]['actual_value']

        print(aml.leader.algo, '(', params, ')')
        print('Misclassification rate',
              1 - sklearn.metrics.accuracy_score(y_test, predictions['predict'].as_data_frame()))
    finally:
        h2o.cluster().shutdown()
        shutil.rmtree(log_dir)


if __name__ == '__main__':
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    task_ids = [
        [3, 12, 15, 23, 24, 29, 31, 41, 53, 2079],
        [3021, 3543, 3560, 3561, 3904, 3917, 3945, 3946, 7592, 7593],
        [9952, 9955, 9977, 9981, 9985, 10101, 14965, 14967, 14968, 14969],
        [34539, 125920, 146195, 146212, 146606, 146818, 146821, 146822, 146825, 167119],
        [167120, 168329, 168330, 168331, 168332, 168335, 168337, 168338, 168868],
        [168908, 168909, 168910, 168911, 168912, 189354, 189355, 189356]]

    idx = None
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        print('Using chunk {}/6'.format(idx))
        task_ids = task_ids[idx]
    else:
        task_ids = list(itertools.chain.from_iterable(task_ids))

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
