import datetime
import traceback
import random

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
    print('Misclassification rate', 1 - sklearn.metrics.accuracy_score(y_test, predictions['predict'].as_data_frame()))


if __name__ == '__main__':
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    h2o.init(nthreads=jobs, port=str(60000 + random.randrange(0, 5000)))
    h2o.no_progress()

    task_ids = [15]
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
    h2o.cluster().shutdown()
