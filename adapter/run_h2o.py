import random
import shutil
import tempfile

import h2o
import numpy as np
import pandas as pd
import sklearn
from h2o.automl import H2OAutoML

from benchmark import OpenMLBenchmark


def skip(id: int) -> bool:
    failed = [167125]
    return id in failed


def setup():
    pass


def main(bm: OpenMLBenchmark, timeout: int, run_timeout: int, jobs: int) -> float:
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
        score = 1 - sklearn.metrics.accuracy_score(y_test, predictions['predict'].as_data_frame())
        return score
    finally:
        h2o.cluster().shutdown()
        shutil.rmtree(log_dir)
