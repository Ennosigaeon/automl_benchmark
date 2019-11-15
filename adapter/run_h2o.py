import random
import shutil
import tempfile
from typing import List

import h2o
import numpy as np
import pandas as pd
import sklearn
from h2o.automl import H2OAutoML
from h2o.estimators import H2OXGBoostEstimator, H2OGeneralizedLinearEstimator, H2OGradientBoostingEstimator, \
    H2ODeepLearningEstimator, H2ORandomForestEstimator

from benchmark import OpenMLBenchmark


def skip(id: int) -> bool:
    failed = [167125]
    return id in failed


def setup():
    pass


def main(bm: OpenMLBenchmark, timeout: int, run_timeout: int, jobs: int) -> float:
    try:
        log_dir = tempfile.mkdtemp()
        avg_score = 0
        for fold in bm.folds:
            setup()
            X_train, y_train, X_test, y_test = fold

            h2o.init(nthreads=jobs, port=str(60000 + random.randrange(0, 5000)), ice_root=log_dir)
            h2o.no_progress()

            train = np.append(X_train, np.atleast_2d(y_train).T, axis=1)
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
            avg_score += 1 - sklearn.metrics.accuracy_score(y_test, predictions['predict'].as_data_frame())
        return avg_score / len(bm.folds)
    finally:
        h2o.cluster().shutdown()
        shutil.rmtree(log_dir)


def load_pipeline(input: str) -> List[List[str]]:
    def _map_algo(algo: str):
        if algo == 'xgboost':
            return H2OXGBoostEstimator()
        elif algo == 'gbm':
            return H2OGradientBoostingEstimator()
        elif algo == 'glm':
            return H2OGeneralizedLinearEstimator()
        elif algo == 'deeplearning':
            return H2ODeepLearningEstimator()
        elif algo == 'drf' or algo == 'xrt':
            return H2ORandomForestEstimator()
        else:
            raise ValueError(algo)

    res = []

    prefix = input.split(' ')[0]
    if prefix == 'stackedensemble':
        models = eval(input[len(prefix) + 2:-2])['base_models']
        for idx, m in enumerate(models):
            n = m['name'].split('_')[0].lower()
            res.append([type(_map_algo(n)).__name__])
    else:
        res.append([type(_map_algo(prefix)).__name__])

    for j in range(len(res)):
        for i in range(len(res[j])):
            n = res[j][i]

            if n == 'H2ORandomForestEstimator':
                n = 'RandomForestClassifier'
            if n == 'H2ODeepLearningEstimator':
                n = 'DeepLearningClassifier'
            if n == 'H2OXGBoostEstimator':
                n = 'XGBClassifier'
            if n == 'H2OGeneralizedLinearEstimator':
                n = 'GeneralizedLinearClassifier'
            if n == 'H2OGradientBoostingEstimator':
                n = 'GradientBoostingClassifier'

            res[j][i] = n

    return sorted(res)
