import os
import pickle
import traceback
from typing import Dict, List, Tuple

import numpy as np
import openml
import sklearn
from sklearn.ensemble import VotingClassifier

from adapter.run_h2o import _createFrame
from benchmark import OpenMLBenchmark, create_estimator
from evaluation.visualization import plot_cash_overfitting, plot_framework_overfitting

all_tasks = [3, 6, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 24, 28, 29, 31, 32, 36, 37, 41, 43, 45, 49, 53, 58, 219,
             2074, 2079, 3021, 3022, 3481, 3485, 3492, 3493, 3494, 3510, 3512, 3543, 3549, 3560, 3561, 3567, 3573, 3889,
             3891, 3896, 3899, 3902, 3903, 3904, 3913, 3917, 3918, 3945, 3946, 3948, 3954, 7592, 7593, 9910, 9914, 9946,
             9950, 9952, 9954, 9955, 9956, 9957, 9960, 9964, 9967, 9968, 9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981,
             9983, 9985, 9986, 10093, 10101, 14952, 14954, 14964, 14965, 14966, 14967, 14968, 14969, 14970,
             34537, 34538, 34539, 125920, 125921, 125922, 125923, 146195, 146212, 146606, 146607, 146800, 146817,
             146818, 146819, 146820, 146821, 146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140,
             167141, 168329, 168330, 168331, 168332, 168335, 168337, 168338, 168868, 168908, 168909, 168910, 168911,
             168912, 189354, 189355, 189356]
all_datasets = [3, 6, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 24, 28, 29, 31, 32, 36, 37, 42, 44, 46, 50, 54, 60, 151,
                182, 188, 38, 307, 300, 312, 333, 334, 335, 375, 377, 451, 458, 469, 470, 478, 554, 1036, 1038, 1043,
                1046, 1049, 1050, 1053, 1063, 1067, 1068, 1111, 1112, 1114, 1120, 1590, 1596, 4134, 1570, 1510, 1515,
                1489, 1491, 1492, 1493, 1494, 1497, 1501, 1504, 1505, 1479, 1480, 1485, 1486, 1487, 1466, 1467, 1468,
                1471, 1475, 1476, 1462, 1464, 4534, 6332, 1459, 1461, 4134, 23380, 6332, 4538, 1478, 4534, 4550, 4135,
                23381, 40496, 40499, 40509, 40668, 40685, 23512, 40536, 40966, 40982, 40981, 40994, 40983, 40975, 40984,
                40979, 40996, 41027, 23517, 40923, 40927, 40978, 40670, 40701, 41169, 41168, 41166, 41165, 41150, 41159,
                41161, 41138, 41142, 41163, 41164, 41143, 41146, 1169, 41167, 41147]

cash_tasks = [3, 6, 11, 12, 14, 16, 18, 20, 21, 22, 23, 28, 31, 32, 36, 37, 43, 45, 49, 53, 58, 219, 2074, 3022, 3481,
              3485, 3492, 3493, 3494, 3510, 3512, 3549, 3560, 3567, 3573, 3889, 3891, 3896, 3899, 3902, 3903, 3913,
              3917, 3918, 3954, 7593, 9910, 9914, 9946, 9950, 9952, 9954, 9955, 9956, 9957, 9960, 9964, 9967, 9968,
              9970, 9971, 9976, 9977, 9978, 9979, 9980, 9981, 9983, 9985, 9986, 10093, 10101, 14952, 14964, 14965,
              14966, 14969, 14970, 34537, 34539, 125921, 125922, 125923, 146195, 146212, 146817, 146818, 146819, 146820,
              146821, 146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141, 168329, 168330,
              168331, 168332, 168335, 168337, 168338, 168908, 168909, 168910, 168911, 168912, 189354, 189355]
cash_datasets = [3, 6, 11, 12, 14, 16, 18, 20, 21, 22, 23, 28, 31, 32, 36, 37, 44, 46, 50, 54, 60, 151, 182, 307, 300,
                 312, 333, 334, 335, 375, 377, 458, 469, 478, 554, 1036, 1038, 1043, 1046, 1049, 1050, 1063, 1067, 1068,
                 1120, 1596, 4134, 1570, 1510, 1515, 1489, 1491, 1492, 1493, 1494, 1497, 1501, 1504, 1505, 1479, 1480,
                 1485, 1486, 1487, 1466, 1467, 1468, 1471, 1475, 1476, 1462, 1464, 4534, 1459, 1461, 4134, 4538, 1478,
                 4534, 4135, 40496, 40499, 40509, 40668, 40685, 40982, 40981, 40994, 40983, 40975, 40984, 40979, 40996,
                 41027, 23517, 40923, 40927, 40978, 40670, 40701, 41169, 41168, 41166, 41165, 41150, 41159, 41161,
                 41142, 41163, 41164, 41143, 41146, 1169, 41167]

framework_tasks = [3, 12, 15, 23, 24, 29, 31, 41, 53, 2079, 3021, 3543, 3560, 3561, 3904, 3917, 3945, 3946, 3948, 7592,
                   7593, 9910, 9952, 9955, 9977, 9981, 9985, 10101, 14952, 14954, 14965, 14967, 14968, 14969, 34538,
                   34539, 125920, 146195, 146212, 146606, 146607, 146800, 146817, 146818, 146819, 146820, 146821,
                   146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 167141, 168329, 168330,
                   168331, 168332, 168335, 168337, 168338, 168868, 168908, 168909, 168910, 168911, 168912, 189354,
                   189355, 189356]
framework_datasets = [3, 12, 15, 23, 24, 29, 31, 42, 54, 188, 38, 451, 469, 470, 1053, 1067, 1111, 1112, 1114, 1590,
                      1596, 4134, 1489, 1492, 1486, 1468, 1475, 1464, 4534, 6332, 1461, 23380, 6332, 4538, 4550, 4135,
                      23381, 40668, 40685, 23512, 40536, 40966, 40982, 40981, 40994, 40983, 40975, 40984, 40979, 40996,
                      41027, 23517, 40923, 40927, 40978, 40670, 40701, 41169, 41168, 41166, 41165, 41150, 41159, 41161,
                      41138, 41142, 41163, 41164, 41143, 41146, 1169, 41167, 41147]


class Dataset:

    def __init__(self, task_id=None, name=None, dataset_id=None, NumberOfInstances=None, NumberOfClasses=None,
                 NumberOfMissingValues=None, NumberOfInstancesWithMissingValues=None, NumberOfNumericFeatures=None,
                 NumberOfSymbolicFeatures=None, MinorityClassPercentage=None):
        self.task_id = task_id
        self.name = name
        self.dataset_id = dataset_id

        self.NumberOfInstances = NumberOfInstances
        self.NumberOfClasses = NumberOfClasses
        self.NumberOfMissingValues = NumberOfMissingValues
        self.NumberOfInstancesWithMissingValues = NumberOfInstancesWithMissingValues
        self.NumberOfNumericFeatures = NumberOfNumericFeatures
        self.NumberOfSymbolicFeatures = NumberOfSymbolicFeatures
        self.MinorityClassPercentage = MinorityClassPercentage


def merge_cash_results():
    from benchmark import OpenMLBenchmark
    from evaluation.base import MongoPersistence
    tasks = [146212]
    for id in tasks:
        print(id)
        bm = OpenMLBenchmark(id, load=False)

        p1 = MongoPersistence('localhost', db='benchmarks')
        new_results = MongoPersistence('localhost', db='foo2').load_all(bm)

        for res in new_results:
            p1.store_new_run(res)
            for s in res.solvers:
                p1.store_results(res, s)


def print_data_set_stats():
    # noinspection PyUnresolvedReferences
    from evaluation.scripts import Dataset

    ls = []
    for id in all_tasks:
        print(id)
        task = openml.tasks.get_task(id)
        dataset = openml.datasets.get_dataset(dataset_id=task.dataset_id)

        ds = Dataset()
        ds.task_id = task.task_id
        ds.name = dataset.name
        ds.dataset_id = dataset.dataset_id

        ds.NumberOfInstances = dataset.qualities['NumberOfInstances']
        ds.NumberOfClasses = dataset.qualities['NumberOfClasses']
        ds.NumberOfMissingValues = dataset.qualities['NumberOfMissingValues']
        ds.NumberOfInstancesWithMissingValues = dataset.qualities['NumberOfInstancesWithMissingValues']
        ds.NumberOfNumericFeatures = dataset.qualities['NumberOfNumericFeatures']
        ds.NumberOfSymbolicFeatures = dataset.qualities['NumberOfSymbolicFeatures']
        ds.MinorityClassPercentage = dataset.qualities['MinorityClassPercentage']

        ls.append(ds)

    # Print data set ids ordered by task ids
    tmp = [d.dataset_id for d in ls]
    print(tmp)

    ls = sorted(ls, key=lambda ds: ds.dataset_id)
    map = {}
    for ds in ls:
        print(
            '{:15} & ({}) & {:8} & {:8} & {:8} & {:8} & {:8} & {:8} & {:02.2f} \\\\ % {}'.format(
                ds.name[:15].replace('_', '\\_'), ds.dataset_id,
                int(ds.NumberOfClasses),
                int(ds.NumberOfInstances),
                int(ds.NumberOfNumericFeatures),
                int(ds.NumberOfSymbolicFeatures),
                int(ds.NumberOfMissingValues),
                int(ds.NumberOfInstancesWithMissingValues),
                ds.MinorityClassPercentage,
                ds.task_id
            ))
        map[ds.task_id] = ds

    with open('assets/ds.pkl', 'wb') as f:
        pickle.dump(map, f)


def load_atm_results():
    from atm.utilities import base_64_to_object
    from atm import Model

    def load(c):
        result = {}
        for task, res, hyper, cat, method, const in c.execute('''
    SELECT ds.name,
           1 - max(cs.test_judgment_metric) as 'Misclassification rate',
           cs.hyperparameter_values_64,
           h.categorical_hyperparameters_64,
           h.method,
           h.constant_hyperparameters_64
    FROM classifiers cs
             JOIN hyperpartitions h on cs.hyperpartition_id = h.id
             JOIN dataruns dr ON cs.datarun_id = dr.id
             JOIN datasets ds ON dr.dataset_id = ds.id
    GROUP BY cs.datarun_id
    ORDER BY CAST(name AS INTEGER)
    '''):
            if task not in result:
                result[task] = [[], []]

            if res is None:
                continue
            result[task][0].append(res)

            hyper = base_64_to_object(hyper)
            cat = dict(base_64_to_object(cat))
            const = dict(base_64_to_object(const))
            params = {**hyper, **cat, **const}

            model = Model(method, params, None, None)
            model._make_pipeline()
            pipeline = model.pipeline

            result[task][1].append(str(pipeline))
        return result

    import sqlite3

    base_dir = '/mnt/c/local/results/AutoML Benchmark/atm'
    dbs = ['assets/atm.db', 'assets/atm-0.db', 'assets/atm-1.db', 'assets/atm-2.db', 'assets/atm-4.db',
           'assets/atm-5.db', 'assets/atm-5-0.db', 'assets/atm-5-1.db', 'assets/atm-5-2.db', 'assets/atm-5-3.db']
    result = {}
    for db in dbs:
        con = sqlite3.connect(db)
        res = load(con.cursor())

        for key, value in res.items():
            if key not in result:
                result[key] = [[], []]

            result[key][0].extend(value[0])
            result[key][1].extend(value[1])

    for key, value in result.items():
        name = os.path.join(base_dir, '{}.txt'.format(key))
        with open(name, 'w+') as f:
            f.write('{}\n'.format(value[0]))
            f.writelines('\n'.join(value[1]))


def load_file_results(base_dir: str, algorithm: str):
    from adapter import run_atm, run_auto_sklearn, run_h2o, run_hpsklearn, run_tpot

    def reject_outliers(data):
        return data[abs(data - np.mean(data)) < 0.2]

    np.set_printoptions(linewidth=1000)
    pipelines = []
    for task in framework_tasks:
        name = os.path.join(base_dir, algorithm, '{}.txt'.format(task))
        pipelines.append([])

        if os.path.exists(name):
            with open(name, 'r') as f:
                values = np.array(eval(f.readline()[:-1]))
                print('{},  # {}'.format(np.array2string(values, separator=', '), task))
                # print(len(values))

                filtered_values = reject_outliers(values)
                assert values.shape == filtered_values.shape

                line = f.readline()
                while line:
                    if algorithm == 'atm':
                        pipeline = run_atm.load_pipeline(line)
                    elif algorithm == 'random' or algorithm == 'auto-sklearn':
                        pipeline = run_auto_sklearn.load_pipeline(line)
                    elif algorithm == 'h2o':
                        pipeline = run_h2o.load_pipeline(line)
                    elif algorithm == 'hpsklearn':
                        pipeline = run_hpsklearn.load_pipeline(line)
                    elif algorithm == 'tpot':
                        pipeline = run_tpot.load_pipeline(line)
                    else:
                        raise ValueError('Unknown algorithm {}'.format(algorithm))

                    pipelines += pipeline
                    line = f.readline()
        else:
            print('{},  # {}'.format([1], task))

    pipelines = [p for p in pipelines if len(p) > 0]
    print('\n\n')
    print(pipelines)


def calculate_cash_overfitting():
    with open('assets/cash_configs.pkl', 'rb') as f:
        cash_configs: Dict[int, Dict[str, List[Tuple[str, List[Dict]]]]] = pickle.load(f)

    if os.path.exists('assets/overfitting_cash.pkl'):
        with open('assets/overfitting_cash.pkl', 'rb') as f:
            results: Dict[str, List[float]] = pickle.load(f)
    else:
        results = {}

    for task, models in cash_configs.items():
        print('Processing {}'.format(task))
        data = OpenMLBenchmark(task)

        for model, solvers in models.items():
            for solver, params in solvers:
                if solver not in results:
                    results[solver] = []

                for configuration, score in params:
                    for fold in data.folds:
                        try:
                            X_train, y_train, X_test, y_test = fold

                            clf = create_estimator(configuration)
                            clf = clf.fit(X_train, y_train)

                            learn_score = clf.score(X_train, y_train)
                            test_score = clf.score(X_test, y_test)
                            results[solver].append(learn_score - test_score)
                        except Exception as ex:
                            traceback.print_exc()
                            print(ex)

        with open('assets/overfitting_cash.pkl', 'wb') as f:
            pickle.dump(results, f)
    print(results)


def calculate_framework_overfitting(base_dir: str):
    import h2o
    # Preprocessing for ATM
    # =([a-z]\w*)                       =>      ='$1'
    #  _\w+=\w+,?                       =>      EMPTY STRING
    #
    # Preprocessing for TPOT
    # <function copy at 0x\w+>          =>      copy
    # <function f_classif at 0x\w+>     =>      f_classif
    # <class ('\w+')>                   =>      $1

    from adapter import run_atm, run_auto_sklearn, run_h2o, run_hpsklearn, run_tpot

    np.set_printoptions(linewidth=1000)

    if os.path.exists('assets/overfitting_frameworks.pkl'):
        with open('assets/overfitting_frameworks.pkl', 'rb') as f:
            results: Dict[str, List[float]] = pickle.load(f)
    else:
        results = {}

    for task in framework_tasks:
        # 7594, 146195
        if task < 168339:
            continue

        print('Processing {}'.format(task))

        h2o.init(nthreads=4, max_mem_size=4 * 4, port=str(60000), ice_root='/tmp')
        h2o.no_progress()

        # for algorithm in ['random', 'auto-sklearn', 'tpot', 'atm', 'hpsklearn', 'h2o']:
        for algorithm in ['h2o']:
            if algorithm not in results:
                results[algorithm] = []
            print(algorithm)

            name = os.path.join(base_dir, algorithm, '{}.txt'.format(task))
            if not os.path.exists(name):
                print('{} not found'.format(name))
                continue

            data = OpenMLBenchmark(task)

            with open(name, 'r') as f:
                # First line contains performances
                f.readline()

                line = f.readline()
                while line:
                    for fold in data.folds:
                        X_train, y_train, X_test, y_test = fold
                        print(line)

                        try:
                            if algorithm == 'atm':
                                pipeline = [(1.0, run_atm.load_model(line))]
                            elif algorithm == 'random' or algorithm == 'auto-sklearn':
                                pipeline = run_auto_sklearn.load_model(line)
                            elif algorithm == 'hpsklearn':
                                pipeline = [(1.0, run_hpsklearn.load_model(line))]
                            elif algorithm == 'tpot':
                                pipeline = [(1.0, run_tpot.load_model(line))]
                            elif algorithm == 'h2o':
                                pipeline = run_h2o.load_model(line)
                                train = _createFrame(X_train, y_train)
                                learning = _createFrame(X_train)
                                test = _createFrame(X_test)

                                for i in range(len(data.categorical)):
                                    if data.categorical[i]:
                                        train[i] = train[i].asfactor()
                                        learning[i] = learning[i].asfactor()
                                        test[i] = test[i].asfactor()
                                train['class'] = train['class'].asfactor()

                                pipeline.train(y='class', training_frame=train)
                                learning_pred = pipeline.predict(learning)
                                test_pred = pipeline.predict(test)

                                learn_score = sklearn.metrics.accuracy_score(y_train,
                                                                             learning_pred['predict'].as_data_frame())
                                test_score = sklearn.metrics.accuracy_score(y_test,
                                                                            test_pred['predict'].as_data_frame())
                                diff = learn_score - test_score
                                results[algorithm].append(diff)
                                continue
                            else:
                                raise ValueError('Unknown algorithm {}'.format(algorithm))

                            estimators = []
                            weights = []
                            for i, p in enumerate(pipeline):
                                estimators.append((str(i), p[1]))
                                weights.append(p[0])
                            clf = VotingClassifier(estimators=estimators, weights=weights)
                            clf.fit(X_train, y_train)

                            learn_score = clf.score(X_train, y_train)
                            test_score = clf.score(X_test, y_test)
                            results[algorithm].append(learn_score - test_score)
                        except Exception:
                            print(name)
                            traceback.print_exc()
                    line = f.readline()
        with open('assets/overfitting_frameworks.pkl', 'wb') as f:
            pickle.dump(results, f)
        h2o.shutdown()


def comparison_human():
    print('Otto')
    test_score = {
        'atm': [2.91148, 2.93632, 2.65594, 1.34024, 3.53879, 1.84785, 3.67209, 1.87478, 1.10286],
        'auto_sklearn': [0.58090, 0.51243, 0.58211, 0.55919, 0.50697, 0.52733, 0.60403, 0.56736, 0.49913, 0.56866],
        'h2o': [0.50598, 0.49506, 0.49449, 0.49427, 0.49585, 0.49700, 0.49398, 0.49524, 0.49464],
        'hpsklearn': [0.61789, 0.59867, 0.67445, 0.63893, 0.63374, 0.46736, 0.59702, 0.52271, 0.53230],
        'random': [0.85949, 0.85029, 0.94609, 0.86576, 0.86814, 0.94839, 0.88069, 0.93987, 0.95185, 0.88371],
        'tpot': [0.85517, 1.17658, 1.66502, 1.25393, 0.89691, 0.95225, 0.95153, 0.87094, 0.83533]
    }
    validation_score = {
        'atm': [0.9348088, 0.92656238, 0.81354877, 0.65033479, 0.65033479, 0.0581878, 1.6229545, 0.04607008, 0.86187573,
                0.92656238],
        'auto_sklearn': [0.58540871, 0.5233486, 0.57930406, 0.55954705, 0.51190175, 0.53013835, 0.61564406, 0.56739354,
                         0.50095343, 0.57322175],
        'h2o': [0.80925447, 0.81167852, 0.81178625, 0.81388709, 0.81469511, 0.80925447, 0.81367162, 0.81157078,
                0.81017022],
        'hpsklearn': [0.80925447, 0.81167852, 0.81178625, 0.81388709, 0.81469511, 0.80925447, 0.81367162, 0.81157078,
                      0.81017022],
        'random': [0.92235668, 0.92815748, 0.92608496, 0.8150354, 0.78372758, 0.92862843, 0.8748143, 0.91344764,
                   0.84032975, 0.93759275],
        'tpot': [0.72183732, 0.95732825, 0.26069617, 0.85932076, 0.99390921, 0.8926285, 0.85998579, 0.91666936,
                 0.83355711]
    }
    for algo in test_score.keys():
        print(algo)
        print('\\({:.5f}\\)'.format(np.array(validation_score[algo]).mean()), end='\t& ')
        print('\\({:.5f}\\)'.format(np.array(test_score[algo]).mean()))

    print('Santander')
    test_score = {
        'atm': [0.76910, 0.68058, 0.50000, 0.65937, 0.76716, 0.53705, 0.77794, 0.79657, 0.79493, 0.62162],
        'auto_sklearn': [0.83367, 0.83319, 0.83118, 0.83565, 0.83408, 0.83367, 0.82690, 0.83591, 0.83522, 0.83517],
        'h2o': [0.83686, 0.83920, 0.83975, 0.83854, 0.83838, 0.83752, 0.83829, 0.83752, 0.83843, 0.83842],
        'hpsklearn': [0.62383, 0.82659, 0.51228, 0.56846, 0.82068, 0.50153, 0.50222, 0.79401, 0.76574, 0.53392],
        'random': [0.82338, 0.81508, 0.81575, 0.83335, 0.82152, 0.81810, 0.82595, 0.82912, 0.82769, 0.83273],
        'tpot': [0.83197, 0.83152, 0.82893, 0.82905, 0.83336, 0.83260, 0.83209, 0.83033, 0.83100, 0.82910]
    }
    validation_score = {
        'atm': [0.7633770991899569, 0.6660188096642717, 0.5, 0.6219140424287214, 0.7682546990889543, 0.6429712309250604,
                0.5401348467601457, 0.7875460506894172, 0.7975720207247147, 0.7843287893046778],
        'auto_sklearn': [0.8337843124470236, 0.8357795161097868, 0.8349650381064628, 0.8406354920609883,
                         0.8424907206918998, 0.8276812861944557, 0.8442426994473258, 0.8353404221080115,
                         0.8222642129325447, 0.8375380926769591],
        'h2o': [0.8341856835610995, 0.8482967965540849, 0.8391147962759583, 0.8255381911542628, 0.8392471801384983,
                0.8218117072846803, 0.8308815369066873, 0.8437632958718368, 0.8368556559846618, 0.8208760129271573],
        'hpsklearn': [0.6123186721294669, 0.8358891589782879, 0.5789571688755933, 0.8280722527953254,
                      0.5004239378661391,
                      0.5006385962456732, 0.8050012898135842, 0.7609504150399212, 0.5330202502811654],
        'random': [0.8288269646829022, 0.8144617601085264, 0.8310956247122092, 0.8314770531425896, 0.8336662360631153,
                   0.814090865563922, 0.8331325803833143, 0.8456798773020915, 0.8188332141045025, 0.8293519726956391],
        'tpot': [0.8355902870426116, 0.8336183625254824, 0.8360300732943023, 0.8331269669487492, 0.8298068634682588,
                 0.818127618346291, 0.8416381986007053, 0.8342959111221397, 0.840466644307385, 0.8252490381601473]
    }
    for algo in test_score.keys():
        print(algo)
        print('\\({:.5f}\\)'.format(np.array(validation_score[algo]).mean()), end='\t& ')
        print('\\({:.5f}\\)'.format(np.array(test_score[algo]).mean()))


if __name__ == '__main__':
    # merge_cash_results()
    # print_data_set_stats()
    # load_atm_results()
    load_file_results('/mnt/c/local/results/AutoML Benchmark', 'h2o')
    calculate_framework_overfitting('/mnt/c/Users/usimzoller/Dropbox/phd/publication/Benchmark and Survey of Automated Machine Learning Frameworks - JAIR/results/AutoML Benchmark/')
    calculate_cash_overfitting()
    plot_cash_overfitting()
    plot_framework_overfitting()
    comparison_human()
