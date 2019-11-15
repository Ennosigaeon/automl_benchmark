import os
import pickle

import numpy as np
import openml
from atm import Model
from atm.utilities import base_64_to_object

from adapter import run_atm, run_auto_sklearn, run_h2o, run_hpsklearn, run_tpot
from benchmark import OpenMLBenchmark
from evaluation.base import MongoPersistence

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


if __name__ == '__main__':
    # merge_cash_results()
    # print_data_set_stats()
    # load_atm_results()
    load_file_results('/mnt/c/local/results/AutoML Benchmark', 'h2o')
