import os
from unittest import TestCase

import numpy as np
import scipy.stats
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from hyperopt import hp

from config import MetaConfigCollection, ConfigurationSpace, TpotConverter, HyperoptConverter, RandomSearchConverter, \
    GridSearchConverter
from config.converter import ConfigSpaceConverter


class SaneEqualityArray(np.ndarray):
    def __eq__(self, other):
        return (self.shape == other.shape and
                np.allclose(self, other))


class SaneEqualityDist(scipy.stats.distributions.uniform_gen):
    def __eq__(self, other):
        return (self.a == other.kwds['loc'] and self.b == other.kwds['scale'])


class TestConfigSpaceConverter(TestCase):

    def setUp(self):
        self.config = MetaConfigCollection.from_json(os.path.join(os.path.dirname(__file__), 'config.json'))

    def test_cs_convert(self):
        instance = ConfigSpaceConverter()

        actual = instance.convert(self.config)
        expected = self.__get_expected_cs()
        self.assertEqual(expected, actual)

    def test_tpot_convert(self):
        instance = TpotConverter()

        actual = instance.convert(self.config)
        expected = self.__get_expected_tpot()
        self.assertEqual(expected, actual)

    def test_hyperopt_convert(self):
        instance = HyperoptConverter()

        actual = instance.convert(self.config)
        expected = self.__get_expected_hp()

        self.assertEqual(str(expected), str(actual))

    def test_random_search_convert(self):
        instance = RandomSearchConverter()

        actual = instance.convert(self.config)
        expected = self.__get_expected_random_search()
        self.assertEqual(expected, actual)

    def test_grid_search_convert(self):
        instance = GridSearchConverter()

        actual = instance.convert(self.config)
        expected = self.__get_expected_grid_search()
        self.assertEqual(expected, actual)

    @staticmethod
    def __get_expected_cs():
        cs = ConfigurationSpace()
        kernel = CategoricalHyperparameter('kernel', ['linear', 'rbf', 'poly', 'sigmoid'], default_value='poly')
        cs.add_hyperparameter(kernel)
        C = UniformFloatHyperparameter('C', 0.001, 1000.0, default_value=1.0)
        shrinking = CategoricalHyperparameter('shrinking', [True, False], default_value=True)
        cs.add_hyperparameters([C, shrinking])
        degree = UniformIntegerHyperparameter('degree', 1, 5, default_value=3)  # Only used by kernel poly
        coef0 = UniformFloatHyperparameter('coef0', 0.0, 10.0, default_value=0.0)  # poly, sigmoid
        cs.add_hyperparameters([degree, coef0])
        use_degree = InCondition(child=degree, parent=kernel, values=['poly'])
        use_coef0 = InCondition(child=coef0, parent=kernel, values=['poly', 'sigmoid'])
        cs.add_conditions([use_degree, use_coef0])
        gamma = CategoricalHyperparameter('gamma', ['auto', 'value'], default_value='auto')  # only rbf, poly, sigmoid
        gamma_value = UniformFloatHyperparameter('gamma_value', 0.0001, 8, default_value=1)
        cs.add_hyperparameters([gamma, gamma_value])
        cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=['value']))
        cs.add_condition(InCondition(child=gamma, parent=kernel, values=['rbf', 'poly', 'sigmoid']))

        return {'sklearn.svm.SVC': cs}

    @staticmethod
    def __get_expected_tpot():
        SaneEqualityArray((2,), buffer=np.array([10.0, 1.0]))

        return {
            'sklearn.svm.SVC': {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': SaneEqualityArray((10,), buffer=np.array(
                    [1.000000e-03, 1.000009e+02, 2.000008e+02, 3.000007e+02, 4.000006e+02, 5.000005e+02, 6.000004e+02,
                     7.000003e+02, 8.000002e+02, 9.000001e+02])),
                'shrinking': [True, False],
                'degree': range(1, 5),
                'gamma': ['auto', 'value'],
                'gamma_value': SaneEqualityArray((10,), buffer=np.array(
                    [1.00000e-04, 8.00090e-01, 1.60008e+00, 2.40007e+00, 3.20006e+00, 4.00005e+00,
                     4.80004e+00, 5.60003e+00, 6.40002e+00, 7.20001e+00])),

                'coef0': SaneEqualityArray((10,), buffer=np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))
            }
        }

    @staticmethod
    def __get_expected_hp():
        expected_hp_space = hp.choice('estimator_type', [
            {
                'type': 'sklearn.svm.SVC',
                'C': hp.uniform('sklearn.svm.SVC_C', 0.001, 1000.0),
                'kernel': hp.choice('sklearn.svm.SVC_kernel', [
                    {'option': 'linear'},
                    {
                        'option': 'rbf',
                        'gamma': hp.choice('rbf_gamma', [
                            {'option': 'auto'},
                            {'option': hp.uniform('value_gamma_value', 0.0001, 8)}
                        ])
                    },
                    {
                        'option': 'poly',
                        'gamma': hp.choice('poly_gamma', [
                            {'option': 'auto'},
                            {'option': hp.uniform('value_gamma_value', 0.0001, 8)}
                        ]),
                        'degree': hp.quniform('poly_degree', 1, 5, 1),
                        'coef0': hp.uniform('poly_coef0', 0.0, 10.0)
                    },
                    {
                        'option': 'sigmoid',
                        'gamma': hp.choice('sigmoid_gamma', [
                            {'option': 'auto'},
                            {'option': hp.uniform('value_gamma_value', 0.0001, 8)}
                        ]),
                        'coef0': hp.uniform('sigmoid_coef0', 0.0, 10.0)
                    }
                ]),
                'shrinking': hp.choice('sklearn.svm.SVC_shrinking', [
                    {'option': True}, {'option': False}])
            }
        ])
        return expected_hp_space

    @staticmethod
    def __get_expected_random_search():
        return {
            'sklearn.svm.SVC': {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': SaneEqualityDist(a=0.001, b=1000),
                'shrinking': [True, False],
                'degree': range(1, 5),
                'gamma': ['auto', SaneEqualityDist(a=0.0001, b=8)],
                'coef0': SaneEqualityDist(a=0.0, b=10)
            }
        }

    @staticmethod
    def __get_expected_grid_search():
        return {
            'sklearn.svm.SVC': {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': SaneEqualityArray((10,), buffer=np.array(
                    [1.00000e-03, 1.11112e+02, 2.22223e+02, 3.33334e+02, 4.44445e+02, 5.55556e+02, 6.66667e+02,
                     7.77778e+02, 8.88889e+02, 1.00000e+03])),
                'shrinking': [True, False],
                'degree': SaneEqualityArray((4,), buffer=np.array([1., 2., 3., 4.])),
                'gamma': ['auto', 0.0001, 0.8889777777777778, 1.7778555555555555, 2.6667333333333336,
                          3.5556111111111113, 4.4444888888888885, 5.333366666666667, 6.222244444444444,
                          7.111122222222222, 8.0],
                'coef0': SaneEqualityArray((10,), buffer=np.array(
                    [0., 1.11111111, 2.22222222, 3.33333333, 4.44444444, 5.55555556, 6.66666667, 7.77777778, 8.88888889,
                     10.]))
            }
        }
