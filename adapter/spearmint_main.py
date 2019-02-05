import copy

import numpy as np


def _dict_as_array(foo, data_type=np.float):
    """ Decorator to allow the first input argument to 'objective_function' to be an array.

        For all continuous benchmarks it is often required that the input to the benchmark
        can be a (NumPy) array. By adding this to the objective function, both inputs types,
        dict and array, are possible.
    """

    def wrapper(configuration, **kwargs):
        if isinstance(configuration, dict):
            blastoise = np.array(
                [configuration[k] for k in configuration],
                dtype=data_type
            )
        else:
            blastoise = configuration
        return (foo(blastoise, **kwargs))

    return (wrapper)


@_dict_as_array
def levy(x, **kwargs):
    z = 1 + ((x[0] - 1.) / 4.)
    s = np.power((np.sin(np.pi * z)), 2)
    y = (s + ((z - 1) ** 2) * (1 + np.power((np.sin(2 * np.pi * z)), 2)))

    return y


@_dict_as_array
def branin(x, **kwargs):
    y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

    return y


@_dict_as_array
def hartman6(x, **kwargs):
    """6d Hartmann test function
        input bounds:  0 <= xi <= 1, i = 1..6
        global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
        min function value = -3.32237
    """

    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)

    return -external_sum


@_dict_as_array
def rosenbrock10D(x, **kwargs):
    y = 0
    d = 10
    for i in range(d - 1):
        y += 100 * (x[i + 1] - x[i] ** 2) ** 2
        y += (x[i] - 1) ** 2

    return y


def main(job_id, params):
    name = params['__benchmark__'][0]
    config = copy.copy(params)
    del config['__benchmark__']

    if name == 'Levy':
        return levy(config)[0]
    if name == 'Branin':
        return branin(config)[0]
    if name == 'Hartmann6':
        return hartman6(config)[0]
    if name == 'Rosenbrock10D':
        return rosenbrock10D(config)[0]
    raise ValueError('Unknown benchmark {}'.format(name))
