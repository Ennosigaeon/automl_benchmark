import time

import numpy as np


def _dict_as_array(foo, data_type=np.float):
    """ Decorator to allow the first input argument to 'objective_function' to be an array.

        For all continuous benchmarks it is often required that the input to the benchmark
        can be a (NumPy) array. By adding this to the objective function, both inputs types,
        dict and array, are possible.
    """

    def wrapper(self, configuration, **kwargs):
        if isinstance(configuration, dict):
            blastoise = np.array(
                [configuration[k] for k in configuration],
                dtype=data_type
            )
        else:
            blastoise = configuration
        return (foo(self, blastoise, **kwargs))

    return (wrapper)


def meta_information(foo):
    def wrapper(self, confguration, **kwargs):
        start = time.time()
        res = foo(self, confguration, **kwargs)

        res['start'] = start
        res['end'] = time.time()
        # res['config'] = confguration
        return res

    return wrapper
