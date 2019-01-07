import numpy as np
from ConfigSpace import Configuration


def _configuration_as_array(foo, data_type=np.float):
    """ Decorator to allow the first input argument to 'objective_function' to be an array.

        For all continuous benchmarks it is often required that the input to the benchmark
        can be a (NumPy) array. By adding this to the objective function, both inputs types,
        dict and array, are possible.
    """

    def wrapper(self, configuration, **kwargs):
        if isinstance(configuration, dict) or isinstance(configuration, Configuration):
            blastoise = np.array(
                [configuration[k] for k in configuration],
                dtype=data_type
            )
        else:
            blastoise = configuration
        return (foo(self, blastoise, **kwargs))

    return (wrapper)
