import datetime
import itertools
import sys
import time
import warnings

import traceback

import humanfriendly

from adapter import run_auto_sklearn, run_atm, run_baseline, run_h2o, run_hpsklearn, run_tpot
from benchmark import OpenMLBenchmark

timeout = 3600  # in seconds
run_timeout = 600  # in seconds
jobs = 2

if __name__ == '__main__':
    algorithm = sys.argv[1]
    idx = sys.argv[2] if len(sys.argv) > 2 else None

    print('Algorithm: ', algorithm)
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    task_ids = [
        [9910, 14952, 14954, 146800, 146817],
        [146819, 146820, 146824, 167121],
        [167124, 167125, 167140, 167141]
    ]

    if idx is not None:
        print('Using chunk {}/{}'.format(idx, len(task_ids)))
        task_ids = task_ids[idx]
    else:
        print('Using all tasks')
        task_ids = list(itertools.chain.from_iterable(task_ids))

    res = []
    for task in task_ids:
        print('#######\nStarting task {}\n#######'.format(task))
        res.append([])
        for i in range(10):
            try:
                print('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
                start = time.time()
                bm = OpenMLBenchmark(task)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    if algorithm == 'atm':
                        run_atm.setup()
                        score = run_atm.main(bm, timeout, jobs)
                    if algorithm == 'random':
                        run_auto_sklearn.setup()
                        score = run_auto_sklearn.main(bm, timeout, run_timeout, jobs, random=True)
                    elif algorithm == 'auto-sklearn':
                        run_auto_sklearn.setup()
                        score = run_auto_sklearn.main(bm, timeout, run_timeout, jobs, random=False)
                    elif algorithm == 'dummy':
                        run_baseline.setup()
                        score = run_baseline.main(bm, dummy=True)
                    elif algorithm == 'rf':
                        run_baseline.setup()
                        score = run_baseline.main(bm, dummy=False)
                    elif algorithm == 'h2o':
                        run_h2o.setup()
                        score = run_h2o.main(bm, timeout, run_timeout, jobs)
                    elif algorithm == 'hpsklearn':
                        run_hpsklearn.setup()
                        score = run_hpsklearn.main(bm, timeout, run_timeout)
                    elif algorithm == 'tpot':
                        run_tpot.setup()
                        score = run_tpot.main(bm, timeout, run_timeout, jobs)
                    else:
                        raise ValueError('Unknown algorithm {}'.format(algorithm))

                    res[-1].append(score)
                    print('Misclassification Rate', score)
                    print('Duration', humanfriendly.format_timespan(time.time() - start))
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    print(res)
                    raise e
                traceback.print_exc()
                print('Misclassification rate', 1)
    print(res)
