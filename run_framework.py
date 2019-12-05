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
jobs = 4

if __name__ == '__main__':
    algorithm = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print('Algorithm: ', algorithm)
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    task_ids = [
        [3, 12, 31, 53, 3917, 7593, 9952, 9977, 9981, 10101],
        [14965, 34539, 146195, 146212, 146818, 146821, 146822, 146825, 167119, 167120],
        [167121, 167124, 168329, 168330, 168331, 168332, 168335, 168337, 168338],
        [168868, 168908, 168909, 168910, 168911, 168912, 189354, 189355, 189356],
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
        for i in range(5):
            try:
                print('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
                start = time.time()
                bm = OpenMLBenchmark(task)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    if algorithm == 'atm':
                        if run_atm.skip(task):
                            continue

                        score = run_atm.main(bm, timeout, jobs)
                    if algorithm == 'random':
                        if run_auto_sklearn.skip(task):
                            continue

                        score = run_auto_sklearn.main(bm, timeout, run_timeout, jobs, random=True)
                    elif algorithm == 'auto-sklearn':
                        if run_auto_sklearn.skip(task):
                            continue

                        score = run_auto_sklearn.main(bm, timeout, run_timeout, jobs, random=False)
                    elif algorithm == 'dummy':
                        if run_baseline.skip(task):
                            continue

                        score = run_baseline.main(bm, dummy=True)
                    elif algorithm == 'rf':
                        if run_baseline.skip(task):
                            continue

                        score = run_baseline.main(bm, dummy=False)
                    elif algorithm == 'h2o':
                        if run_h2o.skip(task):
                            continue

                        score = run_h2o.main(bm, timeout, run_timeout, jobs)
                    elif algorithm == 'hpsklearn':
                        if run_hpsklearn.skip(task):
                            continue

                        score = run_hpsklearn.main(bm, timeout, run_timeout)
                    elif algorithm == 'tpot':
                        if run_tpot.skip(task):
                            continue

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
        print(res[-1])

    for i in range(len(res)):
        print('        {},  # {}'.format(res[i], task_ids[i]))

    print(res)
