import datetime
import multiprocessing
import sys
import time
import traceback
import warnings

import humanfriendly

from benchmark import SantanderBenchmark

timeout = 3600  # in seconds
run_timeout = 600  # in seconds
jobs = 4


def run(it) -> None:
    bm = SantanderBenchmark()
    print('##\nIteration {} at {}\n##'.format(i, datetime.datetime.now().time()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res = []

        for fold in bm.folds:
            try:
                if algorithm == 'atm':
                    from adapter import run_atm
                    score, estimator = run_atm.main(fold, bm, timeout, jobs, score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                elif algorithm == 'random':
                    from adapter import run_auto_sklearn
                    score, estimator = run_auto_sklearn.main(fold, bm, timeout, run_timeout, jobs, random=True,
                                                             score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                elif algorithm == 'auto-sklearn':
                    from adapter import run_auto_sklearn
                    score, estimator = run_auto_sklearn.main(fold, bm, timeout, run_timeout, jobs, random=False,
                                                             score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                elif algorithm == 'dummy':
                    from adapter import run_baseline
                    score, estimator = run_baseline.main(fold, dummy=True, score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                elif algorithm == 'rf':
                    from adapter import run_baseline
                    score, estimator = run_baseline.main(fold, dummy=False, score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                elif algorithm == 'h2o':
                    from adapter import run_h2o
                    score, estimator = run_h2o.main(fold, bm, timeout, run_timeout, jobs, score=False)
                    df = estimator.predict(run_h2o._createFrame(bm.X_test)).as_data_frame()
                    predictions = df.values[:, 1:]
                    run_h2o._cleanup(None)
                elif algorithm == 'hpsklearn':
                    from adapter import run_hpsklearn
                    score, estimator = run_hpsklearn.main(fold, timeout, run_timeout, score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                elif algorithm == 'tpot':
                    from adapter import run_tpot
                    score, estimator = run_tpot.main(fold, timeout, run_timeout, jobs, score=False)
                    predictions = estimator.predict_proba(bm.X_test)
                else:
                    raise ValueError('Unknown algorithm {}'.format(algorithm))
                print(score)

                bm.format_output(predictions, algorithm, it * 10 + len(res))
                res.append((score, predictions))
            except Exception:
                traceback.print_exc()


if __name__ == '__main__':
    algorithm = sys.argv[1]
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print('Algorithm: ', algorithm)
    print('Timeout: ', timeout)
    print('Run Timeout: ', run_timeout)

    for i in range(0, 3):
        try:
            start = time.time()

            p = multiprocessing.Process(target=run, args=(i,))
            p.start()

            p.join(timeout * 1.5)

            if p.is_alive():
                print('Grace period exceed. Stopping benchmark.')
                p.terminate()
                p.join()
            print('Duration', humanfriendly.format_timespan(time.time() - start))
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            traceback.print_exc()
