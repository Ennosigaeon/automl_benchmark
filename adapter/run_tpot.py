from tpot import TPOTClassifier

from benchmark import OpenMLBenchmark


def skip(id: int) -> bool:
    failed = [9910, 14952, 14954, 167124, 146819]
    return id in failed


def setup():
    pass


def main(bm: OpenMLBenchmark, timeout: int, run_timeout: int, jobs: int) -> float:
    X_train = bm.X_train
    y_train = bm.y_train
    X_test = bm.X_test
    y_test = bm.y_test

    pipeline_optimizer = TPOTClassifier(
        max_time_mins=timeout / 60,
        max_eval_time_mins=run_timeout / 60,
        scoring='accuracy',
        n_jobs=jobs,
        verbosity=1
    )
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.fitted_pipeline_)
    score = 1 - pipeline_optimizer.score(X_test, y_test)
    return score
