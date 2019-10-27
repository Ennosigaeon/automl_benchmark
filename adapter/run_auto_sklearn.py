import multiprocessing
import shutil
import time

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
from autosklearn.metrics import accuracy
from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

from benchmark import OpenMLBenchmark


def get_random_search_object_callback(scenario_dict, seed, ta, backend, metalearning_configurations, runhistory):
    """Random search."""
    scenario_dict['input_psmac_dirs'] = backend.get_smac_output_glob()
    scenario_dict['minR'] = len(scenario_dict['instances'])
    scenario_dict['initial_incumbent'] = 'RANDOM'
    scenario = Scenario(scenario_dict)
    return ROAR(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        runhistory=runhistory,
        run_id=seed
    )


def setup():
    try:
        shutil.rmtree('/tmp/autosklearn/')
    except OSError as e:
        pass


def main(bm: OpenMLBenchmark, timeout: int, run_timeout: int, jobs: int, random: bool) -> float:
    def get_spawn_classifier(X_train, y_train, tmp_folder, output_folder, seed0):
        def spawn_classifier(seed, dataset_name):
            # Use the initial configurations from meta-learning only in one out of
            # the processes spawned. This prevents auto-sklearn from evaluating the
            # same configurations in all processes.
            if seed == seed0 and not random:
                initial_configurations_via_metalearning = 25
                smac_scenario_args = {}
            else:
                initial_configurations_via_metalearning = 0
                smac_scenario_args = {'initial_incumbent': 'RANDOM'}

            callback = None
            if random:
                callback = get_random_search_object_callback

            # Arguments which are different to other runs of auto-sklearn:
            # 1. all classifiers write to the same output directory
            # 2. shared_mode is set to True, this enables sharing of data between
            # models.
            # 3. all instances of the AutoSklearnClassifier must have a different seed!
            automl = AutoSklearnClassifier(
                time_left_for_this_task=timeout,
                per_run_time_limit=run_timeout,
                shared_mode=True,
                tmp_folder=tmp_folder,
                output_folder=output_folder,
                delete_tmp_folder_after_terminate=False,
                ensemble_size=0,
                initial_configurations_via_metalearning=initial_configurations_via_metalearning,
                seed=seed,
                smac_scenario_args=smac_scenario_args,
                get_smac_object_callback=callback,
                ml_memory_limit=4096
            )
            automl.fit(X_train, y_train, dataset_name=dataset_name)
            print(automl.sprint_statistics())

        return spawn_classifier

    name = bm.get_meta_information()['name']

    X_train = bm.X_train
    y_train = bm.y_train
    X_test = bm.X_test
    y_test = bm.y_test

    tmp_folder = '/tmp/autosklearn/{}/tmp'.format(name)
    output_folder = '/tmp/autosklearn/{}/out'.format(name)

    seed = int(time.time())
    ensemble_size = 1 if random else 20

    processes = []
    spawn_classifier = get_spawn_classifier(X_train, y_train, tmp_folder, output_folder, seed)
    for i in range(jobs):
        p = multiprocessing.Process(target=spawn_classifier, args=(seed + i, name))
        p.start()
        processes.append(p)

    start = time.time()
    while time.time() - start <= 1.05 * timeout:
        if any(p.is_alive() for p in processes):
            time.sleep(10)
        else:
            break
    else:
        print('Grace period exceed. Killing workers.')
        for p in processes:
            p.terminate()
            p.join()

    print('Starting to build an ensemble!')
    automl = AutoSklearnClassifier(
        time_left_for_this_task=3600,
        per_run_time_limit=run_timeout,
        shared_mode=True,
        ensemble_size=ensemble_size,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        initial_configurations_via_metalearning=0,
        seed=seed,
        ml_memory_limit=4096
    )
    automl.fit_ensemble(
        y_train,
        task=MULTICLASS_CLASSIFICATION,
        metric=accuracy,
        precision='32',
        dataset_name=name,
        ensemble_size=ensemble_size
    )

    predictions = automl.predict(X_test)
    print(automl.show_models())
    score = 1 - sklearn.metrics.accuracy_score(y_test, predictions)
    return score