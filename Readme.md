# AutoML-Benchmark
This project evaluates the performance of various AutoML frameworks on different benchmark datasets. A detailed description of the evaluated frameworks and detailed evaluation results is available in our [survey paper](https://arxiv.org/abs/1904.12054). The source code is available on [GitHub](https://github.com/Ennosigaeon/automl_benchmark).

## Installation
- Install swig `sudo apt install swig`
- Install build-essential `sudo apt install build-essential`
- Install auto-sklearn requirements `curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install`
- Install hpolib2 `pip3 install git+https://github.com/automl/HPOlib1.5.git`
- Install mysql client `sudo apt install libmysqlclient-dev`
- Install all other requirements via `pip3 install -r requirements.txt`
- Install ATM (https://github.com/HDI-Project/ATM)
- Install RoBo (https://github.com/automl/RoBO)
- Install Optunity (https://optunity.readthedocs.io/en/latest/user/installation.html)

Some of the tested AutoML frameworks require some hotfixes to actually work. Unfortunately, it is not possible to list all required changes as most of the frameworks are still in development and regularly updated.


## Usage
To actually use the benchmark, adapt the _run.py_ file. The tested frameworks are configured at the head of the file
```python
config_dict = {
        'n_jobs': 1,
        'timeout': None,
        'iterations': 500,
        'seed': int(time.time()),

        'random_search': True,
        'grid_search': False,
        'smac': False,
        'hyperopt': False,
        'bohb': False,
        'robo': False,
        'optunity': False,
        'btb': False 
}
```
Parameters are:
* `n_jobs` defines the number of parallel processes
* `timeout` defines the maximum evaluation time. This option is not supported by all frameworks. Can not be used together with `iterations`.
* `iterations` defines the maximum number of iterations. Can not be used together with `timeout`.
* `seed` defines the random state to make evaluations reproducible.
* A list of supported frameworks with boolean parameters, whether this framework should be evaluated or not

Next, configure the tested benchmark at the bottom of the file.

```python
logger.info('Main start')
try:
    persistence = MongoPersistence(args.database, read_only=False)
    b = benchmark.Rosenbrock20D()
    for i in range(20):
        run(persistence, b)
except (SystemExit, KeyboardInterrupt, Exception) as e:
    logger.error(e, exc_info=True)
logger.info('Main finished')
```

Finally, execute the _run.py_ script with the mandatory parameter `--database`
```bash
python3 run.py --database localhost
```
All results are stored in the provided MongoDB.

### Implemented AutoML Frameworks
Currently implemented are adapters for
* [BoHB](https://github.com/automl/HpBandSter)
* [BTB](https://github.com/HDI-Project/BTB)
* [Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)
* [hyperopt](https://github.com/hyperopt/hyperopt)
* [Optunity](https://github.com/claesenm/optunity)
* [Random Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
* [RoBO](https://github.com/automl/RoBO)
* [SMAC](https://github.com/automl/SMAC3)

Each of these frameworks is configured via a unified search space configuration. An example configuration is provided in the _assets_ folder.

### Implemented Benchmarks
Implemented is a bunch of synthetic test functions, the Iris dataset and an OpenML benchmark. The OpenML benchmark is able to test a single OpenML dataset or a complete OpenML suite.

Example usage:
```python
logger.info('Main start')
try:
    persistence = MongoPersistence(args.database, read_only=False)
    for i in range(20):
        for b in benchmark.OpenML100Suite().load(chunk=args.chunk):
             logger.info('Starting OpenML benchmark {}'.format(b.task_id))
             run(persistence, b)
except (SystemExit, KeyboardInterrupt, Exception) as e:
    logger.error(e, exc_info=True)

logger.info('Main finished')
```
Using the optional parameter `--chunk` only a part of the datasets is evaluated. This option can be used to distribute the evaluation in a cluster.


## Evaluating complete ML Pipelines
This code also allows the evaluation of frameworks building complete ML pipelines. Currently implemented are
* [ATM](https://github.com/HDI-Project/ATM)
* [auto-sklearn](https://github.com/automl/auto-sklearn)
* [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
* [TPOT](https://github.com/EpistasisLab/tpot)

For each framework, a dedicated run script exists.
