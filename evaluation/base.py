from typing import List

from hpolib.abstract_benchmark import AbstractBenchmark
from pymongo import MongoClient

from adapter.base import OptimizationStatistic, BenchmarkResult


class MongoPersistence:

    def __init__(self, url: str, port: int = 27017):
        self.client = MongoClient(url, port)
        self.db = self.client.benchmarks

    def clear_old_results(self, benchmark: AbstractBenchmark) -> None:
        collection = self.db[benchmark.get_meta_information()['name']]
        collection.drop()

    def store_new_run(self, res: BenchmarkResult):
        collection = self.db[res.benchmark.get_meta_information()['name']]
        d = res.as_dict()
        collection.insert_one(d)

    def store_results(self, res: BenchmarkResult, stats: OptimizationStatistic) -> None:
        collection = self.db[res.benchmark.get_meta_information()['name']]

        # collection.delete_one({'algorithm': stats.algorithm})

        d = stats.as_dict(include_evaluations=True)
        collection.update_one({'seed': res.seed}, {'$push': {'solvers': d}})

    def load_single(self, benchmark: AbstractBenchmark) -> BenchmarkResult:
        collection = self.db[benchmark.get_meta_information()['name']]
        d = collection.find_one()
        res = BenchmarkResult.from_dict(d)
        res.benchmark = benchmark
        return res

    def load_all(self, benchmark: AbstractBenchmark) -> List[BenchmarkResult]:
        collection = self.db[benchmark.get_meta_information()['name']]
        ls = [BenchmarkResult.from_dict(doc) for doc in collection.find()]
        for res in ls:
            res.benchmark = benchmark
        return ls
