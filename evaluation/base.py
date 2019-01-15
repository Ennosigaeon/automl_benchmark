from typing import List

from hpolib.abstract_benchmark import AbstractBenchmark
from pymongo import MongoClient

from adapter.base import OptimizationStatistic


class MongoPersistent:

    def __init__(self, url: str, port: int = 27017):
        self.client = MongoClient(url, port)
        self.db = self.client.benchmarks

    def clear_old_results(self, benchmark: AbstractBenchmark) -> None:
        collection = self.db[benchmark.get_meta_information()['name']]
        collection.drop()

    def store_results(self, benchmark: AbstractBenchmark, stats: OptimizationStatistic) -> None:
        collection = self.db[benchmark.get_meta_information()['name']]

        d = stats.as_dict(include_evaluations=True)
        collection.insert_one(d)

    def load_results(self, benchmark: AbstractBenchmark) -> List[OptimizationStatistic]:
        collection = self.db[benchmark.get_meta_information()['name']]
        return [OptimizationStatistic.from_dict(doc) for doc in collection.find()]
