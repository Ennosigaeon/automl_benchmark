from typing import List

from bson.errors import InvalidDocument
from hpolib.abstract_benchmark import AbstractBenchmark
from pymongo import MongoClient

from adapter.base import OptimizationStatistic, BenchmarkResult
import util.logger

logger = util.logger.get()

class Persistence:

    def clear_old_results(self, benchmark: AbstractBenchmark) -> None:
        pass

    def store_new_run(self, res: BenchmarkResult):
        pass

    def store_results(self, res: BenchmarkResult, stats: OptimizationStatistic) -> None:
        pass

    def load_single(self, benchmark: AbstractBenchmark) -> BenchmarkResult:
        pass

    def load_all(self, benchmark: AbstractBenchmark) -> List[BenchmarkResult]:
        pass

class MongoPersistence(Persistence):

    def __init__(self, url: str, port: int = 27017, db='benchmarks', read_only: bool = False):
        self.client = MongoClient(url, port)
        self.db = eval('self.client.' + db)
        self.read_only = read_only

    def clear_old_results(self, benchmark: AbstractBenchmark) -> None:
        if self.read_only:
            return
        collection = self.db[benchmark.get_meta_information()['name']]
        collection.drop()

    def store_new_run(self, res: BenchmarkResult):
        if self.read_only:
            return
        collection = self.db[res.benchmark.get_meta_information()['name']]
        d = res.as_dict()
        collection.insert_one(d)

    def store_results(self, res: BenchmarkResult, stats: OptimizationStatistic) -> None:
        if self.read_only:
            return
        collection = self.db[res.benchmark.get_meta_information()['name']]

        # collection.delete_one({'algorithm': stats.algorithm})

        d = stats.as_dict(include_evaluations=True)
        try:
            collection.update_one({'seed': res.seed}, {'$push': {'solvers': d}})
        except InvalidDocument as ex:
            logger.fatal('Invalid document {}'.format(d))
            raise ex

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
