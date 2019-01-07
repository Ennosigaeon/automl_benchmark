from typing import List


class EvaluationResult:

    def __init__(self, start_time: float, end_time: float, score: float, params: dict):
        self.start_time = start_time
        self.end_time = end_time
        self.score = score
        self.params = params

    def __str__(self):
        d = {
            "start": self.start_time,
            "end": self.end_time,
            "score": self.score,
            "params": self.params
        }
        return str(d)

    def __repr__(self):
        return str(self)


class OptimizationStatistic:

    def __init__(self, algorithm: str, start: float):
        self.metadata = {
            "algorithm": algorithm,
            "begin": start
        }
        self.evaluations = []

    def add_result(self, result: List[EvaluationResult]):
        self.evaluations.extend(result)
