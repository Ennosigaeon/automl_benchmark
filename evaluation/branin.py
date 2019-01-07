from adapter.random_search import ObjectiveRandomSearch
from benchmark import BraninFunction

b = BraninFunction()
seed = 42

# Random Search
rs = ObjectiveRandomSearch(30, 10, random_state=seed)
stats = rs.optimize(b)
print(stats.evaluations)
