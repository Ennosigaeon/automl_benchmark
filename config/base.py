import collections
import json
from typing import Dict, Union, List

import networkx as nx
import numpy as np

CATEGORICAL = "categorical"
UNI_FLOAT = "uniform_float"
UNI_INT = "uniform_int"
PARENT = "parent"
VALUE = "value"


class MetaConfigCollection:

    def __init__(self, d: Dict = None):
        self.algos: Dict[str, MetaConfig] = {}
        if (d is not None):
            for key, value in d.items():
                conf = MetaConfig(value)
                conf.sort_keys()
                self.algos[key] = conf

    def keys(self):
        return self.algos.keys()

    def items(self):
        return self.algos.items()

    @staticmethod
    def from_json(file: str, validate: bool = True) -> 'MetaConfigCollection':
        with open(file) as f:
            d = json.load(f, object_pairs_hook=collections.OrderedDict)
            return MetaConfigCollection(d)

    @staticmethod
    def __validate(json_file: json) -> bool:
        # with open("config\schema.json") as schema:  # TODO import schema
        #     schema = json.load(schema)
        #     validate(json_file, schema)  # TODO improve schema
        return True


class MetaConfig:

    def __init__(self, d: Dict = None):
        self.dict: Dict[str, ConfigFeature] = collections.OrderedDict()

        if (d is not None):
            for key, value in d.items():
                self.dict[key] = ConfigFeature(value)

    def add_feature(self, name: str, definition: Union[Dict, str]):
        self.dict[name] = ConfigFeature(definition)

    def sort_keys(self) -> None:
        """
        Process all previously stored ~MetaConfigFeature and order them depending on their dependencies.
        :return:
        """
        graph = ConfigInheritanceGraph(self)
        if (len(graph.simple_cycles()) > 0):
            raise ValueError('Encountered circular dependencies while sorting config features. '
                             'Please check your configuration file')

        d = collections.OrderedDict()
        nodes = graph.bfs_tree(graph.ROOT)
        for key in nodes:
            if key in self.dict.keys():
                d[key] = self.dict[key]
        self.dict = d

    def items(self):
        return self.dict.items()

    @staticmethod
    def continuous_from_bounds(bounds: np.ndarray):
        res = MetaConfig()
        for i in range(bounds.shape[0]):
            res.add_feature(f"x{i}", {
                "type": "uniform_float",
                "lower": bounds[i][0],
                "upper": bounds[i][1],
            })
        return res


class ConfigFeature(collections.MutableMapping):
    TYPE = "type"

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    @property
    def type(self):
        return self.store[self.TYPE].lower()

    @property
    def choices(self):
        return self.store["choices"]

    @property
    def lower(self):
        return self.store["lower"]

    @property
    def upper(self):
        return self.store["upper"]

    @property
    def default(self):
        return self.store.get('default_value')

    @property
    def log(self):
        return self.store.get("log", False)

    @property
    def condition(self):
        return self.store.get("condition")

    def has_condition(self):
        return self.condition is not None


class ConfigInheritanceGraph:
    ROOT = 'root'
    CONFIG = 'config'

    def __init__(self, config: MetaConfig, ignore_options: bool = False):
        self.G = nx.DiGraph()
        self.G.add_node(self.ROOT)

        self.__add_nodes(config, ignore_options)
        self.__add_edges(config)

    def __add_nodes(self, algo: MetaConfig, ignore_options=False) -> None:
        for key, value in algo.items():
            self.G.add_node(key, config=value)
            if value.type == CATEGORICAL and not ignore_options:
                for choice in value.choices:
                    if not self.G.has_node(choice):
                        self.G.add_node(choice, config=None)
                    self.G.add_edge(key, choice)

    def __add_edges(self, config: MetaConfig) -> None:
        for key, config in config.items():
            if config.has_condition():
                for value in config.condition[VALUE]:
                    self.G.add_edge(value, key)
            else:
                self.G.add_edge(self.ROOT, key)

    def get_config(self) -> dict:
        return nx.get_node_attributes(self.G, self.CONFIG)

    def successors(self, node: str) -> nx.DiGraph:
        return self.G.successors(node)

    def edge_dfs(self, source: str) -> List[str]:
        return list(nx.edge_dfs(self.G, source))

    def bfs_tree(self, source: str) -> List[str]:
        return list(nx.bfs_tree(self.G, source))

    def simple_cycles(self) -> List:
        return list(nx.simple_cycles(self.G))
