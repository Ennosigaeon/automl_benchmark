from typing import Dict

import numpy as np

from config import MetaConfigCollection, CATEGORICAL


class ConfigVectorizer:

    def __init__(self, classifier_file='assets/classifier.json'):
        self.config_space = MetaConfigCollection.from_json(classifier_file)

    def vectorize(self, dict: Dict):
        algorithm = dict['algorithm']
        definition = self.config_space.algos[algorithm]

        x = []
        for key in sorted(definition.dict.keys()):
            if key not in dict:
                value = np.nan
            else:
                d = definition.dict[key]
                value = dict[key]
                if d.type == CATEGORICAL:
                    value = -d.choices.index(value) / len(d.choices)
                else:
                    value = (value - d.lower) / (d.upper - d.lower)
            x.append(value)
        return x
