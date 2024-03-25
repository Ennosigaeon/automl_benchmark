import requests
import openml

from evaluation import scripts

task_ids = scripts.all_tasks

mapping = {}

for task in task_ids:
    resp = requests.post('https://www.openml.org/es/run/run/_search?size=0', json={
        "query": {
            "bool": {
                "must": [{
                    "term": {
                        "run_task.task_id": task
                    }
                }, {
                    "nested": {
                        "path": "evaluations",
                        "query": {
                            "exists": {
                                "field": "evaluations"
                            }
                        }
                    }
                }
                ]
            }
        },
        "aggs": {
            "flows": {
                "terms": {
                    "field": "run_flow.flow_id",
                    "size": 100
                },
                "aggs": {
                    "top_score": {
                        "top_hits": {
                            "_source": ["run_id", "run_flow.name", "run_flow.parameters", "run_flow.flow_id",
                                        "uploader", "evaluations.evaluation_measure", "evaluations.value"],
                            "sort": [{
                                "evaluations.value": {
                                    "order": "desc",
                                    "nested_path": "evaluations",
                                    "nested_filter": {
                                        "term": {
                                            "evaluations.evaluation_measure": "predictive_accuracy"
                                        }
                                    }
                                }
                            }
                            ],
                            "size": 100
                        }
                    }
                }
            }
        }
    }).json()
    try:
        score = -1
        for flow in resp['aggregations']['flows']['buckets']:
            for run in flow['top_score']['hits']['hits']:
                score = max(score, float(run['sort'][0]))
    except Exception:
        score = -1

    dataset = openml.tasks.get_task(task).get_dataset().dataset_id
    mapping[dataset] = score
print(mapping)
