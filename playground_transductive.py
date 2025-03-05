import os
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

from utils import concat_nt_in_folder

FOLDER_P = "./ChronoGrapher/data/kg_base_prop"
columns = ["subject", "predicate", "object", "."]

triples = concat_nt_in_folder(folder_p=FOLDER_P, columns=columns)
sh = TriplesFactory.from_labeled_triples(triples[columns[:3]].values)
train, valid, test = sh.split([0.8, 0.1, 0.1], random_state=42)

pipeline_result = pipeline(
    model="transe",
    model_kwargs={"embedding_dim": 128, "scoring_fct_norm": 2},
    random_seed=23,
    loss_kwargs={"margin": 1.2895626564788845},
    training=train,
    #validation=valid,
    testing=valid,
    training_kwargs={"num_epochs": 10, "batch_size": 512},
    evaluation_kwargs={"use_tqdm": True},
    optimizer_kwargs={"lr": 0.08003518425075115},
    negative_sampler_kwargs={"num_negs_per_pos": 43}
)

METRICS = ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']
for metric in METRICS:
    print(f"{metric}: {pipeline_result.metric_results.get_metric(metric)}")
