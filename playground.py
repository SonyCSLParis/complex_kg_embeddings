import os
import json
from tqdm import tqdm
import pandas as pd
from rdflib import Graph
from pykeen.models import CompGCN, InductiveNodePieceGNN
from urllib.parse import quote, unquote
from pykeen.triples import TriplesFactory
from pykeen.datasets import EagerDataset
from pykeen.pipeline import pipeline
from sklearn.model_selection import train_test_split

FOLDER = "./ChronoGrapher/data/kg_base/"
columns = ["subject", "predicate", "object", "."]

triples = {}
for td in ["train", "valid", "test"]:
    df = pd.read_csv(os.path.join(FOLDER, f"{td}.csv"), sep=" ", header=None)
    df.columns = columns
    triples[td] = df[['subject', 'predicate', 'object']].values

triples_factory = {x: TriplesFactory.from_labeled_triples(triples, create_inverse_triples=True) for x, triples in triples.items()}


dataset = EagerDataset(
    training=triples_factory["train"],
    validation=triples_factory["valid"],
    testing=triples_factory["test"],
)
model = InductiveNodePieceGNN(
    triples_factory=dataset.training,
    inference_factory=dataset.validation,
    #interaction='distmult',
)
pipeline_result = pipeline(
    model=model,
    dataset=dataset,
    training_kwargs={"num_epochs": 20},
    evaluation_kwargs={"use_tqdm": True},
)
METRICS = ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']
for M in METRICS:
    print(f"{M}: {pipeline_result.metric_results.get_metric(M)}")