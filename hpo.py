# -*- coding: utf-8 -*-
"""
From here: https://pykeen.readthedocs.io/en/stable/tutorial/running_hpo.html
Possible to optimize the following *_kwargs arguments:
model, loss, regularizer, optimizer, negative_sampler
"""
import os
import json
import random
from typing import Union
from datetime import datetime
import click
import numpy as np
import torch
from pykeen.hpo import hpo_pipeline
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from utils import concat_nt_in_folder

SEED = 23
COLUMNS = ["subject", "predicate", "object", "."]
METRICS = ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']
KWARGS_NAMES = [
    "training", "model", "loss", "regularizer", "optimizer", "negative_sampler"
]
KWARGS_RANGES = {
    "training": {
        "num_epochs": {"type": "categorical", "choices": [100*x for x in range(1, 6)]},
        #"num_epochs": {"type": "categorical", "choices": [10, 20]}
    },
    "model": {
        "embedding_dim": {"type": "categorical", "choices": [x*16 for x in range(1, 33)]},
        "random_seed": {"type": "categorical", "choices": [42],}
    },
    "loss": {

    },
    "regularizer": {

    },
    "optimizer": {
        "lr": {"type": "categorical", "choices": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]},
    },
    "negative_sampler": {
        "num_negs_per_pos": {"type": "categorical", "choices": [1] + [10*x for x in range(1, 11)]}
    }
}


def get_kwargs_pipeline(best_params):
    """ output of hpo in format compatible with regular pipeline """
    res = {x: {} for x in KWARGS_NAMES}
    for key, val in best_params.items():
        param, sub_param = key.split('.')
        res[param][sub_param] = val
    return res


class HPOExperiment:
    """ HPO for regular embeddings """
    def __init__(self, folder_p: str, kwargs_ranges: dict, split: list,
                 sampler: str = "tpe", evaluator: str = "rankbased",
                 n_trials: int = 20, model: str = "transe"):
        triples = concat_nt_in_folder(folder_p=folder_p, columns=COLUMNS)
        sh = TriplesFactory.from_labeled_triples(triples[COLUMNS[:3]].values)
        self.train, self.valid, self.test = sh.split(split, random_state=42)

        self.kwargs_ranges = kwargs_ranges

        self.params = {
            "sampler": sampler,
            "evaluator": evaluator,
            "n_trials": n_trials,
            "model": model,
            "split": split,
            "data": folder_p
        }

    def run_hpo(self, save_dir: Union[str, None] = None):
        """ HPO pipeline """
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        result = hpo_pipeline(
            training=self.train, validation=self.valid, testing=self.valid,
            model=self.params["model"],
            n_trials=self.params["n_trials"],
            sampler=self.params["sampler"],  #'random', 'grid', 'tpe'
            evaluator=self.params["evaluator"],
            training_kwargs_ranges=self.kwargs_ranges["training"],
            model_kwargs_ranges=self.kwargs_ranges["model"],
            loss_kwargs_ranges=self.kwargs_ranges["loss"],
            regularizer_kwargs_ranges=self.kwargs_ranges["regularizer"],
            optimizer_kwargs_ranges=self.kwargs_ranges["optimizer"],
            negative_sampler_kwargs_ranges=self.kwargs_ranges["negative_sampler"]
        )
        if save_dir:
            result.save_to_directory(f"ChronoGrapher/hpo/{self.params['model']}")
        return result

    def run_pipeline(self, params):
        """ Run pipeline over best params """
        kwargs_pipeline = get_kwargs_pipeline(best_params=params)
        result = pipeline(
            model=self.params["model"],
            random_seed=23,
            training=self.train,
            testing=self.valid,
            training_kwargs=kwargs_pipeline["training"],
            model_kwargs=kwargs_pipeline["model"],
            loss_kwargs=kwargs_pipeline["loss"],
            regularizer_kwargs=kwargs_pipeline["regularizer"],
            optimizer_kwargs=kwargs_pipeline["optimizer"],
            negative_sampler_kwargs=kwargs_pipeline["negative_sampler"],
            evaluation_kwargs={"use_tqdm": True},
        )
        return result


def start_logs(params):
    """ Logs with params info """
    logs = {"start_pipeline": str(datetime.now())}
    logs.update(params)
    return logs


@click.command()
@click.argument("folder_p")
@click.argument("model")
@click.argument("n_trials")
@click.argument("save_fp", default=None)
def main(folder_p, model, n_trials, save_fp):
    """ Main: run hpo > run model on best params """
    hpo_exp = HPOExperiment(
        folder_p=folder_p, split=[0.8, 0.1, 0.1],
        kwargs_ranges=KWARGS_RANGES, model=model, n_trials=int(n_trials))
    save_dir = None if not save_fp else os.path.join(save_fp, model)
    hpo_result = hpo_exp.run_hpo(save_dir=save_dir)
    logs = start_logs(params=hpo_exp.params)
    pipeline_result = hpo_exp.run_pipeline(params=hpo_result.study.best_params)
    logs["end_pipeline"] = str(datetime.now())

    for metric in METRICS:
        print(f"{metric}: {pipeline_result.metric_results.get_metric(metric)}")

    if save_dir:  # save logs
        logs.update({x: pipeline_result.metric_results.get_metric(x) for x in METRICS})
        with open(os.path.join(save_fp, model, "logs.json"), 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4)

if __name__ == '__main__':
    # python ChronoGrapher/hpo.py ./ChronoGrapher/data/kg_base_prop transe 2 ./ChronoGrapher/hpo
    main()
