# -*- coding: utf-8 -*-
"""
Metrics for the KGs
"""
import os
import click
import pandas as pd
from tqdm import tqdm
from loguru import logger

def get_degree_simple_triple(df):
    """ in-degree/out-degree of a regular KG (s, p, o) """
    return {
        "in_degree_3": df.groupby("object").agg({"subject": "nunique"})["subject"].mean(),
        "out_degree_3": df.groupby("subject").agg({"object": "nunique"})["object"].mean(),
    }

def get_degree_hr_triple(df):
    return {
        "in_degree_5": df.groupby("o2").agg({"p2": "count"})["p2"].mean(),
        "in_degree_simple_hr": df.groupby("o1").agg({"p1": "count"})["p1"].mean(),
        "out_degree_simple_hr": df.groupby("s1").agg({"p1": "count"})["p1"].mean(),
    }


def get_metric_rdf_star(fp):
    """
    Get the RDF* metrics for the given file path

    Max number of triples of a KG of order n (n nodes) (3-arity)
    - Undirected = n * (n - 1) / 2
    - Directed = 2 * Undirected = n * (n - 1)

    Max number of triples of a KG of order n (n nodes) (5-arity)
    - Hypothesis: max # of triples for 3-arity + max # of triples for 5-arity
    - max # of triples for 3-arity: n * (n - 1)
    - max # of triples for 5-arity: every triple of arity 3 is linked to every node except the 2 nodes of the triple
    - Directed = n * (n-1) + n * (n-1) * (n-2) = n * (n-1)**2
    """
    logger.info("Reading hr-triple file")
    statements = open(fp, 'r').readlines()
    statements = [s.strip().split(",") for s in statements if s.strip()]
    logger.info("Splitting df into simple-triples and hr-triples")
    df_simple_triple = pd.DataFrame([x for x in statements if len(x) == 3], columns=["subject", "predicate", "object"]).drop_duplicates()
    df_hr_triple = pd.DataFrame([x for x in statements if len(x) == 5], columns=["s1", "p1", "o1", "p2", "o2"]).drop_duplicates()
    df_simple_triple.to_csv("test.csv")
    nb = len(statements)
    metrics = {}

    entities = set(y for x in statements for i, y in enumerate(x) if i % 2 == 0)
    relations = set(y for x in statements for i, y in enumerate(x) if i % 2 == 1)
    max_statement = nb * nb**2
    logger.info("Calculating simple metrics")
    metrics.update({
        "entities": len(entities), "relations": len(relations),
        "statements": nb, "density": nb / max_statement,
        "3_statements": len(df_simple_triple), "5_statements": len(df_hr_triple),})
    logger.info("Calculating degree metrics")
    metrics.update(get_degree_simple_triple(df_simple_triple))
    metrics.update(get_degree_hr_triple(df_hr_triple))
    return metrics

def get_metrics_simple(fp):
    logger.info("Reading simple-triple file")
    if fp.endswith(".csv"):  # ULTRA
        statements = pd.read_csv(fp, sep=" ", header=None)
        statements.columns = ["subject", "predicate", "object", "."]
        statements = statements[statements.columns[:3]]
    else:  # fp.endswith(".txt"):  # SimKGC
        statements = pd.read_csv(fp, sep="\t", header=None)
        statements.columns = ["subject", "predicate", "object"]
    nb = len(statements)
    metrics = {}

    entities = set(statements["subject"].unique().tolist() + statements["object"].unique().tolist())
    relations = set(statements["predicate"].unique().tolist())
    max_statement = nb * nb**2
    logger.info("Calculating simple metrics")
    metrics.update({
        "entities": len(entities), "relations": len(relations),
        "statements": nb, "density": nb / max_statement,
        "3_statements": len(statements), "5_statements": 0,})
    logger.info("Calculating degree metrics")
    metrics.update(get_degree_simple_triple(statements))
    metrics.update({
        "in_degree_5": 0,
        "in_degree_simple_hr": 0,
        "out_degree_simple_hr": 0})
    return metrics



def get_params(d):
    return {
        "prop": 1 if "prop_1" in d else 0,
        "subevent": 1 if "subevent_1" in d else 0,
        "role": 1 if "role_1" in d else 0,
        "causation": 1 if "causation_1" in d else 0,
        "syntax": d.split("_syntax_")[-1]
    }

INFO = {
    "ilp": {
        "data_path": os.path.expanduser("~/.data/ilp/NarrativeInductiveDataset/inductive/statements/"),
        "file_names": ["transductive_train.txt", "inductive_train.txt", "inductive_val.txt", "inductive_ts.txt"],
        "td": "simple-triple+text+hr-triple",
        "metrics_func": get_metric_rdf_star
    },
    "ultra": {
        "data_path": os.path.expanduser("~/git/ULTRA/kg-datasets/NarrativeInductiveDataset/"),
        "file_names": ["train.csv", "inference_graph.csv", "inference_valid.csv", "inference_test.csv"],
        "td": "simple-triple",
        "metrics_func": get_metrics_simple
    },
    "simkgc": {
        "data_path": os.path.expanduser("~/data/SimKGC/NarrativeInductiveDataset/"),
        "file_names": ["train.txt", "valid.txt", "test.txt"],
        "td": "simple-triple+text",
        "metrics_func": get_metrics_simple
    }
}


@click.command()
@click.argument("method", type=click.Choice(["ilp", "ultra", "simkgc"]))
@click.option("--folder-out", default="./")
def main(method, folder_out):
    data = []
    info = INFO[method]
    datasets = os.listdir(info["data_path"])
    for d in tqdm(datasets):
        logger.info(f"Processing {d}")
        for f in info["file_names"]:
            logger.info(f"Processing {f}")
            curr_data = {"td": info["td"], "dataset": d, "file": f}
            curr_data.update(get_params(d))
            fp = os.path.join(info["data_path"], d, f)
            metrics = info["metrics_func"](fp)
            curr_data.update(metrics)
            data.append(curr_data)
    
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    pd.DataFrame(data).to_csv(os.path.join(folder_out, f"metrics_{method}.csv"))
    


if __name__ == '__main__':
    main()

# FP = os.path.join(
#     FOLDER_DATA_RDF_STAR,
#     "kg_base_prop_0_subevent_0_role_0_causation_1_syntax_hyper_relational_rdf_star",
#     "transductive_train.txt")
# M = get_metric_rdf_star(fp=FP)
# print(M)