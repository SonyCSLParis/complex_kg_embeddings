# -*- coding: utf-8 -*-
"""
Convert outputs of ChronoGrapher's frames to different syntactic KGs
"""
import os
import json
import click
from tqdm import tqdm
from rdflib import Graph
from loguru import logger
from representations import KGRepresentationsConverter
from utils import save_json, update_log, load_json, save_pickle, load_pickle_cache

CONVERTER = KGRepresentationsConverter()

NAME_TO_FUNC = {
    "simple_rdf_prop": CONVERTER.to_simple_rdf_prop,
    "hypergraph_bn": CONVERTER.to_hypergraph_bn,
    "hyper_relational_rdf_star": CONVERTER.to_hyper_relational_rdf_star,
    "simple_rdf_sp": CONVERTER.to_simple_rdf_sp,
    "simple_rdf_reification": CONVERTER.to_simple_rdf_reification
}

def save_graph(name, graph, save_p):
    """ save graph (different ways depending on syntax) """
    if "hyper" in name:
        graph.to_csv(save_p, sep=" ", header=False, index=False, na_rep="")
    else:
        graph.to_csv(save_p, header=None, index=False, sep=" ")
    

def run_one_event(folder_p, cache_sp_p, cache_reification_p):
    """ Run all for one event (shared logs) """
    logs_p = os.path.join(folder_p, "logs.json")
    with open(logs_p, "r", encoding="utf-8") as f:
        logs = json.load(f)
    cache_sp = load_json(cache_sp_p)
    cache_reification = load_pickle_cache(file_p=cache_reification_p, return_val=1)
    roles_f = [x for x in os.listdir(folder_p) if "_role_text" in x]
    for rf in roles_f:  # convert each file to all possible syntactic representation
        file_p = os.path.join(folder_p, rf)
        graph = Graph()
        graph.parse(file_p)
    
        for name, func in NAME_TO_FUNC.items():
            logger.info(f"Converting {rf.replace('.nt', '')} to {name}")
            extension = "nt" if "hyper" not in name else "txt"
            save_p = file_p.replace("_text.nt", f"_{name}.{extension}")
            if not os.path.exists(save_p):
                logs = update_log(logs, f"start_{save_p.split('/')[-1].replace('.nt', '')}")
                if name not in ["simple_rdf_sp", "simple_rdf_reification"]:
                    output_graph = func(graph=graph)
                elif name == "simple_rdf_sp":
                    output_graph, cache_sp = func(graph=graph, cache=cache_sp)
                    save_json(data=cache_sp, save_p=cache_sp_p)
                else:  # name == "simple_rdf_reification"
                    output_graph, cache_reification = func(graph=graph, statement_nb=cache_reification)
                    save_pickle(cache=cache_reification, save_p=cache_reification_p)

                save_graph(name=name, graph=output_graph, save_p=save_p)
                logs = update_log(logs, f"end_{save_p.split('/')[-1].replace('.nt', '')}")
                save_json(data=logs, save_p=logs_p)
        



@click.command()
@click.argument("folder")
@click.argument("cache_sp_p")
@click.argument("cache_reification_p")
def main(folder, cache_sp_p, cache_reification_p):
    events = os.listdir(folder)
    events = ["Cambodian_Civil_War"]

    for event in tqdm(events):
        curr_folder = os.path.join(folder, event)
        run_one_event(folder_p=curr_folder, cache_sp_p=cache_sp_p, cache_reification_p=cache_reification_p)


if __name__ == '__main__':
    # python convert_kg.py exps cache/sp.json cache/reification.pkl
    main()
