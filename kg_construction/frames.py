# -*- coding: utf-8 -*-
"""
Extract role layer from textual descriptions in the KG
"""
import os
import json
import pickle
from datetime import datetime
from urllib.parse import quote
from tqdm import tqdm
from loguru import logger
import pandas as pd
from rdflib import URIRef, Graph
from src.build_ng.frame_semantics import FrameSemanticsNGBuilder
from utils import update_log, save_json
from kglab.helpers.variables import NS_NIF, PREFIX_NIF, NS_EX, PREFIX_EX, NS_RDF, PREFIX_RDF, \
        PREFIX_FRAMESTER_WSJ, NS_FRAMESTER_WSJ, \
            NS_FRAMESTER_FRAMENET_ABOX_GFE, PREFIX_FRAMESTER_FRAMENET_ABOX_GFE, \
                NS_FRAMESTER_ABOX_FRAME, PREFIX_FRAMESTER_ABOX_FRAME, \
                        NS_EARMARK, PREFIX_EARMARK, NS_XSD, PREFIX_XSD, \
                            NS_SKOS, PREFIX_SKOS
from utils import init_graph
import concurrent.futures
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

PREFIX_TO_NS = {
    PREFIX_NIF: NS_NIF, PREFIX_RDF: NS_RDF, PREFIX_EX: NS_EX,
    PREFIX_FRAMESTER_WSJ: NS_FRAMESTER_WSJ,
    PREFIX_FRAMESTER_ABOX_FRAME: NS_FRAMESTER_ABOX_FRAME,
    PREFIX_EARMARK: NS_EARMARK, PREFIX_XSD: NS_XSD,
    PREFIX_SKOS: NS_SKOS}

EVENTS_DATES = "./revs_dates.csv"
NEW_FOLDER = "exps"
FS_KG_BUILDER = FrameSemanticsNGBuilder()
FRAME_KG_CACHE_P = "cache/frame_graphs"


if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)


def build_frame_semantics(path, max_workers):
    """ Build graph related to frame semantics
    `path`: {fn}_text.nt """
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return Graph()

    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    abstracts = df[df.predicate.str.contains("/abstract")]
    graph = init_graph(prefix_to_ns=PREFIX_TO_NS)

    tasks = []
    for name, group in abstracts.groupby("subject"):
    # for name, group in tqdm(abstracts.groupby("subject")):
        # for i, r in group.reset_index(drop=True).iterrows():
        #     event = name[1:-1]
        #     id_abstract = f"{event.split('/')[-1]}_Abstract{i}"
        #     graph.add((URIRef(quote(event, safe=":/")), URIRef("http://example.com/abstract"), URIRef(f"http://example.com/{quote(id_abstract)}")))

        #     curr_graph = process_abstract(id_abstract=id_abstract)
        #     graph += curr_graph
        
        #tasks = []
        for i, r in group.reset_index(drop=True).iterrows():
            event = name[1:-1]
            id_abstract = f"{event.split('/')[-1]}_Abstract{i}"
            graph.add((URIRef(quote(event, safe=":/")), URIRef("http://example.com/abstract"), URIRef(f"http://example.com/{quote(id_abstract)}")))
            tasks.append((id_abstract, r))

    # Parallel execution
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_abstract, id_, r) for id_, r in tasks]
        results = []
        for f in tqdm(concurrent.futures.as_completed(futures), 
                      total=len(futures), desc="Processing abstracts"):
            results.append(f.result())
        #results = [f.result() for f in concurrent.futures.as_completed(futures)]
        for g in results:
            graph += g
        #concurrent.futures.wait(futures)
    return graph

def process_abstract(id_abstract, r):
    """
    Loads or builds an RDF graph for the given abstract ID.

    If a cached graph exists, it is loaded; otherwise, a new graph is built and cached.

    Args:
        id_abstract (str): The abstract's identifier.

    Returns:
        rdflib.Graph: The RDF graph for the abstract.
    """
    if f"{id_abstract}.nt" in os.listdir(FRAME_KG_CACHE_P):
        curr_graph = Graph()
        curr_graph.parse(os.path.join(FRAME_KG_CACHE_P, f"{id_abstract}.nt"))
    else:
        curr_graph = FS_KG_BUILDER(text_input=r["object"], id_abstract=id_abstract)
        curr_graph.serialize(os.path.join(FRAME_KG_CACHE_P, f"{id_abstract}.nt"), format="nt")
    return curr_graph


def run_one_event(name, sd, ed, logs, max_workers):
    """ Run all for one event """
    logs = update_log(logs, "start_all")
    folder_name = name.split('/')[-1]
    folder_p = os.path.join(NEW_FOLDER, folder_name)

    if not os.path.exists(folder_p):
        os.makedirs(folder_p)

    # All with text for (frame-based) roles
    for fn in ["kg_base", "kg_base_prop", "kg_base_subevent", "kg_base_subevent_prop"]:
        text_p = os.path.join(folder_p, f"{fn}_text.nt")
        role_p = os.path.join(folder_p, f"{fn}_role_text.nt")
        if not os.path.exists(role_p):
            logger.info(f"Building {fn} + roles + text")
            logs = update_log(logs, f"start_{fn}_roles_text")
            graph = build_frame_semantics(path=text_p, max_workers=max_workers)
            graph.serialize(role_p, format="nt")
            logs = update_log(logs, f"end_{fn}_roles_text")
            save_json(data=logs, save_p=os.path.join(folder_p, "logs.json"))

    logs["end_all"] = str(datetime.now())
    return logs


def process_event(row, logs_p, folder_out, index, nb_event, max_workers):
    """Processes an individual event and updates logs."""
    perc = round(100 * (index + 1) / nb_event, 2)
    logger.info(f"Processing event {row.event} ({index+1}/{nb_event}, {perc}%)")

    if os.path.exists(logs_p):
        with open(logs_p, "r", encoding="utf-8") as openfile_main:
            curr_logs = json.load(openfile_main)
    else:
        curr_logs = {}

    curr_logs = run_one_event(name=row.event, sd=row.start, ed=row.end, logs=curr_logs, max_workers=max_workers)

    with open(logs_p, 'w', encoding='utf-8') as f:
        json.dump(curr_logs, f, indent=4)


def process_events(events_df, new_folder, max_workers=4):
    """Runs event processing in parallel."""
    nb_event = events_df.shape[0]

    tasks = []
    for index, row in tqdm(events_df.iterrows(), total=events_df.shape[0]):
        folder_out = os.path.join(new_folder, row.event.split('/')[-1])
        logs_p = os.path.join(folder_out, "logs.json")
        process_event(row, logs_p, folder_out, index, nb_event, max_workers)
        #tasks.append((row, logs_p, folder_out, index, nb_event))

    # Parallel execution
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(process_event, row, logs_p, folder_out, index, nb_event) for row, logs_p, folder_out, index, nb_event in tasks]
    #     concurrent.futures.wait(futures)



if __name__ == '__main__':
    events = pd.read_csv(EVENTS_DATES, index_col=0)
    process_events(events, NEW_FOLDER, max_workers=32)
        


