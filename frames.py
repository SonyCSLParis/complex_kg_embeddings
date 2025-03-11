# -*- coding: utf-8 -*-
"""
chronographer
"""
import os
import json
import pickle
import subprocess
from datetime import datetime
from urllib.parse import quote
from tqdm import tqdm
from loguru import logger
import pandas as pd
from rdflib import URIRef, Graph
from src.build_ng.frame_semantics import FrameSemanticsNGBuilder
from utils import update_log, save_json, save_pickle
from kglab.helpers.variables import NS_NIF, PREFIX_NIF, NS_EX, PREFIX_EX, NS_RDF, PREFIX_RDF, \
        PREFIX_FRAMESTER_WSJ, NS_FRAMESTER_WSJ, \
            NS_FRAMESTER_FRAMENET_ABOX_GFE, PREFIX_FRAMESTER_FRAMENET_ABOX_GFE, \
                NS_FRAMESTER_ABOX_FRAME, PREFIX_FRAMESTER_ABOX_FRAME, \
                        NS_EARMARK, PREFIX_EARMARK, NS_XSD, PREFIX_XSD, \
                            NS_SKOS, PREFIX_SKOS
from kglab.helpers.kg_build import init_graph
import concurrent.futures

PREFIX_TO_NS = {
    PREFIX_NIF: NS_NIF, PREFIX_RDF: NS_RDF, PREFIX_EX: NS_EX,
    PREFIX_FRAMESTER_WSJ: NS_FRAMESTER_WSJ,
    PREFIX_FRAMESTER_ABOX_FRAME: NS_FRAMESTER_ABOX_FRAME,
    PREFIX_EARMARK: NS_EARMARK, PREFIX_XSD: NS_XSD,
    PREFIX_SKOS: NS_SKOS}

EVENTS_DATES = "./revs_dates.csv"
NEW_FOLDER = "exps"
FS_KG_BUILDER = FrameSemanticsNGBuilder()
FRAME_KG_CACHE_P = "frame_kg_cache.pkl"


if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)


def build_frame_semantics(path):
    """ Build graph related to frame semantics
    `path`: {fn}_text.nt """
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return Graph()
    # try to load cache, else delete and create new one
    if os.path.exists(FRAME_KG_CACHE_P):
        try:
            with open(FRAME_KG_CACHE_P, 'rb') as f:
                cache = pickle.load(f)
        except:
                if os.path.exists(FRAME_KG_CACHE_P):
                    subprocess.call(f"rm {FRAME_KG_CACHE_P}", shell=True)
                cache = {}
    else:
        cache = {}

    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    abstracts = df[df.predicate.str.contains("/abstract")]
    graph = init_graph(prefix_to_ns=PREFIX_TO_NS)
    for name, group in tqdm(abstracts.groupby("subject")):
        for i, r in group.reset_index(drop=True).iterrows():
            event = name[1:-1]
            id_abstract = f"{event.split('/')[-1]}_Abstract{i}"
            graph.add((URIRef(quote(event, safe=":/")), URIRef("http://example.com/abstract"), URIRef(f"http://example.com/{quote(id_abstract)}")))
            if r["object"] in cache:
                curr_graph = cache[r["object"]]
            else:
                curr_graph = FS_KG_BUILDER(text_input=r["object"], id_abstract=id_abstract)
                cache[r["object"]] = curr_graph
                save_pickle(cache=cache, save_p=FRAME_KG_CACHE_P)
            graph += curr_graph
    return graph


def run_one_event(name, sd, ed, logs):
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
            graph = build_frame_semantics(path=text_p)
            if os.path.exits(FRAME_KG_CACHE_P):
                subprocess.call(f"rm {FRAME_KG_CACHE_P}", shell=True)
            graph.serialize(role_p, format="nt")
            logs = update_log(logs, f"end_{fn}_roles_text")
            save_json(data=logs, save_p=os.path.join(folder_p, "logs.json"))

    logs["end_all"] = str(datetime.now())
    return logs


def process_event(row, logs_p, folder_out, index, nb_event):
    """Processes an individual event and updates logs."""
    perc = round(100 * (index + 1) / nb_event, 2)
    logger.info(f"Processing event {row.event} ({index+1}/{nb_event}, {perc}%)")

    if os.path.exists(logs_p):
        with open(logs_p, "r", encoding="utf-8") as openfile_main:
            curr_logs = json.load(openfile_main)
    else:
        curr_logs = {}

    curr_logs = run_one_event(name=row.event, sd=row.start, ed=row.end, logs=curr_logs)

    with open(logs_p, 'w', encoding='utf-8') as f:
        json.dump(curr_logs, f, indent=4)


def parallel_process(events_df, new_folder, max_workers=4):
    """Runs event processing in parallel."""
    nb_event = events_df.shape[0]

    tasks = []
    for index, row in events_df.iterrows():
        folder_out = os.path.join(new_folder, row.event.split('/')[-1])
        logs_p = os.path.join(folder_out, "logs.json")
        tasks.append((row, logs_p, folder_out, index, nb_event))

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_event, row, logs_p, folder_out, index, nb_event) for row, logs_p, folder_out, index, nb_event in tasks]
        concurrent.futures.wait(futures)



if __name__ == '__main__':
    events = pd.read_csv(EVENTS_DATES, index_col=0)
    parallel_process(events, NEW_FOLDER, max_workers=32)
        


