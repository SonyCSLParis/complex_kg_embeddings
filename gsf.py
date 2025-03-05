# -*- coding: utf-8 -*-
"""
chronographer
"""
import os
import json
import subprocess
from datetime import datetime
from urllib.parse import quote
from tqdm import tqdm
from loguru import logger
import pandas as pd
from rdflib import URIRef
from src.framework import GraphSearchFramework
from src.build_ng.frame_semantics import FrameSemanticsNGBuilder
from src.build_ng.generic_kb_to_ng import KGConverter
from utils import update_log, get_iteration, get_nodes, get_text, get_properties
from kglab.helpers.variables import NS_NIF, PREFIX_NIF, NS_EX, PREFIX_EX, NS_RDF, PREFIX_RDF, \
        PREFIX_FRAMESTER_WSJ, NS_FRAMESTER_WSJ, \
            NS_FRAMESTER_FRAMENET_ABOX_GFE, PREFIX_FRAMESTER_FRAMENET_ABOX_GFE, \
                NS_FRAMESTER_ABOX_FRAME, PREFIX_FRAMESTER_ABOX_FRAME, \
                        NS_EARMARK, PREFIX_EARMARK, NS_XSD, PREFIX_XSD, \
                            NS_SKOS, PREFIX_SKOS
from kglab.helpers.kg_build import init_graph

PREFIX_TO_NS = {
    PREFIX_NIF: NS_NIF, PREFIX_RDF: NS_RDF, PREFIX_EX: NS_EX,
    PREFIX_FRAMESTER_WSJ: NS_FRAMESTER_WSJ,
    PREFIX_FRAMESTER_ABOX_FRAME: NS_FRAMESTER_ABOX_FRAME,
    PREFIX_EARMARK: NS_EARMARK, PREFIX_XSD: NS_XSD,
    PREFIX_SKOS: NS_SKOS}

EVENTS_DATES = "./Rev/revs_dates.csv"
CONFIG_PATH = "ChronoGrapher/config_complexkg.json"
MODE = "search_type_node_no_metrics"
NS = "all"
WALK = "informed"
CONVERTER = KGConverter(dataset="dbpedia")
OLD_FOLDER = "../graph_search_framework/experiments"
NEW_FOLDER = "ChronoGrapher/exps"
FS_KG_BUILDER = FrameSemanticsNGBuilder()

if not os.path.exists(NEW_FOLDER):
    os.makedirs(NEW_FOLDER)

with open(CONFIG_PATH, "r", encoding="utf-8") as openfile_main:
    CONFIG = json.load(openfile_main)
if "rdf_type" in CONFIG:
    CONFIG["rdf_type"] = list(CONFIG["rdf_type"].items())

def save_log(logs, save_p):
    """ save logs """
    with open(save_p, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)

def convert_generic_to_event_kg(df, sd, ed, save_p):
    """ Output of chronographer -> event-centric KG (simple) """
    graph = CONVERTER(input_df=df, start_d=sd, end_d=ed)
    graph.serialize(save_p, format="nt")

def move_exps(folder_p):
    """ Move exp to different folder """
    experiments = sorted(os.listdir(OLD_FOLDER))
    old_folder_p = os.path.join(OLD_FOLDER, experiments[-1])
    subprocess.call(
        f'cp -r "{old_folder_p}/" "{folder_p}/"', shell=True)
    subprocess.call(f'rm -rf "{old_folder_p}"', shell=True)

def add_text_to_base(path, interface):
    """Add text description and entity labels to base KG"""
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    nodes = get_nodes(df=df)
    for node in tqdm(nodes):
        outgoing = get_text(node=node, interface=interface)
        curr_df = pd.DataFrame(data=outgoing, columns=columns)
        df = pd.concat([df, curr_df])
    return df

def get_text_from_kg(path, interface):
    """Add text description and entity labels to base KG"""
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    nodes = get_nodes(df=df)
    output_df = pd.DataFrame(columns=columns)
    for node in tqdm(nodes):
        outgoing = get_text(node=node, interface=interface)
        curr_df = pd.DataFrame(data=outgoing, columns=columns)
        output_df = pd.concat([output_df, curr_df])
    return output_df

def add_prop_to_base(path, interface):
    """Add properties base KG"""
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    nodes = get_nodes(df=df)
    for node in tqdm(nodes):
        outgoing = get_properties(node=node, interface=interface)
        curr_df = pd.DataFrame(data=outgoing, columns=columns)
        df = pd.concat([df, curr_df])
    return df

def get_prop_from_base(path, interface):
    """Add properties base KG"""
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    nodes = get_nodes(df=df)
    output_df = pd.DataFrame(columns=columns)
    for node in tqdm(nodes):
        outgoing = get_properties(node=node, interface=interface)
        curr_df = pd.DataFrame(data=outgoing, columns=columns)
        output_df = pd.concat([output_df, curr_df])
    for col in columns[:3]:
        output_df[col] = output_df[col].apply(lambda x: f"<{x}>")
    return output_df

def build_frame_semantics(path):
    """ Build graph related to frame semantics
    `path`: {fn}_text.nt """
    try:
        df = pd.read_csv(path, sep=" ", header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    columns = ["subject", "predicate", "object", "."]
    df.columns = columns
    abstracts = df[df.predicate.str.contains("/abstract")]
    graph = init_graph(prefix_to_ns=PREFIX_TO_NS)
    for name, group in tqdm(abstracts.groupby("subject")):
        for i, r in group.reset_index(drop=True).iterrows():
            event = name[1:-1]
            id_abstract = f"{event.split('/')[-1]}_Abstract{i}"
            graph.add((URIRef(quote(event, safe=":/")), URIRef("http://example.com/abstract"), URIRef(f"http://example.com/{quote(id_abstract)}")))
            curr_graph = FS_KG_BUILDER(text_input=r["object"], id_abstract=id_abstract)
            graph += curr_graph
    return graph

def run_one_event(name, sd, ed, logs):
    """ Run all for one event """
    logs = update_log(logs, "start_all")
    folder_name = name.split('/')[-1]
    folder_p = os.path.join(NEW_FOLDER, folder_name)

    if not os.path.exists(folder_p):
        os.makedirs(folder_p)

    # Base KG (plain, simple event): events, actors, locations
    kg_base_p = os.path.join(folder_p, "kg_base.nt")
    if not os.path.exists(kg_base_p):
        logger.info("Building base KG")
        logs = update_log(logs, "start_kg_base")
        df = pd.DataFrame(
            data=[(name, "rdf:type", "sem:Event", "ingoing", 1, None)],
            columns=["subject", "predicate", "object", "type_df", "iteration", "regex_helper"]
        )
        convert_generic_to_event_kg(
            df=df, sd=sd, ed=ed, save_p=kg_base_p)
        logs = update_log(logs, "end_kg_base")
        save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    # Base KG (with properties)
    kg_base_prop_p = os.path.join(folder_p, "kg_base_prop.nt")
    if not os.path.exists(kg_base_prop_p):
        logger.info("Building props of base KG")
        logs = update_log(logs, "start_kg_base_prop")
        df = get_prop_from_base(path=kg_base_p, interface=CONVERTER.interface)
        df.to_csv(kg_base_prop_p, header=None, index=False, sep=" ")
        logs = update_log(logs, "end_kg_base_prop")
        save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    # Base KG + sub-events
    ## Graph search to retrieve sub-events
    if not os.path.exists(os.path.join(folder_p, "config.json")):
        config = CONFIG
        config.update({"start": name, "start_date": sd,
                       "end_date": ed, "name_exp": folder_name})
        logs = update_log(logs, "start_cg")
        chronographer = GraphSearchFramework(config=config, mode=MODE,
                                             node_selection=NS, walk=WALK)
        logger.info("Running ChronoGrapher")
        chronographer()
        # Move exps
        move_exps(folder_p=folder_p)
        logs = update_log(logs, "end_cg")
        save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    iteration = get_iteration(folder=folder_p)

    ## Base KG + sub-events (simple)
    kg_base_subevent_p = os.path.join(folder_p, "kg_base_subevent.nt")
    if not os.path.exists(kg_base_subevent_p):
        logger.info("Building base KG + sub-events ")
        logs = update_log(logs, "start_kg_base_subevent")
        df = pd.read_csv(os.path.join(folder_p, f"{str(iteration)}-subgraph.csv"), index_col=0)
        convert_generic_to_event_kg(df=df, sd=sd, ed=ed, save_p=kg_base_subevent_p)
        logs = update_log(logs, "end_kg_base_subevent")
        save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    ## Base KG  + sub-events (with properties)
    kg_base_subevent_prop_p = os.path.join(folder_p, "kg_base_subevent_prop.nt")
    if not os.path.exists(kg_base_subevent_prop_p):
        logger.info("Building props of base KG + sub-events")
        logs = update_log(logs, "start_kg_base_subevent_prop")
        df = get_prop_from_base(path=kg_base_subevent_p, interface=CONVERTER.interface)
        df.to_csv(kg_base_subevent_prop_p, header=None, index=False, sep=" ")
        logs = update_log(logs, "end_kg_base_subevent_prop")
        save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    # All the above (with text)
    for fn in ["kg_base", "kg_base_prop", "kg_base_subevent", "kg_base_subevent_prop"]:
        text_p = os.path.join(folder_p, f"{fn}_text.nt")
        if not os.path.exists(text_p):
            logger.info(f"Building {fn} + text")
            logs = update_log(logs, f"start_{fn}_text")
            df = get_text_from_kg(
                path=os.path.join(folder_p, f"{fn}.nt"), interface=CONVERTER.interface)
            df.to_csv(text_p, header=None, index=False, sep=" ")
            logs = update_log(logs, f"end_{fn}_text")
            save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    # All with text for (frame-based) roles
    for fn in ["kg_base", "kg_base_prop", "kg_base_subevent", "kg_base_subevent_prop"]:
        text_p = os.path.join(folder_p, f"{fn}_text.nt")
        role_p = os.path.join(folder_p, f"{fn}_role_text.nt")
        if not os.path.exists(role_p):
            logger.info(f"Building {fn} + roles + text")
            logs = update_log(logs, f"start_{fn}_roles_text")
            graph = build_frame_semantics(path=text_p)
            graph.serialize(role_p, format="nt")
            logs = update_log(logs, f"end_{fn}_roles_text")
            save_log(logs=logs, save_p=os.path.join(folder_p, "logs.json"))

    logs["end_all"] = str(datetime.now())
    return logs




if __name__ == '__main__':
    events = pd.read_csv(EVENTS_DATES, index_col=0)
    nb_event = events.shape[0]
    for index, row in events.iterrows():
        perc = round(100*(index+1)/nb_event, 2)
        logger.info(f"Processing event {row.event} ({index+1}/{nb_event}, {perc}%)")
        folder_out = os.path.join(NEW_FOLDER, row.event.split('/')[-1])
        logs_p = os.path.join(folder_out, "logs.json")
        if os.path.exists(logs_p):
            with open(logs_p, "r", encoding="utf-8") as openfile_main:
                curr_logs = json.load(openfile_main)
        else:
            curr_logs = {}
        curr_logs = run_one_event(name=row.event, sd=row.start, ed=row.end, logs=curr_logs)
        with open(os.path.join(folder_out, "logs.json"), 'w', encoding='utf-8') as f:
            json.dump(curr_logs, f, indent=4)

