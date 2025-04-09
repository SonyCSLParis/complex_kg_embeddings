# -*- coding: utf-8 -*-
"""
Get templated descriptions from BN event and actors
"""
import re
import os
from tqdm import tqdm
import click
from collections import defaultdict
import pandas as pd
from loguru import logger
from rdflib import Graph, URIRef, Literal
from urllib.parse import unquote, quote
from src.build_ng.generic_kb_to_ng import KGConverter

COLUMNS_OUT = ["node", "description"]
COLUMNS_RDF = ["subject", "predicate", "object", "."]
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
WSJ = "https://w3id.org/framester/wsj/"
INTERFACE = KGConverter(dataset="dbpedia").interface
CACHE = defaultdict(dict)

def get_text(row):
    """ Get text (label+description) for row["node"] """
    if str(row['node']) in CACHE:
        row["label"] = CACHE[str(row['node'])]["label"]
        row["abstract"] = CACHE[str(row['node'])]["abstract"]
    else:
        preds = [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://dbpedia.org/ontology/abstract",
        ]
        params = {"subject": unquote(row['node'])}
        outgoing = INTERFACE.get_triples(**params)
        outgoing = [t for t in outgoing if (t[1] in preds and '@en' in t[2])]

        row["label"], row["abstract"] = "", ""
        for elt in outgoing:
            col = "label" if elt[1] == preds[0] else "abstract"
            row[col] = elt[2].replace('@en', '')
        CACHE[str(row['node'])]["abstract"] = row["abstract"]
        CACHE[str(row['node'])]["label"] = row["label"]
    return row

def parse_nt_line(line):
    pattern = r'<([^>]+)>\s<([^>]+)>\s(.+)\s\.'
    match = re.match(pattern, line)

    if match:
        subject = URIRef(match.group(1))
        predicate = URIRef(match.group(2))
        # Either the URI object or the literal object will be non-empty
        object_value = match.group(3)
        if object_value.startswith('<'):
            object_value = URIRef(object_value[1:-1])
        else:
            object_value = Literal(object_value.strip('"'))
        return (subject, predicate, object_value)
    return None

def rebuild_graph_from_nt(file_path):
    lines = [x.strip() for x in open(file_path, "r", encoding='utf-8').readlines()]
    graph=Graph()
    for line in tqdm(lines):
        parsed_line = parse_nt_line(line)
        if parsed_line:
            graph.add(parsed_line)
    return graph

def pretty_print(x):
    return unquote(str(x).split("/")[-1].replace("_", " "))

QUERY_ACTOR = """
PREFIX wsj: <https://w3id.org/framester/wsj/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?node ?role ?value (GROUP_CONCAT(?ent; separator=",") AS ?ents) ?et WHERE {
    ?node rdf:type wsj:MappedRole ;
           wsj:withfnfe ?role ;
           rdf:value ?value .
    OPTIONAL {?node skos:related ?ent .}
    ?event wsj:withmappedrole ?node ;
           wsj:onFrame ?et .
}
GROUP BY ?node ?role ?value ?event ?et
"""

def get_actor_description(row):
    output = f"Has the role of {pretty_print(row['role'])} during an event of type {pretty_print(row['et'])}. Has value '{pretty_print(row['value'])}'"
    if str(row['ents']):
        ents = str(row['ents']).split(",")
        if len(ents) == 1:
            output += f" that is related to {pretty_print(ents[0])}."
        else:
            ents_text =  ', '.join([pretty_print(x) for x in ents[:-1]]) + " and " + pretty_print(ents[-1])
            output += f" that is related to {ents_text}."
    else:
        output += "."
    row["description"] = output
    return row

QUERY_EVENT = """
PREFIX wsj: <https://w3id.org/framester/wsj/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?node ?lemma ?et ?doc_value WHERE {
    ?node rdf:type wsj:CorpusEntry ;
           wsj:onLemma ?lemma ;
           wsj:onFrame ?et ;
           wsj:fromDocument ?doc .
    ?doc rdf:value ?doc_value .
}
"""

def get_event_description(row):
    row["description"] = f"Event of type {pretty_print(row['et'])} triggered from lemma '{pretty_print(row['lemma'])}'. Document value: '{pretty_print(row['doc_value'])}'."
    return row

QUERY_BASE = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?node ?abstract ?label WHERE {
    ?node rdfs:label ?label ;
          dbo:abstract ?abstract .
}
"""

QUERY_OUTCOME = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.com/>
SELECT DISTINCT ?node ?abstract ?label WHERE {
    ?event ex:hasOutcome ?node .
    ?node rdf:value ?label ;
          ex:abstract ?abstract .
}
"""


def get_base_description(row):
    if str(row['abstract']):
        row["description"] = row['abstract']
    elif str(row['label']):
        row["description"] = row['label']
    else:
        row["description"] = pretty_print(row['node'])
    return row

QUERY_ENTS = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?node WHERE {
    ?s skos:related ?node .
}
"""


def get_info_file_bn(fp):
    graph = rebuild_graph_from_nt(file_path=fp)
    data = []

    # actors
    logger.info("Processing actors")
    output_query = graph.query(QUERY_ACTOR)
    df_actor = pd.DataFrame(output_query, columns=["node", "role", "value", "ents", "et"])
    if df_actor.shape[0] > 0:
        tqdm.pandas()
        df_actor = df_actor.progress_apply(get_actor_description, axis=1)
        data.append(df_actor[["node", "description"]])
    
    # events
    logger.info("Processing events")
    output_query = graph.query(QUERY_EVENT)
    df_event = pd.DataFrame(output_query, columns=["node", "lemma", "et", "doc_value"])
    if df_event.shape[0] > 0:
        tqdm.pandas()
        df_event = df_event.progress_apply(get_event_description, axis=1)
        data.append(df_event[["node", "description"]])

    # entities
    logger.info("Processing entities")
    output_query = graph.query(QUERY_ENTS)
    df_ents = pd.DataFrame(output_query, columns=["node"])
    if df_ents.shape[0] > 0:
        tqdm.pandas()
        df_ents = df_ents.progress_apply(get_text, axis=1)
        df_ents = df_ents.progress_apply(get_base_description, axis=1)
        data.append(df_ents[["node", "description"]])

    if "causation" in fp:
        # outcomes
        logger.info("Processing outcomes")
        output_query = graph.query(QUERY_OUTCOME)
        df_outcome = pd.DataFrame(output_query, columns=["node", "abstract", "label"])
        if df_outcome.shape[0] > 0:
            tqdm.pandas()
            df_outcome = df_outcome.progress_apply(get_base_description, axis=1)
            data.append(df_outcome[["node", "description"]])
        
    return pd.concat(data).drop_duplicates()

def get_info_file_base(fp):
    graph = rebuild_graph_from_nt(file_path=fp)
    logger.info("Processing base")
    output_query = graph.query(QUERY_BASE)
    df = pd.DataFrame(output_query, columns=["node", "abstract", "label"])
    if df.shape[0] > 0:
        tqdm.pandas()
        df = df.progress_apply(get_base_description, axis=1)
        return df[["node", "description"]].drop_duplicates()
    return pd.DataFrame(columns=["node", "description"])
    
    

@click.command()
@click.argument("folder")
def main(folder):
    events = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
    logger.info(f"Total # of events: {str(len(events))}")
    events = [x for x in events if not os.path.exists(os.path.join(folder, x, "descriptions.csv"))]
    nb_event = len(events)
    logger.info(f"Total # of events to process: {str(nb_event)}")
    
    for i, event in tqdm(enumerate(events)):
        logger.info(f"EVENT: {event} ({str(i+1)}/{str(nb_event)})")
        curr_folder = os.path.join(folder, event)
        df_to_concat = []
        files = [x for x in os.listdir(curr_folder) if x.endswith("_text.nt")]
        for file in files:
            logger.info(f"Processing file: {file}")
            file_path = os.path.join(curr_folder, file)
            if os.path.getsize(file_path) <= 1:
                logger.info(f"Skipping empty file: {file}")
            else:
                if any(x in file for x in ["role", "causation"]):
                    df = get_info_file_bn(os.path.join(curr_folder, file))
                else:
                    df = get_info_file_base(os.path.join(curr_folder, file))
                df_to_concat.append(df)
        df_all = pd.concat(df_to_concat).drop_duplicates()
        df_all["node"] = df_all["node"].apply(lambda x: f"<{x}>")
        df_all.to_csv(os.path.join(curr_folder, "descriptions.csv"))
        logger.info(f"Descriptions for {event} stored in descriptions.csv")

if __name__ == '__main__':
    main()


