# -*- coding: utf-8 -*-
"""
Different inductive splits for the data
"""
import pandas as pd
from sklearn.model_selection import train_test_split
COLUMNS = ["subject", "predicate", "object", "."]

SEM_PREFIX = "http://semanticweb.cs.vu.nl/2009/11/sem/"
RDF_TYPE = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
PRED_CONTEXT = [
    RDF_TYPE,
    f'<{SEM_PREFIX}hasBeginTimeStamp>',
    '<http://www.w3.org/2000/01/rdf-schema#subPropertyOf>'
]

def add_class_type(df, pred, class_type):
    """ Add (?s, rdf:type, class_type) to the dataframe """
    instances = df[df["predicate"].str.startswith(pred)]["object"].unique()
    data = [(x, RDF_TYPE, f"<{class_type}>", ".") for x in instances]
    return pd.concat([
        df, pd.DataFrame(data, columns=COLUMNS)
    ])

def update_df(df):
    """ Add all class type info """
    for pred, ct in [
        (f"<{SEM_PREFIX}hasActor>", f"{SEM_PREFIX}Actor"),
        (f"<{SEM_PREFIX}hasPlace>", f"{SEM_PREFIX}Place"),
        (f"<{SEM_PREFIX}subEventOf>", f"{SEM_PREFIX}Event"),
        ('<https://w3id.org/framester/framenet/abox/gfe/', 'http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#Phrase'),
        ('<http://www.w3.org/2004/02/skos/core#related>', 'https://dbpedia.org/resource/Entity'),
        ('<http://semanticweb.cs.vu.nl/2009/11/sem/hasActor#', f"{SEM_PREFIX}Actor"),
        ('<http://example.com/subject>', 'https://dbpedia.org/resource/Entity'),
        ('<http://example.com/object>', 'https://dbpedia.org/resource/Entity'),
        ('<http://example.com/Statement#', 'http://example.com/Statement'),
        ('<http://semanticweb.cs.vu.nl/2009/11/sem/hasRole>', f"{SEM_PREFIX}Role"),
    ]:
        df = add_class_type(df, pred, ct)
    
    # add statement-related info
    statement_start = "<http://example.com/Statement#"
    instances = df[df["subject"].str.startswith(statement_start)]["subject"].unique()
    data = [(x, RDF_TYPE, "<http://example.com/Statement>", ".") for x in instances]
    df = pd.concat([df, pd.DataFrame(data, columns=COLUMNS)])
    return df

def assert_all_in_inference_graph(inference_pred, inference_graph):
    pred_in_pred = set(inference_pred["predicate"])
    pred_in_graph = set(inference_graph["predicate"])
    print(f"PRED DIFF: {pred_in_pred - pred_in_graph}")
    assert pred_in_pred.issubset(pred_in_graph), "Some predicates in inference_pred are not in inference_graph"

    nodes_in_pred = set(inference_pred["subject"]).union(set(inference_pred["object"]))
    nodes_in_graph = set(inference_graph["subject"]).union(set(inference_graph["object"]))
    print(f"NODE DIFF: {nodes_in_pred - nodes_in_graph}")
    assert nodes_in_pred.issubset(nodes_in_graph), "Some nodes in inference_pred are not in inference_graph"

def split_data_inductive_random(df):
    """ Takes an input KG and splits it into `inference_graph` 
    (~structural context) and `inference_pred` (~predictions) 
    - Random strategy:
        * Some predicates automatically in context
        * All others split randomly """
    df = update_df(df)
    inference_graph = df[df["predicate"].isin(PRED_CONTEXT)]
    events_added = inference_graph["subject"].unique()
    inference_pred = pd.DataFrame(columns=COLUMNS)
    df = df.drop(inference_graph.index)

    # DBpedia predicate -> context
    df_dbpedia = df[df["predicate"].str.startswith("<http://dbpedia.org/")]
    inference_graph = pd.concat([inference_graph, df_dbpedia])
    df = df.drop(df_dbpedia.index)

    # Find nodes that appear only once in either subject or object columns
    all_node_counts = pd.concat([df[~df["subject"].isin(events_added)]["subject"], df[~df["object"].isin(events_added)]["object"]]).value_counts()
    single_occurrence_nodes = set(all_node_counts[all_node_counts == 1].index)

    # Move triples containing nodes that appear only once to inference_graph
    single_occurrence_mask = df["subject"].isin(single_occurrence_nodes) | df["object"].isin(single_occurrence_nodes)
    inference_graph = pd.concat([inference_graph, df[single_occurrence_mask]])
    df = df.drop(df[single_occurrence_mask].index)
    
    # Group by predicate to ensure all predicates appear in both splits
    pred_groups = df.groupby("predicate")
    for _, group in pred_groups:
        if group.shape[0] == 1:  # only one sample, can only be inference_graph
            inference_graph = pd.concat([inference_graph, group])
        else:
            graph, pred = train_test_split(group, test_size=0.4, random_state=23)
            inference_graph = pd.concat([inference_graph, graph])
            inference_pred = pd.concat([inference_pred, pred])
    
    # Assert that all predicates in inference_pred are also in inference_graph
    assert_all_in_inference_graph(inference_pred, inference_graph)

    return inference_graph, inference_pred

def update_df_hr(df):
    data = []
    for col in ["actor", "role", "event"]:
        vals = df[col].unique()
        data += [(x, RDF_TYPE, f"<{SEM_PREFIX}{col.capitalize()}>") for x in vals]
    return pd.DataFrame(data, columns=COLUMNS[:3])
    