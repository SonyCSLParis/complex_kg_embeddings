# -*- coding: utf-8 -*-
"""
various utils
"""
import os
import pandas as pd
from datetime import datetime
from urllib.parse import quote, unquote

FILTER_OUT = [
    "http://dbpedia.org/ontology/wikiPageWikiLink",
    "http://dbpedia.org/ontology/wikiPageRedirects",
    "http://dbpedia.org/ontology/wikiPageDisambiguates",
    "http://dbpedia.org/ontology/thumbnail",
    "http://dbpedia.org/ontology/wikiPageExternalLink",
    "http://dbpedia.org/ontology/wikiPageID",
    "http://dbpedia.org/ontology/wikiPageLength",
    "http://dbpedia.org/ontology/wikiPageRevisionID",
    "http://dbpedia.org/ontology/wikiPageWikiLinkText",
    "http://dbpedia.org/ontology/wikiPageOutDegree",
    "http://dbpedia.org/ontology/abstract"]

def update_log(logs, name):
    """update 'name' key with current date (if not already exists) """
    logs[name] = str(datetime.now())
    return logs

def get_iteration(folder):
    """ Real iteration number (sometimes early stop)"""
    subgraph = [x for x in os.listdir(folder) if x.endswith("-subgraph.csv")]
    return subgraph[0].split('-')[0]

def get_nodes(df):
    """ Get unique nodes from kg """
    output = set(list(df["subject"].unique()) + list(df["object"].unique()))
    return list({x[1:-1] for x in output})

def get_text(node, interface):
    """ Get text (label+description) for node """
    preds = [
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://dbpedia.org/ontology/abstract",
    ]
    params = {"subject": unquote(node)}
    outgoing = interface.get_triples(**params)
    outgoing = [t for t in outgoing if (t[1] in preds and '@en' in t[2])]
    return [(f'<{quote(a, safe=":/")}>', f"<{b}>", c.replace('@en', ''), '.') for (a, b, c) in outgoing]


def get_properties(node, interface):
    """ Get binary properties to describe a node """
    params = {"subject": unquote(node)}
    outgoing = interface.get_triples(**params)
    dbo = "http://dbpedia.org/ontology/"
    outgoing = [x for x in outgoing if (x[1].startswith(dbo) or \
        x[2].startswith(dbo))]
    outgoing = [x for x in outgoing if not x[2].startswith('"')]
    outgoing = [x for x in outgoing if x[1] not in FILTER_OUT]
    return [(quote(a, safe=":/"), b, quote(c, safe=":/"), '.') for (a, b, c) in outgoing]


def concat_nt_in_folder(folder_p, columns):
    """ Concat .nt encoded as .csv in a folder """
    files = [os.path.join(folder_p, x) for x in os.listdir(folder_p) if x.endswith('.csv')]
    files = [pd.read_csv(x, sep=' ', header=None) for x in files]
    output = pd.concat(files)
    output.columns = columns
    return output