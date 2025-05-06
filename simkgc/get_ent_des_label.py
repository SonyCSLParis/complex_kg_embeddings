# -*- coding: utf-8 -*-
"""
Get relations labels
"""
import os
import re
import json
import gzip
import pickle
import pandas as pd
from tqdm import tqdm
from urllib.parse import unquote

def open_gzip(fp):
    with gzip.open(fp, 'rb') as f:
        return pickle.load(f)
DES = open_gzip("./ilp/entity_descriptions.pkl.gz")

FOLDER_D = "./data/simkgc"
FOLDER_O = "./simkgc/data"

ENTITY_LIST_P = os.path.join(FOLDER_O, "entities.txt")
ENTITY_LABEL_P = os.path.join(FOLDER_O, "entities_label.json")
ENTITY_DES_P = os.path.join(FOLDER_O, "entities_description.json")


CACHE = {
    '<http://semanticweb.cs.vu.nl/2009/11/sem/Event>': "event",
    '<http://semanticweb.cs.vu.nl/2009/11/sem/hasActor>': "has actor",
}

def camel_to_readable(text):
    """
    Convert camelCase to space-separated words.
    Example: hasRole -> has role
    """
    # Insert space before capital letters and lowercase the result
    readable = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    # Also handle cases where multiple capitals appear together
    readable = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', readable)
    return readable.replace("_", " ")

def get_label(x):
    if x in CACHE:
        return CACHE[x]
    if "Statement#" in x:
        return "statement"
    if "hasActor#" in x:
        return "has actor"
    if "example.com" in x:
        return DES[x] \
            .replace("is an actor during an event", "") \
                .split("Document value")[0]
    if any(y in x for y in ["dbpedia.org", "/factbook/", "/abox/"]):
        x = x[1:-1].split('/')[-1]
        return unquote(camel_to_readable(x))
    raise ValueError(f"No label found for {x}")

def get_description(x):
    if x in DES:
        return DES[x]
    if "Statement#" in x:
        return "This node represents a reified statement that connects a subject, predicate, and object, allowing for the expression of metadata or additional context about a basic triple."
    if "hasActor" in x:
        return "has actor"
    raise ValueError(f"No description found for {x}")

def read_csv(file):
    df = pd.read_csv(file, sep=' ', header=None)
    df.columns = ["subject", "predicate", "object", "."]
    return df[df.columns[:3]]

def load_data():
    files = [x for x in os.listdir(FOLDER_D) if x.startswith("all")]
    files = [os.path.join(FOLDER_D, x, fn) for x in files for fn in os.listdir(os.path.join(FOLDER_D, x)) if fn.endswith(".csv")]
    dfs = [read_csv(file) for file in tqdm(files, desc="Reading CSV files")]
    return pd.concat(dfs, ignore_index=True)

def main():
    # retrieving entities
    if not os.path.exists(ENTITY_LIST_P):
        df = load_data()
        entities = list(set(df["subject"].unique()).union(set(df["object"].unique())))
        with open(ENTITY_LIST_P, "w", encoding="utf-8") as f:
            f.write("\n".join(entities))
    else:
        with open(ENTITY_LIST_P, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entities = [x.replace("\n", "") for x in lines]
    
    # retrieving labels
    if not os.path.exists(ENTITY_LABEL_P):
        labels = {}
    else:
        with open(ENTITY_LABEL_P, "r", encoding="utf-8") as f:
            labels = json.load(f)
    for ent in tqdm(entities, desc="Processing entities"):
        if ent not in labels:
            labels[ent] = get_label(ent)
    with open(ENTITY_LABEL_P, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)
    
    # retrieving descriptions
    if not os.path.exists(ENTITY_DES_P):
        descriptions = {}
    else:
        with open(ENTITY_DES_P, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
    for ent in tqdm(entities, desc="Processing descriptions"):
        if ent not in descriptions:
            descriptions[ent] = get_description(ent)
    with open(ENTITY_DES_P, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=4)

if __name__ == "__main__":
    main()
