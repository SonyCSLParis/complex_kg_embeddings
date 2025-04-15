# -*- coding: utf-8 -*-
"""
Get relations labels
"""
import os
import re
import json
import pandas as pd
from tqdm import tqdm
from urllib.parse import unquote

CACHE = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "type",
    "http://www.w3.org/2004/02/skos/core#related": "related",
}

FOLDER_D = "./data/simkgc"
FOLDER_O = "./simkgc/data"

PREDICATE_LIST_P = os.path.join(FOLDER_O, "predicates.txt")
PREDICATE_LABEL_P = os.path.join(FOLDER_O, "predicates.json")

def camel_to_readable(text):
    """
    Convert camelCase to space-separated words.
    Example: hasRole -> has role
    """
    # Insert space before capital letters and lowercase the result
    readable = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    # Also handle cases where multiple capitals appear together
    readable = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', readable)
    return readable.lower().replace("_", " ")

def get_label(x):
    x = x[1:-1]
    if x in CACHE:
        return CACHE[x]
    x = x.split("/")[-1].split("#")[0]
    return unquote(camel_to_readable(x))
    


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
    # retrieving predicates
    if not os.path.exists(PREDICATE_LIST_P):
        df = load_data()
        with open(PREDICATE_LIST_P, "w", encoding="utf-8") as f:
            f.write("\n".join(df["predicate"].unique()))
        predicates = list(df["predicate"].unique())
    else:
        with open(PREDICATE_LIST_P, "r", encoding="utf-8") as f:
            lines = f.readlines()
        predicates = [x.replace("\n", "") for x in lines]
    
    # retrieving labels
    if not os.path.exists(PREDICATE_LABEL_P):
        labels = {}
    else:
        with open(PREDICATE_LABEL_P, "r", encoding="utf-8") as f:
            labels = json.load(f)
    for pred in tqdm(predicates, desc="Processing predicates"):
        if pred not in labels:
            labels[pred] = get_label(pred)
    with open(PREDICATE_LABEL_P, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)

if __name__ == "__main__":
    main()
