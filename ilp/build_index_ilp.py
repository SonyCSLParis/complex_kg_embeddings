# -*- coding: utf-8 -*-
"""
Building `index.txt` and `embeddings.pkl` for the embeddings model
"""
import os
import json
import gzip
import pickle
from urllib.parse import unquote
import click
from tqdm import tqdm
import polars as pl
import pandas as pd

FILES = ["inductive_train.txt", "inductive_ts.txt",
         "inductive_val.txt", "transductive_train.txt"]
ENTITY_DES_GZ = "./ilp/entity_descriptions.pkl.gz"
MISSING_DES_P = "./ilp/descriptions/missing_descriptions.csv"

if os.path.exists(MISSING_DES_P):
    MISSING_DES = pd.read_csv(MISSING_DES_P)
    MISSING_DES = MISSING_DES[[x for x in MISSING_DES.columns if x != "Unnamed: 0"]]
else:
    MISSING_DES = pd.DataFrame(columns=["node", "description"])

def concat_line(folder):
    """
    Concatenate lines from multiple files in a folder.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the files.
    
    Returns
    -------
    list
        All lines from files specified in FILES.
    """

    all_lines = []
    for fn in FILES:
        with open(os.path.join(folder, fn), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_lines.extend(lines)
    return [x.replace("\n", "") for x in all_lines]


def get_unique_entities(lines):
    """
    Extract unique entities from a list of comma-separated values.
    For each line in the input, entities are considered to be the items at even positions 
    (0, 2, 4, ...) when split by commas.
    Args:
        lines: List of strings where each string contains comma-separated values.
    Returns:
        A list of unique entities extracted from all lines.
    """

    entities = set()
    for line in lines:
        new_ents = set(x for i, x in enumerate(line.split(",")
) if i%2==0)
        entities.update(new_ents)
    return list(entities)


def write_index(entities, index_path):
    """
    Write unique entities to index.txt file.
    
    Parameters
    ----------
    entities : list
        List of unique entities to write.
    index_path : str
        Index file path.
    """
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(entities))


def concat_des(exp_folder, frame_des, fe_cache):
    """ concatenate descriptions from different sources """
    res = {}
    des = os.listdir(exp_folder)
    des = [os.path.join(exp_folder, x, "descriptions.csv") for x in des]
    for fp in tqdm(des):
        df = pl.read_csv(fp)
        res.update(dict(zip(df["node"], df["description"])))
    

    frames = pd.read_csv(frame_des).rename(columns={"frame": "node"})
    frames["node"] = frames["node"].apply(lambda x: f"<{x}>")
    res.update(dict(zip(frames["node"], frames["description"])))

    with open(fe_cache, 'r', encoding='utf-8') as f:
        fe = json.load(f)
    fe = pd.DataFrame([(f"<{k}>", val) for k, val in fe.items()], columns=["node", "description"])
    res.update(dict(zip(fe["node"], fe["description"])))

    res.update({
        "<http://semanticweb.cs.vu.nl/2009/11/sem/Actor>": "Actors are entities that take part in an Event, either actively or passively. Actors do not necessarily have to be sentient. They can also be objects. Actors are a thing, animate or inanimate, physical or non-physical.",
        "<http://semanticweb.cs.vu.nl/2009/11/sem/Role>": "Roles are properties with a subspecified function or position indicated by a RoleType in the scope of an Event.",
        "<http://semanticweb.cs.vu.nl/2009/11/sem/Event>": "Events are things that happen. This comprises everything from historical events to web site sessions and mythical journeys. Event is the central class of SEM.",
        "<http://semanticweb.cs.vu.nl/2009/11/sem/Place>": "Places are locations where an Event happens. They do not need to be physical places and hence do not necessarily need coordinates. Neither do they need to have any significance apart from them being the location of an Event." 
    })

    return res

def fix_missing_entity(input_set):
    """ fix missing entities by checking if they are in DBpedia or Factbook -> human-readable label"""
    res = {}
    for x in input_set:
        if any(y in x for y in ["dbpedia.org", "/factbook/"]):
            res[x] = unquote(x.split('/')[-1].replace("_", " ").replace(">", ""))
    return res


@click.command()
@click.argument("folder_data")
@click.argument("index_path")
@click.option("--exp_folder", default="exps", help="Path to the folder containing experiments")
@click.option("--frame_des", default="ilp/data/frames_descriptions.csv", help="Path to the frame descriptions file")
@click.option("--fe_cache", default="ilp/data/fe_cache.json", help="Path to the frame element cache file")
def main(folder_data, index_path, exp_folder, frame_des, fe_cache):
    if not os.path.exists(index_path):
        lines = concat_line(folder=folder_data)
        entities = get_unique_entities(lines)
        write_index(entities=entities, index_path=index_path)
        print(f"Index file written to {index_path}")
    else:
        with open(index_path, 'r', encoding='utf-8') as f:
            entities = [line.strip() for line in f.readlines()]
        print(f"Loaded entities from existing {index_path}")
    print(f"Total # of entities: {len(entities)}")

    des = concat_des(exp_folder=exp_folder, frame_des=frame_des, fe_cache=fe_cache)
    print(f"Total # of descriptions: {len(des)}")
    missing = set(entities).difference(set(des.keys()))
    print(f"Total # of missing entities: {len(missing)}")
    fixed = fix_missing_entity(missing)
    new_missing = pd.concat([
        MISSING_DES,
        pd.DataFrame([(k, v) for k, v in fixed.items()], columns=["node", "description"]).drop_duplicates().reset_index(drop=True)
    ])
    new_missing.to_csv(MISSING_DES_P)
    des.update(fixed)
    with gzip.open(ENTITY_DES_GZ, 'wb') as f:
        pickle.dump(des, f)
    missing = set(entities).difference(set(des.keys()))
    print(f"Total # of missing entities (after fix): {len(missing)}")
    print(missing)


if __name__ == "__main__":
    main()
