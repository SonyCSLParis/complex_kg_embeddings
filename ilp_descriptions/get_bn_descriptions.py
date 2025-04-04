# -*- coding: utf-8 -*-
"""
Get templated descriptions from BN event and actors
"""
import re
import os
from tqdm import tqdm
import click
import pandas as pd
from urllib.parse import unquote

COLUMNS_OUT = ["node", "description"]
COLUMNS_RDF = ["subject", "predicate", "object", "."]
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
WSJ = "https://w3id.org/framester/wsj/"

def parse_nt_file(file_path):
    triples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check if there's a string literal (value between quotes)
            if ' "' in line:
                # Handle lines with string literals
                subject, predicate, rest = line.split(' ', 2)
                
                # Extract the value between quotes, handling escaped quotes
                match = re.match(r'"(.*?)(?<!\\)"', rest)
                if match:
                    value = match.group(1)
                    # Get the final period (or anything after the closing quote)
                    remainder = rest[match.end():].strip()
                    triples.append([subject, predicate, value, remainder])
            else:
                # Handle standard triples (no string literals)
                parts = line.split(' ', 3)
                # Make sure we have exactly 4 parts (including the period)
                if len(parts) == 3 and parts[2].endswith(' .'):
                    # Split the last part to separate object from period
                    object_part = parts[2][:-2]
                    triples.append([parts[0], parts[1], object_part, '.'])
                else:
                    # Fallback for other cases
                    parts = line.split(' ')
                    if len(parts) >= 4:
                        triples.append([parts[0], parts[1], parts[2], parts[3]])
    
    # Create DataFrame
    df = pd.DataFrame(triples, columns=['subject', 'predicate', 'object', '.'])
    return df

def pretty_print(x):
    return unquote(x.split("/")[-1].replace(">", "").replace("_", " "))

def get_actor_description(role, et, value, ents):
    output = f"Has the role of {pretty_print(role)} during an event of type {pretty_print(et)}. Has value '{pretty_print(value)}'"
    if ents:
        if len(ents) == 1:
            output += f" that is related to {pretty_print(ents[0])}."
        else:
            ents_text =  ', '.join([pretty_print(x) for x in ents[:-1]]) + " and " + pretty_print(ents[-1])
            output += f" that is related to {ents_text}."
    else:
        output += "."
    return output

def get_event_description(lemma, et, doc_value):
    return f"Event of type {pretty_print(et)} triggered from lemma '{pretty_print(lemma)}'. Document value: '{pretty_print(doc_value)}'."

def get_event_type(df, actor):
    event = df[(df["predicate"] == f"<{WSJ}withmappedrole>") & (df["object"] == actor)]["subject"].values[0]
    event_type = df[(df["predicate"] == f"<{WSJ}onFrame>") & (df["subject"] == event)]["object"].values[0]
    return event_type

def get_actor_info(df, actor):
    subdf = df[df["subject"] == actor]
    role = subdf[subdf["predicate"].str.contains("withfnfe")]["object"].values[0]
    value = subdf[subdf["predicate"].str.contains("#value")]["object"].values[0]
    ents = list(subdf[subdf["predicate"].str.contains("#related")]["object"].values)
    return role, value, ents

def get_event_info(df, event):
    subdf = df[df["subject"] == event]
    lemma = subdf[subdf["predicate"].str.contains("onLemma")]["object"].values[0]
    et = subdf[subdf["predicate"].str.contains("onFrame")]["object"].values[0]
    doc = subdf[subdf["predicate"].str.contains("fromDocument")]["object"].values[0]
    doc_value = df[(df["subject"] == doc) & (df["predicate"].str.contains("#value"))]["object"].values[0]
    return lemma, et, doc_value


def get_type(df, ct):
    return df[(df["predicate"] == RDF_TYPE) & (df["object"] == ct)]["subject"].values

def get_info_file(fp):
    # try:
    #     df = pd.read_csv(fp, sep=" ", header=None)
    #     df.columns = COLUMNS_RDF
        
    # except pd.errors.EmptyDataError:
    #     return pd.DataFrame(columns=COLUMNS_OUT)
    df = parse_nt_file(fp)
    
    data = []
    # actors
    actors = get_type(df, "<https://w3id.org/framester/wsj/MappedRole>")
    for actor in actors:
        et = get_event_type(df, actor)
        role, value, ents = get_actor_info(df, actor)
        des = get_actor_description(role, et, value, ents)
        data.append((actor, des))
        
    # events
    events = get_type(df, "<https://w3id.org/framester/wsj/CorpusEntry>")
    for event in events:
        lemma, et, doc_value = get_event_info(df, event)
        des = get_event_description(lemma, et, doc_value)
        data.append((event, des))
    
    res = pd.DataFrame(data, columns=COLUMNS_OUT).drop_duplicates()
    return res

@click.command()
@click.argument("folder")
def main(folder):
    events = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
    #to-do: remove when tested
    events = [x for x in events if "French_Revolution" == x]
    for event in tqdm(events):
        curr_folder = os.path.join(folder, event)
        df_to_concat = []
        files = [x for x in os.listdir(curr_folder) if x.endswith("_text.nt")]
        for file in files:
            print("FILE: ", file)
            res = get_info_file(os.path.join(curr_folder, file))
            df_to_concat.append(res)
        df_all = pd.concat(df_to_concat)
        print(df_all)


if __name__ == '__main__':
    main()


