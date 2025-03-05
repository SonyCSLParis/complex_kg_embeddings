# -*- coding: utf-8 -*-
"""
Prep data for embeddings
"""
import os
import csv
from urllib.parse import unquote
import click
import pandas as pd

REVS_TD = "./Rev/revs_td.csv"
EXP_F = "ChronoGrapher/exps"
DATA_F = "ChronoGrapher/data"
COLUMNS = ["subject", "predicate", "object", "."]
COLUMNS_TEXT = ["iri", "content"]
SAMPLE_F = "ChronoGrapher/exps/Cambodian_Civil_War"

def filter_dates(df):
    """ rm timestamps """
    df.columns = COLUMNS
    return df[~df["object"].str.contains("XMLSchema#date")]


def get_files(options):
    """ Get files to concat depending on options """
    all_files = [x for x in os.listdir(SAMPLE_F) if x.endswith(".nt")]
    # Iterate over options
    for o, val in options.items():
        if not val:  # remove corresponding file
            all_files = [x for x in all_files if not o in x]
    return all_files


def prep_data_kg_only(df, file_names):
    """ Split data for KG files with no text """
    data = {x: [] for x in ["train", "valid", "test"]}
    for _, row in df.iterrows():
        for fn in file_names:
            try:
                kg_p = os.path.join(EXP_F, row.event.split('/')[-1], fn)
                curr_df = filter_dates(df=pd.read_csv(kg_p, sep=" ", header=None))
                data[row.td].append(curr_df)
            except pd.errors.EmptyDataError:
                pass
    for key, val in data.items():
        df = pd.concat(val)
        df.columns = COLUMNS
        data[key] = df.drop_duplicates()
    return data


def update_w_missing_nodes(df, nodes):
    """ update descriptions and names with missing info """
    df.columns = COLUMNS_TEXT
    missing_des = nodes.difference(set(df["iri"].unique()))
    curr_df = pd.DataFrame(
        data=[(node, unquote(node.split('/')[-1]) \
            .replace("_", "").replace(">", "")) for node in missing_des],
        columns=COLUMNS_TEXT)
    df = pd.concat([df, curr_df])
    df["content"] = df["content"].apply(lambda x: x.replace('"', '').replace("\n", " "))
    return df


def prep_data_kg_text(df, file_name):
    """ Split data for baseKG embedding """
    data = {x: [] for x in ["train", "valid", "test"]}
    pred_des = "<http://dbpedia.org/ontology/abstract>"
    pred_name = "<http://www.w3.org/2000/01/rdf-schema#label>"
    des, names = [], []
    for _, row in df.iterrows():
        try:
            kg_p = os.path.join(EXP_F, row.event.split('/')[-1], file_name)
            curr_df = filter_dates(df=pd.read_csv(kg_p, sep=" ", header=None))
            des += curr_df[curr_df.predicate == pred_des][["subject", "object"]].values.tolist()
            names += curr_df[curr_df.predicate == pred_name][["subject", "object"]].values.tolist()
            data[row.td].append(curr_df[~curr_df.predicate.isin([pred_des, pred_name])])
        except pd.errors.EmptyDataError:
            pass
    for key, val in data.items():
        df = pd.concat(val)
        df.columns = COLUMNS
        data[key] = df

    all_data = pd.concat([val for _, val in data.items()])
    des, names = pd.DataFrame(des), pd.DataFrame(names)
    nodes = set(all_data["subject"].unique()).union(set(all_data["object"].unique()))
    return data, update_w_missing_nodes(des, nodes), update_w_missing_nodes(names, nodes)


@click.command()
@click.argument('save_fp')
@click.option("--prop", help="Whether to include properties in narratives", default=0)
@click.option("--subevent", help="Whether to include subevent in narratives", default=0)
@click.option("--text", help="Whether to include text in narratives", default=0)
def main(save_fp, prop, subevent, text):
    """ Main prep data """
    options = {"prop": int(prop), "subevent": int(subevent), "text": int(text)}
    files = get_files(options=options)
    if not os.path.exists(save_fp):
        os.makedirs(save_fp)
    if not options["text"]:  # KG only, no text
        data = prep_data_kg_only(df=pd.read_csv(REVS_TD, index_col=0), file_names=files)
        for key, val in data.items():
            val.to_csv(os.path.join(save_fp, f"{key}.csv"), header=None, index=False, sep=" ")


if __name__ == '__main__':
    # python ChronoGrapher/prep_data.py ./ChronoGrapher/data/kg_base
    # python ChronoGrapher/prep_data.py ./ChronoGrapher/data/kg_base_prop --prop 1
    main()

# DATA_BASE_F = os.path.join(DATA_F, "basekg")
# if not os.path.exists(DATA_BASE_F):
#     os.makedirs(DATA_BASE_F)
# data = prep_data_kg_only(df=pd.read_csv(REVS_TD, index_col=0), file_name="base_kg.nt")
# for key, val in data.items():
#     val.to_csv(os.path.join(DATA_BASE_F, f"{key}.csv"), header=None, index=False, sep=" ")

# DATA_BASE_PROP_F = os.path.join(DATA_F, "basekg_prop")
# if not os.path.exists(DATA_BASE_PROP_F):
#     os.makedirs(DATA_BASE_PROP_F)
# data = prep_data_kg_only(df=pd.read_csv(REVS_TD, index_col=0), file_name="base_kg_prop.nt")
# for key, val in data.items():
#     val.to_csv(os.path.join(DATA_BASE_PROP_F, f"{key}.csv"), header=None, index=False, sep=" ")

# DATA_BASE_TEXT_F = os.path.join(DATA_F, "basekg_text")
# if not os.path.exists(DATA_BASE_TEXT_F):
#     os.makedirs(DATA_BASE_TEXT_F)
# data, des, names = prep_data_kg_text(df=pd.read_csv(REVS_TD, index_col=0), file_name="base_kg_text.nt")
# for key, val in data.items():
#     val[COLUMNS[:3]].to_csv(os.path.join(DATA_BASE_TEXT_F, f"{key}.txt"), header=None, index=False, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
# des.to_csv(os.path.join(DATA_BASE_TEXT_F, "FB15k_mid2description.txt"), header=None, index=False, sep="\t")
# names.to_csv(os.path.join(DATA_BASE_TEXT_F, "FB15k_mid2name.txt"), header=None, index=False, sep="\t")

# get_files(options={})

# options = [
#     {"prop": 0, "subevent": 0, "text": 0},
#     {"prop": 1, "subevent": 0, "text": 0},
#     {"prop": 0, "subevent": 1, "text": 0},
#     {"prop": 0, "subevent": 0, "text": 1},
#     {"prop": 1, "subevent": 1, "text": 0},
#     {"prop": 1, "subevent": 0, "text": 1},
#     {"prop": 0, "subevent": 1, "text": 1},
#     {"prop": 1, "subevent": 1, "text": 1},
# ]
# for option in options:
#     print(option)
#     get_files(options=option)
#     print("=====")