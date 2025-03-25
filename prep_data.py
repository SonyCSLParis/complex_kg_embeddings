# -*- coding: utf-8 -*-
"""
Prep data for embeddings
"""
import os
import csv
from urllib.parse import unquote
import click
import pandas as pd

REVS_TD = "./revs_td.csv"
EXP_F = "exps"
DATA_F = "data"
COLUMNS = ["subject", "predicate", "object", "."]
COLUMNS_TEXT = ["iri", "content"]
SAMPLE_F = "exps/Cambodian_Civil_War"

def filter_dates(df):
    """ rm timestamps """
    df.columns = COLUMNS
    return df[~df["object"].str.contains("XMLSchema#date")]

def filter_literals(df):
    """ rm literals """
    for col in df.columns:
        if not all(df[df.columns[3]]=="."):
            df = df[df[col].str.contains('http')]
    return df

def get_syntax_files(all_files, syntax: str, include: bool, elt: str, options):
    if include == "1":
        files = [x for x in all_files if f"{elt}_{syntax}" in x]
        for o, val in options.items():
            if not val:
                files = [x for x in files if not o in x.replace(f"{elt}_{syntax}", "")]
        return files
    return []

def get_files(options, role, causation, syntax):
    """ Get files to concat depending on options 
    Options: `prop`, `subevent`, `text` """
    all_files = [x for x in os.listdir(SAMPLE_F) if (x.endswith(".nt") or x.endswith(".txt"))]
    files_roles = get_syntax_files(all_files=all_files, syntax=syntax, include=role, elt="role", options=options)
    files_causation = get_syntax_files(all_files=all_files, syntax=syntax, include=causation, elt="causation", options=options)

    files_options = [x for x in all_files if not any(y in x for y in ["role", "causation"])]
    # Iterate over options
    for o, val in options.items():
        if not val:  # remove corresponding file
            files_options = [x for x in files_options if not o in x]

    return files_roles + files_options + files_causation


def prep_data_kg_only(df, file_names):
    """ Split data for KG files with no text """
    data = {x: [] for x in ["train", "valid", "test"]}
    for _, row in df.iterrows():
        for fn in file_names:
            kg_p = os.path.join(EXP_F, row.event.split('/')[-1], fn)
            if os.path.exists(kg_p):
                try:
                    curr_df = filter_dates(df=pd.read_csv(kg_p, sep=" ", header=None))
                    curr_df = filter_literals(df=curr_df)
                    data[row.td].append(curr_df)
                except pd.errors.EmptyDataError:
                    pass
    for key, val in data.items():
        df = pd.concat(val)
        df.columns = COLUMNS
        data[key] = df.drop_duplicates()
    return data


def format_line_hypergraph(line):
    """ format line to be compatible with data format for exps """
    elements = line.split(" ")
    relation = "_".join([x for i, x in enumerate(elements) if i%2 == 1])
    nodes = "\t".join([x for i, x in enumerate(elements) if i%2 == 0])
    return relation + "\t" + nodes


def prep_data_hypergraph(df, file_names):
    """ Prep data for hypergraphs """
    data = {x: [] for x in ["train", "valid", "test"]}
    for _, row in df.iterrows():
        for fn in file_names:
            kg_p = os.path.join(EXP_F, row.event.split('/')[-1], fn)
            if os.path.exists(kg_p):
                with open(kg_p, 'r', encoding='utf-8') as f:
                    for line in f:
                        cleaned_line = line.rstrip()
                        if '"' not in cleaned_line:
                            if cleaned_line.endswith(" ."):
                                cleaned_line = cleaned_line[:-2]
                            data[row.td].append(format_line_hypergraph(line=cleaned_line))
    for key, val in data.items():
        data[key] = [x for x in val if x.strip()]
    return data


def prep_data_rdf_star(df, file_names):
    """ Prep data for RDF* """
    data = {x: [] for x in ["train", "valid", "test"]}
    for _, row in df.iterrows():
        for fn in file_names:
            kg_p = os.path.join(EXP_F, row.event.split('/')[-1], fn)
            if os.path.exists(kg_p):
                if kg_p.endswith(".nt"):
                    try:
                        curr_df = filter_dates(df=pd.read_csv(kg_p, sep=" ", header=None))
                        curr_df = filter_literals(df=curr_df)
                        for _, curr_row in curr_df.iterrows():
                            data[row.td].append(f"{curr_row.predicate}\t{curr_row.subject}\t{curr_row.object}")
                    except pd.errors.EmptyDataError:
                        pass
                else:  # kg_p.endswith(".txt")
                    with open(kg_p, 'r', encoding='utf-8') as f:
                        for line in f:
                            cleaned_line = line.rstrip()[:-2].replace("<< ", "").replace(" >>", "")
                            if '"' not in cleaned_line:
                                data[row.td].append(format_line_hypergraph(line=cleaned_line))
    for key, val in data.items():
        data[key] = [x for x in val if x.strip()]
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


def remove_overlapping_data_df(data_dict):
    """
    Remove overlapping data between train, valid, and test DataFrames.
    
    Args:
        data_dict: Dictionary with keys 'train', 'valid', 'test' and DataFrame values
        
    Returns:
        Dictionary with filtered DataFrames
    """
    # Create copies to avoid modifying the original data
    filtered_data = {
        'train': data_dict['train'].copy(),
    }
    
    # Training set hash sets (faster lookup), hash keys = tuples of all columns
    train_set = set(map(tuple, filtered_data['train'].values))
    
    # Filter validation data to remove any rows found in training data
    valid_df = data_dict['valid'].copy()
    mask_valid = ~valid_df.apply(tuple, axis=1).isin(train_set)
    filtered_data['valid'] = valid_df[mask_valid].reset_index(drop=True)
    
    # Create hash set for validation data
    valid_set = set(map(tuple, filtered_data['valid'].values))
    
    # Filter test data to remove any rows found in training or validation data
    test_df = data_dict['test'].copy()
    combined_set = train_set.union(valid_set)
    mask_test = ~test_df.apply(tuple, axis=1).isin(combined_set)
    filtered_data['test'] = test_df[mask_test].reset_index(drop=True)
    
    # Report statistics
    original_counts = {k: len(v) for k, v in data_dict.items()}
    filtered_counts = {k: len(v) for k, v in filtered_data.items()}
    removed = {k: original_counts[k] - filtered_counts[k] for k in original_counts.keys()}
    
    print(f"Original counts: {original_counts}")
    print(f"Filtered counts: {filtered_counts}")
    print(f"Removed: {removed}")
    
    return filtered_data


def remove_overlapping_data_list(data_dict):
    """
    Remove overlapping strings between train, valid, and test lists.
    
    Args:
        data_dict: Dictionary with keys 'train', 'valid', 'test' and list of strings as values
        
    Returns:
        Dictionary with filtered lists where:
        - train remains unchanged
        - valid contains no strings from train
        - test contains no strings from train or valid
    """
    # Create copies to avoid modifying the original data
    filtered_data = {
        'train': data_dict['train'].copy()  # Train data remains unchanged
    }
    
    # Convert training data to a set for efficient lookups
    train_set = set(filtered_data['train'])
    
    # Filter validation data to remove any strings found in training
    valid_list = data_dict['valid'].copy()
    filtered_data['valid'] = [item for item in valid_list if item not in train_set]
    
    # Convert validation data to a set
    valid_set = set(filtered_data['valid'])
    
    # Filter test data to remove any strings found in training or validation
    test_list = data_dict['test'].copy()
    combined_set = train_set.union(valid_set)  # Combine both sets
    filtered_data['test'] = [item for item in test_list if item not in combined_set]
    
    # Report statistics
    original_counts = {k: len(v) for k, v in data_dict.items()}
    filtered_counts = {k: len(v) for k, v in filtered_data.items()}
    removed = {k: original_counts[k] - filtered_counts[k] for k in original_counts.keys()}
    
    print(f"Original counts: {original_counts}")
    print(f"Filtered counts: {filtered_counts}")
    print(f"Removed: {removed}")
    
    return filtered_data


@click.command()
@click.argument('save_fp')
@click.option("--prop", help="Whether to include properties in narratives", 
              default="0", type=click.Choice(["0", "1"]))
@click.option("--subevent", help="Whether to include subevent in narratives",
              default="0", type=click.Choice(["0", "1"]))
@click.option("--text", help="Whether to include text in narratives",
             default="0", type=click.Choice(["0", "1"]))
@click.option("--role", help="Whether to include roles in narratives",
              default="0", type=click.Choice(["0", "1"]))
@click.option("--causation", help="Whether to include causation in narratives",
              default="0", type=click.Choice(["0", "1"]))
@click.option("--syntax", help="Syntax to use for roles", default=None,
              type=click.Choice(["simple_rdf_prop", "hypergraph_bn", "hyper_relational_rdf_star", "simple_rdf_sp", "simple_rdf_reification"]))
#@click.pass_context
def main(save_fp, prop, subevent, text, role, causation, syntax):
    """ Main prep data """
    options = {"prop": int(prop), "subevent": int(subevent), "text": int(text)}
    if (role == "1" or causation == "1") and not syntax:
        raise click.BadParameter('If "--role" or "--causation" is provided, "--syntax" must also be provided.')
    if syntax == "hypergraph_bn" and text == "1":
        raise click.BadParameter("Text cannot be included if considering hypergraph syntax")
    files = get_files(options=options, role=role, causation=causation, syntax=syntax)
    print(files)
    if not os.path.exists(save_fp):
        os.makedirs(save_fp)
    if not options["text"]:  # KG only, no text
        if syntax == "hypergraph_bn":
            data = prep_data_hypergraph(df=pd.read_csv(REVS_TD, index_col=0), file_names=files)
            data = remove_overlapping_data_list(data_dict=data)
            for key, val in data.items():
                with open(os.path.join(save_fp, f"{key}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(val))
                f.close()
        elif syntax == "hyper_relational_rdf_star":
            data = prep_data_rdf_star(df=pd.read_csv(REVS_TD, index_col=0), file_names=files)
            data = remove_overlapping_data_list(data_dict=data)
            for key, val in data.items():
                with open(os.path.join(save_fp, f"{key}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(val))
                f.close()
        else:
            data = prep_data_kg_only(df=pd.read_csv(REVS_TD, index_col=0), file_names=files)
            data = remove_overlapping_data_df(data_dict=data)
            for key, val in data.items():
                val.to_csv(os.path.join(save_fp, f"{key}.csv"), header=None, index=False, sep=" ")


if __name__ == '__main__':
    # python prep_data.py ./data/kg_base
    # python prep_data.py ./data/kg_base_prop --prop 1
    # python prep_data.py ./data/kg_base_prop_role_simple_rdf_prop --prop 1 --role 1 --syntax simple_rdf_prop
    # python prep_data.py ./data/kg_base_prop_role_simple_rdf_sp --prop 1 --role 1 --syntax simple_rdf_sp
    # python prep_data.py ./data/kg_base_prop_role_simple_rdf_reification --prop 1 --role 1 --syntax simple_rdf_reification

    # python prep_data.py ./data/kg_base_subevent_prop_role_simple_rdf_prop --prop 1 --subevent 1 --role 1 --syntax simple_rdf_prop
    # python prep_data.py ./data/kg_base_subevent_prop_role_simple_rdf_sp --prop 1 --subevent 1 --role 1 --syntax simple_rdf_sp
    # python prep_data.py ./data/kg_base_subevent_prop_role_simple_rdf_reification --prop 1 --subevent 1 --role 1 --syntax simple_rdf_reification

    # python prep_data.py ./data/kg_base_prop_role_hypergraph_bn --prop 1 --role 1 --syntax hypergraph_bn

    # python prep_data.py ./data/kg_base_role_rdf_star --role 1 --syntax hyper_relational_rdf_star
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
#     #{"prop": 0, "subevent": 0, "text": 0},
#     #{"prop": 1, "subevent": 0, "text": 0},
#     # {"prop": 0, "subevent": 1, "text": 0},
#     # {"prop": 0, "subevent": 0, "text": 1},
#     # {"prop": 1, "subevent": 1, "text": 0},
#     {"prop": 1, "subevent": 0, "text": 1},
#     # {"prop": 0, "subevent": 1, "text": 1},
#     # {"prop": 1, "subevent": 1, "text": 1},
# ]
# for option in options:
#     print(option)
#     print(get_files(options=option, role=1, syntax="simple_rdf_prop"))
#     print("=====")