# -*- coding: utf-8 -*-
"""
Building `index.txt` and `embeddings.pkl` for the embeddings model
"""
import os
import click
from utils import get_text

FILES = ["inductive_train.txt", "inductive_ts.txt", 
         "inductive_val.txt", "transductive_train.txt"]

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


def write_index(entities, folder):
    """
    Write unique entities to index.txt file.
    
    Parameters
    ----------
    entities : list
        List of unique entities to write.
    folder : str
        Path to the folder where the index.txt file will be saved.
    """
    
    with open(os.path.join(folder, "index.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(entities))


@click.command()
@click.argument("folder")
def main(folder):
    lines = concat_line(folder=folder)
    entities = get_unique_entities(lines)
    write_index(entities=entities, folder=folder)
    print(f"Index file written to {folder}/index.txt")


if __name__ == "__main__":
    main()
