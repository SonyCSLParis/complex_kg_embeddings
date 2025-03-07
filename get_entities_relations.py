# -*- coding: utf-8 -*-
"""
Get entities and relations (useful for some data format)
"""
import os
from typing import Set
import click

def extract_entity_relations(folder):
    """ Extract entities and relations from train, valid, and test files in a folder.
    as of now (march 6): hypergraph 
    Returns:
        tuple: (entities set, relations set)"""
    entities, relations = set(), set()
    for f in ["test", "train", "valid"]:
        f_p = os.path.join(folder, f"{f}.txt")
        with open(f_p, 'r', encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if parts:
                    relations.add(parts[0])
                    entities.update(parts[1:])

    return entities, relations


def save_info(content: Set[str], save_p: str) -> None:
    """Save elements to a file, each line formatted as:
    
    {elt_id}\t{elt}
    
    Args:
        content (set[str]): A set of elements to save.
        save_p (str): Path to save the file.
    """
    with open(save_p, "w", encoding="utf-8") as file:
        file.write("\n".join(f"{i}\t{elt}" for i, elt in enumerate(content)) + "\n")


@click.command()
@click.argument("folder")
def main(folder):
    """ main: get relations/entities and save them """
    entities, relations = extract_entity_relations(folder=folder)
    save_info(entities, os.path.join(folder, "entities.dict"))
    save_info(relations, os.path.join(folder, "relations.dict"))

if __name__ == '__main__':
    main()