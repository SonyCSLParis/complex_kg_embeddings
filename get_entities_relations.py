# -*- coding: utf-8 -*-
"""
Get entities and relations (useful for some data format)
"""
import os
import json
import random
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


def save_info_hypermln(content: Set[str], save_p: str) -> None:
    """Save elements to a file, each line formatted as:
    
    {elt_id}\t{elt}
    
    Args:
        content (set[str]): A set of elements to save.
        save_p (str): Path to save the file.
    """
    with open(save_p, "w", encoding="utf-8") as file:
        file.write("\n".join(f"{i}\t{elt}" for i, elt in enumerate(content)) + "\n")

def save_info_kg_ilc(content: Set[str], save_p: str) -> None:
    """Save elements to a file, each line formatted as:
    
    {elt_id}\t{elt}
    
    Args:
        content (set[str]): A set of elements to save.
        save_p (str): Path to save the file.
    """
    with open(save_p, "w", encoding="utf-8") as file:
        file.write("\n".join(f"{elt}\t{i}" for i, elt in enumerate(content)) + "\n")


def save_info_hahe(entities: Set[str], relations: Set[str], save_p: str) -> None:
    """Save entities and relations to a file, formatted as:
    [PAD]
    [MASK]
    rel1
    ...
    reln
    ent1
    ...
    entm
    """
    with open(save_p, "w", encoding="utf-8") as file:
        file.write("[PAD]\n[MASK]\n")
        file.write("\n".join(relations) + "\n")
        file.write("\n".join(entities) + "\n")

def reformat_line_hahe(line: str) -> dict:
    """ Reformat a line from the HAHE dataset to be used in the HAHE model."""
    elements = line.split("\t")
    res = {
        "N": len(elements) -1,
        "relation": elements[0],
        "subject": elements[1],
        "object": elements[2]
    }
    for index, elt in enumerate(elements[3:]):
        res[f"{elements[0]}_{index}"] = [elt]
    return res



def reformat_data_hahe(folder: str) -> None:
    """Reformat data from the HAHE dataset to be used in the HAHE model."""
    all_data = set()
    for f in ["test", "train", "valid"]:
        f_p = os.path.join(folder, f"{f}.txt")
        with open(f_p, 'r', encoding="utf-8") as file:
            for line in file:
                all_data.add(line.strip())

    all_data = list(all_data)
    all_data = [reformat_line_hahe(line) for line in all_data]
    random.shuffle(all_data)
    num_data = len(all_data)
    train_data = all_data[:int(0.8 * num_data)]
    valid_data = all_data[int(0.8 * num_data):int(0.9 * num_data)]
    test_data = all_data[int(0.9 * num_data):]

    for name, data in [("train", train_data), ("valid", valid_data), ("test", test_data), ("all", all_data)]: 
        with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as file:
            for element in data:
                file.write(json.dumps(element) + "\n")

@click.command()
@click.argument("folder")
@click.argument("dataset",
                default="hypermln", type=click.Choice(["hypermln", "hahe", "kg_ilc"]))
@click.argument("extension", default="txt", type=click.Choice(["txt", "dict"]))
def main(folder, dataset, extension):
    """ main: get relations/entities and save them """
    entities, relations = extract_entity_relations(folder=folder)
    print(f"# Entities: {len(entities)} | # Relations: {len(relations)}")

    if dataset == "hypermln":
        save_info_hypermln(entities, os.path.join(folder, f"entities.{extension}"))
        save_info_hypermln(relations, os.path.join(folder, f"relations.{extension}"))
    
    if dataset == "kg_ilc":
        save_info_kg_ilc(entities, os.path.join(folder, f"entities.{extension}"))
        save_info_kg_ilc(relations, os.path.join(folder, f"relations.{extension}"))
    
    if dataset == "hahe":
        new_rel = {x for x in relations if ">_<" in x}
        relations.update({f"{x}_0" for x in new_rel})
        print(f"# Entities: {len(entities)} | # Relations (updated): {len(relations)}")
        save_info_hahe(entities, relations, os.path.join(folder, f"vocab.{extension}"))
        reformat_data_hahe(folder)

if __name__ == '__main__':
    main()