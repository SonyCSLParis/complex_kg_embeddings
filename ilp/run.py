# -*- coding: utf-8 -*-
"""
Preparing all data for ILP
"""
import os
import subprocess
import click
from loguru import logger
from sklearn.model_selection import ParameterGrid

FOLDER = "~/.data/ilp/NarrativeInductiveDataset/"
FOLDER = os.path.expanduser(FOLDER)

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

PARAM_GRID = {
    x: ["0", "1"] for x in ["prop", "subevent", "role", "causation"]
}
GRID = [x for x in list(ParameterGrid(PARAM_GRID)) if (x['role'] == "1" or x['causation'] == "1")]


@click.command()
def main():
    
    for param in GRID:
        logger.info(f"Params: {param}")
        fn = os.path.join(FOLDER, "inductive/statements", f"kg_base_prop_{param['prop']}_subevent_{param['subevent']}_role_{param['role']}_causation_{param['causation']}_syntax_hyper_relational_rdf_star")
        if not os.path.exists(fn):
            os.makedirs(fn)

        
        if not os.path.exists(os.path.join(fn, "transductive_train.txt")):
            logger.info(f"DOING -- Preparing data for {fn}")
            command = f"python prep_data.py {fn} --prop {param['prop']} --subevent {param['subevent']} --text 0 --role {param['role']} --causation {param['causation']} --syntax hyper_relational_rdf_star --inductive_split"
            logger.info(f"Running: {command}")
            subprocess.run(command, shell=True, check=False)
            subprocess.run("sh clean.sh", shell=True, check=False)
            logger.info(f"DONE -- Preparing data for {fn}")
        else:
            logger.info(f"SKIPPING -- {fn} data already prepared")

    # Building index for all
    if not os.path.exists(os.path.join(FOLDER, "index.txt")):
        folder_data = os.path.join(FOLDER, "inductive/statements", "kg_base_prop_1_subevent_1_role_1_causation_1_syntax_hyper_relational_rdf_star")
        index_path = os.path.join(FOLDER, "index.txt")
        logger.info(f"DOING -- Building index for {fn}")
        command = f"python ilp/build_index_ilp.py {folder_data} {index_path}"
        subprocess.run(command, shell=True, check=False)
        logger.info(f"DONE -- Building index for {fn}")
    else:
        logger.info(f"SKIPPING -- {fn} index already exists")
    
    # Building embeddings for all
    if not os.path.exists(os.path.join(FOLDER, "embeddings.pkl")):
        index_p = os.path.join(FOLDER, "index.txt")
        emb_p = os.path.join(FOLDER, "embeddings.pkl")
        logger.info(f"DOING -- Retrieving embeddings for {fn}")
        command = f"python ilp/build_embeddings_ilp.py {index_p} {emb_p} ilp/entity_descriptions.pkl.gz ilp/entity_embeddings.pkl.gz"
        subprocess.run(command, shell=True, check=False)
        logger.info(f"DONE -- Retrieving embeddings for {fn}")
    else:
        logger.info(f"SKIPPING -- {fn} embeddings already exist")


if __name__ == "__main__":
    # python ilp/run.py
    main()
