# -*- coding: utf-8 -*-
"""
Updating cache for various embeddings and descriptions.
"""
import os
import gzip
import pickle
import pandas as pd
from tqdm import tqdm

FOLDER_DES = "./ilp/data/"
FILES_DES = ["fe", "frames", "missing"]
FOLDER_EXP = "./exps"
CACHE_P = "./ilp/entity_embeddings.pkl.gz"

def get_des_from_csv(fp):
    df = pd.read_csv(fp)
    return list(df["description"].values)

def update_cache(cache, descriptions, embeddings):
    for i, des in enumerate(descriptions):
        cache[des] = embeddings[i,:]
    return cache

def main():
    cache = {}

    for fn in tqdm(FILES_DES):
        descriptions = get_des_from_csv(os.path.join(FOLDER_DES, f"{fn}_descriptions.csv"))
        with open(os.path.join(FOLDER_DES, f"{fn}_embeddings.pkl"), 'rb') as f:
            embeddings = pickle.load(f)
        cache = update_cache(cache, descriptions, embeddings)

    for exp in tqdm(os.listdir(FOLDER_EXP)):
        descriptions = get_des_from_csv(os.path.join(FOLDER_EXP, exp, "descriptions.csv"))
        with open(os.path.join(FOLDER_EXP, exp, "embeddings.pkl"), 'rb') as f:
            embeddings = pickle.load(f)
        cache = update_cache(cache, descriptions, embeddings)

    with gzip.open(CACHE_P, 'wb') as f:
        pickle.dump(cache, f)

if __name__ == '__main__':
   main()
