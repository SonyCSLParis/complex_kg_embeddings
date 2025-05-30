# -*- coding: utf-8 -*-
"""
Building `embeddings.pkl` for the embeddings model
"""
import gzip
import pickle
import click
from tqdm import tqdm
import numpy as np
from loguru import logger


def open_gzip(fp):
    with gzip.open(fp, 'rb') as f:
        return pickle.load(f)

@click.command()
@click.argument("index_fp", type=click.Path(exists=True))
@click.argument("embeddings_fp", type=click.Path(exists=False))
@click.argument("entity_des_cache", type=click.Path(exists=True))
@click.argument("entity_embed_cache", type=click.Path(exists=True))
def main(index_fp, embeddings_fp, entity_des_cache, entity_embed_cache):
    with open(index_fp, 'r', encoding='utf-8') as f:
        entities = [line.strip() for line in f.readlines()]
    logger.info(f"# of entities: {len(entities)}")
    
    des_cache = open_gzip(entity_des_cache)
    descriptions = [des_cache[x] for x in entities]
    logger.info(f"# of descriptions: {len(entities)}")

    embed_cache = open_gzip(entity_embed_cache)
    embeddings = np.array([embed_cache[x] for x in descriptions])
    logger.info(f"Embedding size: {embeddings.shape}")
    
    with open(embeddings_fp, 'wb') as f:
        pickle.dump(embeddings, f)
    
    


if __name__ == "__main__":
    main()
