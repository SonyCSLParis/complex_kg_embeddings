# -*- coding: utf-8 -*-
"""
Using huggingface sentence-transformer model to extract embeddings of nodes' descriptions
"""
import os
import gzip
import pickle
import click
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process, set_start_method, Queue

# Try to set multiprocessing start method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

class DescriptionDataset(Dataset):
    """ Simple dataset class for loading descriptions """
    def __init__(self, descriptions):
        self.descriptions = descriptions

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.descriptions[idx]


def process_chunk(gpu_id, chunk, batch_size, model_name, queue):
    torch.cuda.set_device(gpu_id)
    model = SentenceTransformer(model_name)
    model = model.to(f"cuda:{gpu_id}")
    
    # Create DataLoader for batching
    dataset = DescriptionDataset(chunk)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Process batches with progress bar
    chunk_emb = []
    for batch in tqdm(dataloader, desc=f"GPU {gpu_id}"):
        emb = model.encode(batch, show_progress_bar=False)
        chunk_emb.append(emb)
    
    # Store result from this GPU
    embeddings = np.vstack(chunk_emb)
    if queue is not None:
        return queue.put((gpu_id, embeddings))
    return embeddings

def load_cache(cache_p):
    if os.path.exists(cache_p):
        with gzip.open(cache_p, 'rb') as f:
            return pickle.load(f)
    return {}

def extract_embeddings(descriptions, cache_file, batch_size=32, model_name="sentence-transformers/all-roberta-large-v1"):
    """
    Extract embeddings using multiple GPUs if available
    
    Args:
        descriptions: List of text descriptions
        batch_size: Batch size for processing per GPU
        model_name: Name of the sentence transformer model
    
    Returns:
        numpy.ndarray of shape (n, 1024) containing the embeddings
    """
    cache = load_cache(cache_file)

    desc_to_process = []
    cached_indices = []

    for i, desc in enumerate(descriptions):
        if desc in cache:
            cached_indices.append(i)
        else:
            desc_to_process.append(desc)
    
    if not desc_to_process:
        print("All descriptions found in cache, no need for computation")
        return np.array([cache[desc] for desc in descriptions])

    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 0:
        print("No GPUs available, falling back to CPU")
        model = SentenceTransformer(model_name)
        return model.encode(desc_to_process, batch_size=batch_size, show_progress_bar=True)
    
    elif num_gpus == 1:
        print("Using single GPU")
        return process_chunk(0, desc_to_process, batch_size, model_name, None)
    
    else:  # multiple GPUs
        print(f"Using {num_gpus} GPUs")
        chunks = np.array_split(desc_to_process, num_gpus)
        processes = []
        result_queue = Queue()
    
        # Start processes for each GPU
        for gpu_id, chunk in enumerate(chunks):
            p = Process(target=process_chunk, args=(gpu_id, chunk, batch_size, model_name, result_queue))
            p.start()
            processes.append(p)
        
        # Collect results from the queue
        results = {}
        for _ in range(len(processes)):
            gpu_id, emb = result_queue.get()
            results[gpu_id] = emb
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results from all GPUs
        all_embeddings = [results[i] for i in range(num_gpus)]
        new_embeddings = np.vstack(all_embeddings)
    
        for i, desc in enumerate(desc_to_process):
            cache[desc] = new_embeddings[i]
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        return np.array([cache[desc] for desc in descriptions])


def save_embeddings(embeddings, output_file):
    """
    Save embeddings to a pickle file.
    
    Args:
        embeddings: numpy.ndarray of embeddings
        output_file: Path to output file
    """
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {output_file}")


@click.command()
@click.argument("des_csv")
@click.argument("des_gz")
@click.argument("des_pkl")
@click.option("--batch-size", default=32, help="Batch size for processing")
@click.option("--model", default="sentence-transformers/all-roberta-large-v1", help="Model name")
def main(des_csv, des_gz, des_pkl, batch_size, model):
    """
    Main function to extract embeddings from descriptions in CSV file.
    
    Args:
        des_csv: Path to CSV file containing descriptions
        batch_size: Batch size for processing
        model: Name of the sentence transformer model
    """
    descriptions = pd.read_csv(des_csv, index_col=0)["description"].values
    
    embeddings = extract_embeddings(
        descriptions,
        cache_file=des_gz,
        batch_size=batch_size,
        model_name=model
    )
    
    save_embeddings(embeddings, des_pkl)


if __name__ == "__main__":
    main()
