#!/bin/bash

csv_list="ilp/data/fe_descriptions.csv ilp/data/frames_descriptions.csv ilp/data/missing_descriptions.csv"
for csv in ${csv_list}; do
    dir=$(dirname "$csv")
    printf "\033[1;32m%-30s\033[0m %s\n" "Processing $csv" "at $(date)"
    base=$(basename "$csv" _descriptions.csv)
    
    python ilp/extract_embeddings.py "$csv" ilp/entity_embeddings.pkl.gz "$dir"/"$base"_embeddings.pkl
    printf "\033[1;32m%-30s\033[0m %s\n" "Finished processing $csv" "at $(date)"
done