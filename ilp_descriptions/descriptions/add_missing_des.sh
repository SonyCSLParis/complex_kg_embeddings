#!/bin/bash

csv_list="ilp_descriptions/descriptions/fe_descriptions.csv ilp_descriptions/descriptions/frames_descriptions.csv ilp_descriptions/descriptions/missing_descriptions.csv"
for csv in ${csv_list}; do
    dir=$(dirname "$csv")
    printf "\033[1;32m%-30s\033[0m %s\n" "Processing $csv" "at $(date)"
    base=$(basename "$csv" _descriptions.csv)
    
    python ilp_descriptions/extract_embeddings.py "$csv" ilp_descriptions/entity_embeddings.pkl.gz "$dir"/"$base"_embeddings.pkl
    printf "\033[1;32m%-30s\033[0m %s\n" "Finished processing $csv" "at $(date)"
done