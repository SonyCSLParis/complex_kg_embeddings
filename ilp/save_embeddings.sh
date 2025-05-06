#!/bin/bash

INPUT_FOLDER=$1

# Check if the input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: The directory '$INPUT_FOLDER' does not exist."
    exit 1
fi

echo "Input folder '$INPUT_FOLDER' exists. Proceeding..."

# Iterate over all subdirectories of the input folder
for DIR in "$INPUT_FOLDER"/*/; do
    if [ -d "$DIR" ]; then
        DIR_NAME=$(basename "$DIR")
        echo "Processing directory: $DIR_NAME"
        
        # Only run if embeddings.pkl does not exist
        if [ ! -f "$DIR/embeddings.pkl" ]; then
            echo "  Creating embeddings file: $DIR/embeddings.pkl"
            python ilp/extract_embeddings.py "$DIR/descriptions.csv" ilp/entity_embeddings.pkl.gz "$DIR/embeddings.pkl"
        else
            echo "  Embeddings file already exists: $DIR/embeddings.pkl"
        fi
    fi
done
