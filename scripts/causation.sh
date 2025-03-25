#!/bin/bash

# Check if input folder is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

input_folder="$1"

# Check if the input folder exists
if [ ! -d "$input_folder" ]; then
    echo "Error: Directory '$input_folder' does not exist."
    exit 1
fi

# Iterate over subdirectories
# Count total number of directories for progress bar
total_dirs=$(find "$input_folder" -maxdepth 1 -mindepth 1 -type d | wc -l)
current_dir=0

for dir in "$input_folder"/*/; do
    if [ -d "$dir" ]; then
        # Update progress
        current_dir=$((current_dir + 1))
        progress=$((current_dir * 100 / total_dirs))
        
        # Display progress bar
        bar_size=30
        filled_size=$((progress * bar_size / 100))
        bar=$(printf "%${filled_size}s" | tr " " "#")
        remaining_size=$((bar_size - filled_size))
        remaining=$(printf "%${remaining_size}s" | tr " " "-")
        
        dir_name=$(basename "$dir")
        echo "Progress: [${bar}${remaining}] ${progress}% - Processing: ${dir_name}\r"
        
        # Check for specific files
        for file in "kg_base_prop.nt" "kg_base_subevent_prop.nt" "kg_base_subevent.nt" "kg_base.nt"; do
            if [ -f "$dir/$file" ]; then
                echo "Processing ${file}"
                python causation.py "$input_folder/$dir_name" "$file"
            fi
        done
    fi
done

echo "Finished processing all subdirectories."