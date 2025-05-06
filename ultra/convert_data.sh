#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

input_folder=$1
output_folder=$2

# Check if input folder exists
if [ ! -d "$input_folder" ]; then
    echo "Error: Input folder '$input_folder' does not exist."
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$output_folder"

# Counter for processed files
processed=0

# Iterate over all .csv files in the input folder
for input_file in "$input_folder"/*.csv; do
    # Check if file exists (in case there are no .csv files)
    if [ ! -f "$input_file" ]; then
        echo "No .csv files found in '$input_folder'."
        exit 0
    fi
    
    # Get just the filename without the path and extension
    filename=$(basename "$input_file" .csv)
    
    # Construct output file path (change extension to .txt)
    output_file="$output_folder/${filename}.txt"
    
    echo "Processing: $(basename "$input_file") → ${filename}.txt"
    
    # Process the file using the same AWK script
    awk '
        !/^\/\// {
            # Remove trailing " ."
            sub(/ \.$/, "");
            
            # Find position of first space
            pos1 = index($0, " ");
            if (pos1 > 0) {
                # Extract part before first space
                part1 = substr($0, 1, pos1-1);
                
                # Find position of second space after the first space
                rest = substr($0, pos1+1);
                pos2 = index(rest, " ");
                if (pos2 > 0) {
                    # Extract middle part and ending part
                    part2 = substr(rest, 1, pos2-1);
                    part3 = substr(rest, pos2+1);
                    
                    # Print as tab-separated
                    print part1 "\t" part2 "\t" part3;
                }
            }
        }
    ' "$input_file" > "$output_file"
    
    processed=$((processed+1))
done

if [ $processed -gt 0 ]; then
    echo "Conversion complete. Processed $processed files to $output_folder/"
else
    echo "No .csv files found in '$input_folder'."
fi