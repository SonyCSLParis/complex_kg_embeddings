# -*- coding: utf-8 -*-
"""
Inspect frames extracted from DBpedia abstracts
"""
import os
import click
from tqdm import tqdm
import pandas as pd
from rdflib import Graph

PREFIX_WSJ = "https://w3id.org/framester/wsj/"

QUERY_FRAME = """
PREFIX wsj: <""" + PREFIX_WSJ + """>
SELECT DISTINCT ?frame_inst ?frame ?role ?role_type WHERE {
    ?frame_inst wsj:onFrame ?frame ;
                wsj:withmappedrole ?role .
    ?role wsj:withfnfe ?role_type .
}
"""
COLUMNS_FRAME = [x.replace('?', '') for x in \
    QUERY_FRAME.split(' WHERE ', maxsplit=1)[0].split(' ') if x.startswith('?')]

@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.option('--kg_fn', default='kg_base_prop_role_text.nt',
              help='Name of the KG file (output of ChronoGrapher)')
@click.option('--output', default='analysis/frames.csv',
              help='Name of the output csv file')
def main(folder, kg_fn, output):
    """
    Process knowledge graphs from experiments and extract frame information.
    This function iterates through experiment directories, parses knowledge graph files,
    queries them for frame information, and combines the results into a single DataFrame
    that is then saved to a CSV file.
    Args:
        folder (str): Path to the directory containing experiment folders
        kg_fn (str): Name of the knowledge graph file within each experiment folder
        output (str): Path where the resulting CSV file should be saved
    Returns:
        None: The function saves the results to the specified output file
    Note:
        - Currently processes only the first 2 experiment folders (limited by [:2])
        - Requires COLUMNS_FRAME and QUERY_FRAME to be defined in the global scope
        - Assumes knowledge graph files are in N-Triples format
    """

    df = pd.DataFrame(columns=COLUMNS_FRAME)
    for exp in tqdm(os.listdir(folder)):
        kg_p = os.path.join(folder, exp, kg_fn)
        graph = Graph()
        graph.parse(kg_p, format='nt')
        graph.query(QUERY_FRAME)
        curr_df = pd.DataFrame(
            data=graph.query(QUERY_FRAME), columns=COLUMNS_FRAME)
        df = pd.concat([df, curr_df], ignore_index=True)
    df.to_csv(output)


if __name__ == '__main__':
    main()
