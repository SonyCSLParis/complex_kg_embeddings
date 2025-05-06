# -*- coding: utf-8 -*-
"""
Generate data from ./data folder for ULTRA
"""
import os
import subprocess
import click
from loguru import logger

@click.command()
@click.argument("input_f")
@click.argument("output_f")
def main(input_f, output_f):
    """ Generate data for ULTRA """
    for data in [x for x in os.listdir(input_f) if os.path.isdir(os.path.join(input_f, x))]:
        logger.info(f"Processing {data}")
        for path in [os.path.join(output_f, data),
                     os.path.join(output_f, data, "raw"),
                     os.path.join(output_f, data, "processed")]:
            if not os.path.exists(path):
                os.makedirs(path)
        command = f"sh ultra/convert_data.sh {os.path.join(input_f, data)} {os.path.join(output_f, data, 'raw')}"
        subprocess.run(command, shell=True, check=False)


if __name__ == '__main__':
    # python ultra/convert_data.py ~/git/ULTRA/kg-datasets/NarrativeInductiveDataset ~/git/ULTRA/kg-datasets/NarrativeInductiveDataset
    main()
