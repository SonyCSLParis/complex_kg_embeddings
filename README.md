# From Symbols to Numbers: Measuring the Impact of Narrative Complexity on Embeddings

Code submitted with the paper titled "From Symbols to Numbers: Measuring the Impact of Narrative Complexity on Embeddings" to ISWC 2025.

In this paper, we investigate how narrative semantic and syntactic levels impact embedding performance. This repo contains all the code to construct the KGs and to prepare the data for training with the following three methods: ULTRA, SimKGC, ILP.

The `examples` folder contains 3 examples of KG constructed on revolutions. Upon acceptance of the paper, we will release all constructed KGs, as well as the resulting datasets for the experiments on Zenodo.

Detailed description of the content:

## Main
* `kg_construction` folder: all related to KG construction part. All files should be run from within that folder.
    * `causation.py`: extracting causation layer
    * `causation.sh`: extract causation narrative layer (script)
    * `config_complex.json`: config used for subevent extraction
    * `representations.py`: class for KG conversion
    * `convert_kg.py`: converting KG to different syntaxes (script)
    * `frames.py`: extracting role layer
    * `gsf.py`: extracting base, prop, subevent, text layers
    * `utils.py`: utils (generic)

* `prep_data.py`: preparing data (see file for arguments)
* `utils_inductive.py`: utils (inductive splits)


## Methods (ULTRA, SimKGC, ILP)

* `ultra` folder: all related to ULTRA
    * `convert_data.py`: convert data for ULTRA format
    * `convert_data.sh`: convert data for ULTRA format (script)
    * `prep_data.sh`: prep data for ULTRA

* `simkgc` folder: all related to SimKGC (also reuses descriptions obtained for ILP, see below)
    * `get_ent_des_label.py`: entity description
    * `get_relation_label.py`: relation description
    * `prep_data.sh`: prep data 

* `ilp` folder: all related to ILP 
    * `data`: cached values, including: frame and frame elements description from Framester, and DBpedia entities that have no label or description (in this case, we take the human-readable label from the IRI)
    * `extract_embeddings.py`: using a transformer-based model to extract embeddings from a list of descriptions
    * `add_missing_des.sh`: running `extract_embeddings.py` script for each file
    * `build_embeddings_ilp.py`: building `embeddings.pkl` for the embeddings model necessary as inputs for ILP
    * `build_index_ilp.py`: building `index.txt` for the embeddings model necessary as inputs for ILP
    * `get_bn_descriptions.py`: descriptions from blank nodes
    * `get_fe_descriptions.py`: descriptions for frame elements
    * `concat_embeddings.py`: updating cached embeddings for more efficiency
    * `run.py`: main, extracts all (data + descriptions)
    * `save_embeddings.sh`: save embeddings for ILP


## Other (analysis, tests, etc)
* `analysis` folder: all related to the analysis of the experiments
    * `analysis.ipynb`: statistical tests for syntax comparison
    * `analysis.py`: correlation text/metrics
    * `helpers_paper.py`: latex table formatting for the paper
    * `metrics_kg.py`: extracting metrics from KGs

* `stats` folder: all metrics from other repositories, to aggregate the results

* `tests` folder: various tests. All files should be run from within that folder.
    * `kg_test`folder: KG used for testing certain functions
    * `tests.py`: tests
* `revs_td.csv`: revolutiond data
