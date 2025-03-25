# -*- coding: utf-8 -*-
"""
Building KG that adds causal information based on frames and dbp:results in DBpedia
"""
import os
from typing import List
from urllib.parse import quote
import click
from tqdm import tqdm
import pandas as pd
from loguru import logger
from rdflib import URIRef, Graph, Literal
from frame_semantic_transformer.FrameSemanticTransformer import DetectFramesResult
from kglab.helpers.variables import NS_EX
from src.build_ng.frame_semantics import FrameSemanticsNGBuilder
from utils import read_nt, run_request, init_graph, save_pickle, load_pickle_cache
from frames import PREFIX_TO_NS
from representations import KGRepresentationsConverter

class FrameSemanticsNGBuilderCache(FrameSemanticsNGBuilder):
    """
    A specialized version of FrameSemanticsNGBuilder that includes caching 
    functionality for frame detection. This class extends the base FrameSemanticsNGBuilder 
    to add caching capabilities when processing text inputs.
    It generates RDF graphs that represent frame semantic structures from text, but uses 
    a cache to avoid 
    redundant frame detection operations for previously processed text.
    Attributes:
        ex (str): The example.com namespace prefix used for custom predicates.
        rdfs (str): The RDF Schema namespace prefix.
    Methods:
        __call__(text_input, id_abstract, cache, event, add_text=True): Processes text input 
        and generates an RDF graph representing its frame semantics structure, using cache 
        for previously analyzed texts.
        add_no_frame_outcome(graph, event, id_abstract, text_input, add_text): Adds an outcome 
        to an event in the RDF graph without using a frame structure.
    """
    def __init__(self):
        super().__init__()
        self.ex = "http://example.com/"
        self.rdfs = "http://www.w3.org/2000/01/rdf-schema#"

    def __call__(self, text_input, id_abstract, cache, event, add_text: bool = True) -> Graph:
        # Init graph with phrases+sentences+binding
        graph = init_graph(prefix_to_ns=self.prefix_to_ns)
        doc = self.nlp(text_input)
        graph = self.add_nif_phrase_sent(graph=graph, doc=doc, id_abstract=id_abstract)

        # DBpedia Spotlight entities
        ents = [ent._.dbpedia_raw_result for ent in doc.ents if ent._.dbpedia_raw_result]
        surf_to_ent = {x['@surfaceForm']: x['@URI'] for x in ents}

        # Get frames
        if text_input in cache:
            results = cache[text_input]
        else:
            results = self.frame_transformer.detect_frames(text_input)

        # Add frame info - minimal
        graph = self.add_no_frame_outcome(graph=graph, event=event, id_abstract=id_abstract, text_input=text_input, add_text=add_text)
        # Add frame info - if at least one frame
        if len(results.frames) > 0:  
            graph = self.add_frame(graph=graph,
                                   result=results, doc=doc, surf_to_ent=surf_to_ent, id_sent=f"{id_abstract}_0")

        return graph

    def add_no_frame_outcome(self, graph, event, id_abstract, text_input, add_text) -> Graph:
        """
        Adds an outcome to an event in the RDF graph without using a frame.
        The outcome had no identified frames.
        Args:
            graph: The RDF graph to add the outcome to.
            event: URI reference of the event.
            id_abstract: URI reference for the outcome.
            text_input: Text label/abstract for the outcome.
            add_text: Boolean flag indicating whether to add text labels to the graph. Default is True.
        Returns:
            Graph: The modified RDF graph with the added outcome.
        """

        graph.add((URIRef(event), URIRef(f"{self.ex}hasOutcome"), NS_EX[quote(id_abstract)]))
        if add_text:
            graph.add((NS_EX[quote(id_abstract)], URIRef(f"{self.rdfs}label"), Literal(text_input)))
            graph.add((NS_EX[quote(id_abstract)], URIRef(f"{self.ex}abstract"), Literal(text_input)))
        return graph


RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
EVENT = "<http://semanticweb.cs.vu.nl/2009/11/sem/Event>"

HEADERS_CSV = {"Accept": "text/csv"}
DB_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
QUERY_CAUSATION = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX yago: <http://dbpedia.org/class/yago/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX dbo: <http://dbpedia.org/ontology/>
select ?subject ?result where {
?subject dbp:result ?result .
VALUES ?subject {to-replace}
}
"""

CAUSATION_TEXT_CACHE_P = "./cache/causation_text_cache.csv"
CAUSATION_GRAPH_CACHE_P = "./cache/cause_graphs"
FRAME_CAUSATION_CACHE_P = "./cache/frame_causation_cache.pkl"
FS_KG_BUILDER = FrameSemanticsNGBuilderCache()

def get_events(graph_p: str) -> List[str]:
    """ Get all events from the graph encoded as .nt

    Args:
        graph_p (str): The path to the graph file in .nt format.

    Returns:
        List[str]: A list of all events in the graph.

    """
    df = read_nt(file_p=graph_p)
    return(df[(df.predicate == RDF_TYPE) & (df.object == EVENT)]["subject"].values.tolist())


def get_cache() -> pd.DataFrame:
    """ Get cache of causation text.

    Returns:
        pd.DataFrame: The cache of causation text, with columns "subject" and "result".
    """
    cols = ["subject", "result"]
    if os.path.exists(CAUSATION_TEXT_CACHE_P):
        return pd.read_csv(CAUSATION_TEXT_CACHE_P)
    return pd.DataFrame(columns=cols)


def extract_outcomes(res: str):
    """ 
    Extract all outcomes from one event.

    Args:
        res (str): The input string containing the outcomes.

    Returns:
        list: A list of extracted outcomes.
    """
    outcomes = [x.strip() for x in str(res).replace("@en", "").replace("*", "").split('\n') if x]
    outcomes = [x for x in outcomes if (not x == "nan" and not x[0].isdigit())]
    outcomes = [y.strip() for x in outcomes for y in x.split(". ")]
    return outcomes


def get_frames(results):
    """
    Process input sentences to detect frames using the frame transformer.
    
    This function first checks a local cache for previously processed sentences,
    then only processes new sentences that are not in the cache. The results
    are merged and saved back to cache.
    
    Parameters
    ----------
    results : list
        List of sentences to process for frame detection.
        
    Returns
    -------
    dict
        Dictionary mapping each sentence to its detected frame object.
    """
    cache = load_pickle_cache(file_p=FRAME_CAUSATION_CACHE_P, return_val={})
    results = [x for x in results if x not in cache]
    outcome_ft = FS_KG_BUILDER.frame_transformer.detect_frames_bulk(sentences=results)

    for elt in outcome_ft:
        cache[elt.sentence] = elt

    save_pickle(cache=cache, save_p=FRAME_CAUSATION_CACHE_P)
    return cache


class KGCausationRepresentationsConverter(KGRepresentationsConverter):
    """
    A specialized knowledge graph representations converter focused on causation relationships.

    This class inherits from KGRepresentationsConverter and provides functionality to convert
    cause-effect relationships into knowledge graph triples.

    Attributes:
        spo_cols (list): Column names for subject-predicate-object triples.
        ex (str): Base URI for the example namespace.
        rdfs (str): Base URI for the RDFS namespace.
        dbo (str): Base URI for the DBpedia ontology namespace.

    Methods:
        build_kg_one_event: Converts a single event and its detection results into a knowledge graph.
    """

    def __init__(self):
        super().__init__()
        self.triples_col = ["subject", "predicate", "object"]
        self.ex = "http://example.com/"
        self.rdfs = "http://www.w3.org/2000/01/rdf-schema#"
        self.dbo = "http://dbpedia.org/ontology/"

        self.queries = {
            "simple_rdf_with_properties": """
            SELECT DISTINCT ?event ?sub_event ?frame ?role ?role_type ?entity WHERE {
            ?event ?pred ?abstract .
            ?abstract nif:sentence ?sentence .
            ?sub_event wsj:fromDocument ?sentence ;
                    a wsj:CorpusEntry ;
                    wsj:onFrame ?frame ;
                    wsj:withmappedrole ?role .
            ?role wsj:withfnfe ?role_type .
            OPTIONAL {?role skos:related ?entity}
            VALUES ?pred {ex:abstract ex:hasOutcome}
            }
            """,
            "causation_no_frame": """
            PREFIX ex: <http://example.com/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
            PREFIX wsj: <http://www.w3.org/2005/01/wsj#>
            CONSTRUCT {
            ?event ex:hasOutcome ?sub_event .
            #?sub_event rdfs:label ?label ;
            #           ex:abstract ?abstract .
            }
            WHERE {
            ?event ex:hasOutcome ?sub_event .
            #?sub_event rdfs:label ?label ;
            #           ex:abstract ?abstract .
            ?sub_event nif:sentence ?sentence .
            FILTER NOT EXISTS {?e wsj:fromDocument ?sentence}
            }
            """
        }
        self.queries = {k: self.prefixes_query + "\n" + val for k, val in self.queries.items()}

    def get_simple_outcome(self, graph):
        """
        Extracts simple causation relations without frame semantics.
        
        Parameters
        ----------
        graph : rdflib.Graph
            The RDF graph to query
            
        Returns
        -------
        pandas.DataFrame
            DataFrame of causation triples with formatted URIs
        """

        res = graph.query(self.queries["causation_no_frame"])
        df = pd.DataFrame(data=res, columns=self.triples_col)
        df["."] = "."
        for col in self.triples_col:
            df[col] = df[col].apply(lambda x: f"<{x}>")
        return df

    def to_simple_rdf_prop(self, graph, mode: str = "causation"):
        """ Extend parent function by adding simple causation relations without frame semantics. """        
        output = super().to_simple_rdf_prop(graph=graph, mode=mode)
        df = self.get_simple_outcome(graph=graph)
        return pd.concat([output, df])
    
    def to_simple_rdf_sp(self, graph, cache, mode: str = "causation"):
        """ Extend parent function by adding simple causation relations without frame semantics. """ 
        output, cache = super().to_simple_rdf_sp(graph=graph, cache=cache, mode=mode)
        df = self.get_simple_outcome(graph=graph)
        return pd.concat([output, df]), cache
    
    def to_simple_rdf_reification(self, graph, statement_nb, mode: str = "causation"):
        """ Extend parent function by adding simple causation relations without frame semantics. """ 
        output, statement_nb = super().to_simple_rdf_reification(graph=graph, statement_nb=statement_nb, mode=mode)
        df = self.get_simple_outcome(graph=graph)
        return pd.concat([output, df]), statement_nb
    
    def to_hypergraph_bn(self, graph, mode: str = "causation"):
        """ Extend parent function by adding simple causation relations without frame semantics. """ 
        output = super().to_hypergraph_bn(graph=graph, mode=mode)
        res = graph.query(self.queries["causation_no_frame"])
        df = pd.DataFrame(data=res)
        df.columns = range(df.shape[1])
        return pd.concat([output, df], ignore_index=True)
    
    def to_hyper_relational_rdf_star(self, graph, mode: str = "causation"):
        """ Extend parent function by adding simple causation relations without frame semantics. """ 
        output = super().to_hyper_relational_rdf_star(graph=graph, mode=mode)
        res = graph.query(self.queries["causation_no_frame"])
        df = pd.DataFrame(data=res)
        df.columns = range(df.shape[1])
        return pd.concat([output, df], ignore_index=True)
    
    


def get_outcome(event, cache):
    """
    Retrieves the outcome of a given event from cache or DBpedia if not cached.
    
    This function checks if the event (with angle brackets removed) exists in the cache.
    If it does, it returns the cached result. Otherwise, it queries DBpedia using
    SPARQL, stores the result in the cache, updates the cache file, and returns the result.
    
    Parameters:
    ----------
    event : str
        The event to look up, expected to be enclosed in angle brackets (e.g., "<Some_Event>")
    cache : pandas.DataFrame
        DataFrame containing previously processed events with columns "subject" and "result"
        
    Returns:
    -------
    str
        The outcome/result of the event, or empty string if no result is found
    
    Notes:
    -----
    - The event string is processed by removing the first and last characters (angle brackets)
    - Cache is saved to a file defined by CAUSATION_TEXT_CACHE_P constant
    - DBpedia is queried using a template query QUERY_CAUSATION with the event substituted
    """
    if event[1:-1] in cache["subject"].values:  # already processed, removing the <>
        result = cache[cache["subject"] == event[1:-1]]["result"].values[0]
    else:  # retrieving result from DBpedia
        query = QUERY_CAUSATION.replace("to-replace", event)
        data = run_request(endpoint=DB_SPARQL_ENDPOINT, header=HEADERS_CSV,
                        query=query)
        if data.shape[0] == 0:
            result = "" # no result found 
        else:
            result = data["result"].values[0]
        cache.loc[cache.shape[0]] = {"subject": event[1:-1], "result": result}
        cache.to_csv(CAUSATION_TEXT_CACHE_P, index=False)
    return result


@click.command()
@click.argument('folder', type=click.Path(exists=True))
@click.argument('file_name')
@click.option('--add_text', is_flag=True, default=True, help="Whether to add text labels to the graph")
def main(folder, file_name, add_text):
    """
    From one graph extract all the causation graphs.
    Args:
        graph_p (str): The path to the input graph.
    Returns:
        rdflib.Graph: A graph containing all the causation relationships extracted from the input graph.
    The function performs the following steps:
    1. Retrieves events from the input graph.
    2. Initializes a cache and an empty graph.
    3. For each event:
        a. Checks if the event has already been processed and is in the cache.
        b. If not in the cache, queries DBpedia for the causation result.
        c. Updates the cache with the new result.
        d. Extracts outcomes from the result.
        e. Generates a subgraph for each outcome and adds it to the main graph.
    4. Returns the final graph containing all causation relationships.
    """
    graph_p = os.path.join(folder, file_name)
    graph_output_p = os.path.join(folder, file_name.replace(".nt", "_causation_text.nt"))
    if not os.path.exists(graph_output_p):
        if os.path.getsize(graph_p) > 0:  # run below only if file is not empty
            events = get_events(graph_p=graph_p)
            cache = get_cache()
            graph = init_graph(prefix_to_ns=PREFIX_TO_NS)
            for event in tqdm(events):
                # retrieving outcome (either already cached, or querying the endpoint)
                result = get_outcome(event=event, cache=cache)

                # caching outcome frames
                outcomes = extract_outcomes(res=result)
                logger.info(f"Processing outcomes: {outcomes}")
                cache_frames = get_frames(results=outcomes)

                # building graphs step by step
                for index, outcome in enumerate(outcomes):
                    id_result = f"{event[1:-1].split('/')[-1]}_Result{index}"
                    curr_graph = FS_KG_BUILDER(text_input=outcome, id_abstract=id_result,
                                               cache=cache_frames, event=event[1:-1], add_text=add_text)
                    graph += curr_graph

            graph.serialize(graph_output_p, format="nt")
            logger.info(f"Saved graph to {graph_output_p}")

        else:
            with open(graph_output_p, 'w', encoding='utf-8') as f:
                pass

if __name__ == '__main__':
    main()
